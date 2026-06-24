#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.batched_streaming import (
    NvidiaSmiMonitor,
    file_sha256,
    load_local_predictions,
    parse_monitor_csv,
    privacy_safe_arm_summary,
)
from slaif_asr.corpus_v2_training import (
    DIAGNOSTIC_CERTIFICATE_PATH,
    DIAGNOSTIC_STATUS,
    EXPECTED_TRAINABLE_PARAMETERS,
    assert_epoch_covers_once,
    assert_public_report_safe,
    classify_batching,
    classify_scientific,
    compare_batched_loss_to_individual,
    deterministic_epoch_batches,
    evaluate_prompt_column_integrity,
    git_head,
    load_experiment_config,
    load_real_gate_eval_records,
    load_synthetic_eval_records,
    load_training_records,
    make_training_batch,
    metric_pair,
    optimizer_parameter_ids,
    original_state_dict_from_prompt_delta_model,
    parameter_integrity_before_merge,
    repo_path,
    rnnt_loss,
    run_dir,
    run_evaluation_arm,
    runtime_summary,
    select_probe_records,
    selection_from_benchmark,
    state_dict_cpu,
    verify_all_input_identities,
    verify_diagnostic_certificate,
    write_markdown_report,
)
from slaif_asr.corpus_v2_scoring import (
    ATT_CONTEXT_SIZE,
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NEMO_REVISION,
    checkpoint_path,
    verify_runtime_identities,
)
from slaif_asr.prompt_column import install_prompt_delta, merge_prompt_delta, trainable_delta_parameters
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json


DEFAULT_CONFIG = Path("configs/experiments/corpus_v2_prompt_column_diagnostic_v1.json")


def require_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def restore_base_model(config: dict[str, Any]):
    import nemo.collections.asr as nemo_asr

    checkpoint = repo_path(config["model"]["checkpoint_path"]).resolve()
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0")
    model = model.cuda().eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None
    return model


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def finite_grad_norm(parameters: list[Any]) -> tuple[float, bool]:
    import torch

    total = 0.0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all())
        total += float(torch.sum(grad * grad).detach().cpu())
    return total**0.5, finite


def mean_loss(model: Any, selection: Any, records: list[Any], *, device: str) -> float:
    import torch

    losses = []
    with torch.no_grad():
        for record in records:
            loss = rnnt_loss(model, make_training_batch(model, [record], device=device), selection.prompt_index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def benchmark_one(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    torch = configure_torch()
    records = select_probe_records(load_training_records(config), int(config["batch_benchmark"]["subset_rows"]))
    model = restore_base_model(config)
    base_state = state_dict_cpu(model)
    selection, wrapper = install_prompt_delta(model, "sl-SI")
    if selection.effective_trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("unexpected trainable parameter count")
    correctness_records = records[:batch_size]
    correctness = compare_batched_loss_to_individual(model, selection, correctness_records, device="cuda")
    correctness["passed"] = (
        correctness["finite"]
        and float(correctness["relative_difference"]) <= float(config["batch_benchmark"]["loss_relative_tolerance"])
    )
    if not correctness["passed"]:
        payload = {"batch_size": batch_size, "status": "FAILED", "correctness": correctness}
        write_json(run_dir(config) / "batch-benchmark" / f"batch-{batch_size}.local.json", payload)
        return payload

    optimizer_params = trainable_delta_parameters(wrapper, weight_decay=0)
    monitor_path = run_dir(config) / "batch-benchmark" / f"batch-{batch_size}-gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    layout = deterministic_epoch_batches(records, batch_size=batch_size, epoch=0, seed=int(config["training"]["seed"]), bucketed=True)
    assert_epoch_covers_once(layout, len(records))
    batches = [[records[index] for index in batch] for batch in layout.batches]

    warmup = int(config["batch_benchmark"]["warmup_cycles"])
    timed = int(config["batch_benchmark"]["timed_cycles"])
    finite = True
    grad_finite = True
    audio_seconds = 0.0
    examples = 0
    torch.cuda.reset_peak_memory_stats(0)
    monitor.start()
    try:
        for cycle in range(warmup + timed):
            batch_records = batches[cycle % len(batches)]
            for parameter in optimizer_params:
                parameter.grad = None
            loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), selection.prompt_index)
            finite = finite and bool(torch.isfinite(loss))
            loss.backward()
            _grad_norm, grads_ok = finite_grad_norm(optimizer_params)
            grad_finite = grad_finite and grads_ok
            torch.cuda.synchronize()
            if cycle == warmup - 1:
                start = time.perf_counter()
                audio_seconds = 0.0
                examples = 0
            if cycle >= warmup:
                audio_seconds += sum(row.duration for row in batch_records)
                examples += len(batch_records)
    finally:
        monitor.stop()
    wall = time.perf_counter() - start
    reserved_mib = torch.cuda.max_memory_reserved(0) / 1024 / 1024
    allocated_mib = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    pre_merge = parameter_integrity_before_merge(base_state, original_state_dict_from_prompt_delta_model(model, selection), selection=selection)
    status = "PASSED"
    if not finite or not grad_finite or reserved_mib > float(config["batch_benchmark"]["max_peak_reserved_mib"]):
        status = "FAILED"
    payload = {
        "batch_size": batch_size,
        "status": status,
        "correctness": correctness,
        "cycles": {"warmup": warmup, "timed": timed},
        "examples": examples,
        "audio_seconds": round(audio_seconds, 6),
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(examples / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "padding_ratio": layout.padding_ratio,
        "finite_loss": finite,
        "finite_gradients": grad_finite,
        "peak_allocated_mib": round(allocated_mib, 3),
        "peak_reserved_mib": round(reserved_mib, 3),
        "monitor": parse_monitor_csv(monitor_path),
        "pre_merge_integrity": pre_merge,
    }
    write_json(run_dir(config) / "batch-benchmark" / f"batch-{batch_size}.local.json", payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def run_benchmark(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_diagnostic_certificate(config_path)
    verify_runtime_identities(check_gpu=True)
    rows = []
    for batch_size in config["batch_benchmark"]["candidate_batch_sizes"]:
        command = [sys.executable, __file__, "--config", str(config_path), "--stage", "benchmark-one", "--batch-size", str(batch_size)]
        completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        (run_dir(config) / "batch-benchmark" / f"batch-{batch_size}.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            rows.append({"batch_size": batch_size, "status": "FAILED", "log": f"batch-{batch_size}.log"})
        else:
            summary_path = run_dir(config) / "batch-benchmark" / f"batch-{batch_size}.local.json"
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        if rows[-1].get("status") != "PASSED" and "out of memory" in completed.stdout.lower():
            break
        if float(rows[-1].get("peak_reserved_mib", 0.0)) > float(config["batch_benchmark"]["max_peak_reserved_mib"]):
            break
    selection = selection_from_benchmark(rows, within_best_fraction=float(config["batch_benchmark"]["within_best_fraction"]))
    payload = {"status": "PASSED", "arms": rows, "selection": selection}
    write_json(run_dir(config) / "batch-benchmark" / "summary.local.json", payload)
    return payload


def train_arm(config: dict[str, Any], config_path: Path, *, arm_name: str, batch_size: int, bucketed: bool) -> dict[str, Any]:
    verify_diagnostic_certificate(config_path)
    runtime = verify_runtime_identities(check_gpu=True)
    torch = configure_torch()
    records = load_training_records(config)
    model = restore_base_model(config)
    model.eval()
    base_state = state_dict_cpu(model)
    selection, wrapper = install_prompt_delta(model, "sl-SI")
    if selection.effective_trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("unexpected prompt-column trainable count")
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable != [(selection.first_linear_name + ".delta", EXPECTED_TRAINABLE_PARAMETERS)]:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}")
    optimizer = torch.optim.AdamW(trainable_delta_parameters(wrapper, weight_decay=0), lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != optimizer_parameter_ids(wrapper):
        raise RuntimeError("optimizer contains parameters other than the prompt-column delta")

    probe_records = select_probe_records(records, int(config["training"]["probe_rows"]))
    initial_probe = mean_loss(model, selection, probe_records, device="cuda")
    initial_full = mean_loss(model, selection, records, device="cuda")
    probe_curve = [{"epoch": 0, "mean_loss": round(initial_probe, 6)}]
    delta_norm_curve = []
    grad_norms = []
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    arm_dir = run_dir(config) / arm_name
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    monitor.start()
    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            layout = deterministic_epoch_batches(records, batch_size=batch_size, epoch=epoch, seed=int(config["training"]["seed"]), bucketed=bucketed)
            assert_epoch_covers_once(layout, len(records))
            for batch_indices in layout.batches:
                batch_records = [records[index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), selection.prompt_index)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                grad_norm, grads_ok = finite_grad_norm(trainable_delta_parameters(wrapper, weight_decay=0))
                if not grads_ok:
                    raise RuntimeError("non-finite prompt-column gradient")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                grad_norms.append(grad_norm)
            probe_loss = mean_loss(model, selection, probe_records, device="cuda")
            probe_curve.append({"epoch": epoch, "mean_loss": round(probe_loss, 6)})
            delta_norm_curve.append({"epoch": epoch, "delta_norm": round(float(torch.linalg.vector_norm(wrapper.delta.detach()).cpu()), 6)})
    finally:
        monitor.stop()
    wall = time.perf_counter() - start
    final_probe = mean_loss(model, selection, probe_records, device="cuda")
    final_full = mean_loss(model, selection, records, device="cuda")
    pre_merge = parameter_integrity_before_merge(base_state, original_state_dict_from_prompt_delta_model(model, selection), selection=selection)
    delta_path = arm_dir / "artifacts" / "prompt-column-delta.pt"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "1.0", "selection": selection.__dict__, "delta": wrapper.delta.detach().cpu()}, delta_path)
    merge_prompt_delta(model, selection)
    merged_state = state_dict_cpu(model)
    merged_integrity = evaluate_prompt_column_integrity(base_state, merged_state, selection=selection)
    checkpoint_out = arm_dir / "artifacts" / f"{arm_name}.nemo"
    model.save_to(str(checkpoint_out))
    verify_command = [sys.executable, __file__, "--config", str(config_path), "--stage", "verify-checkpoint", "--arm", arm_name]
    completed = subprocess.run(verify_command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
    (arm_dir / "verify-checkpoint.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{arm_name}: restored checkpoint integrity failed")
    restored_report_path = arm_dir / "restore-integrity.local.json"
    if not restored_report_path.exists():
        raise RuntimeError(f"{arm_name}: restored checkpoint integrity report is missing")
    restored_integrity = json.loads(restored_report_path.read_text(encoding="utf-8"))
    monitor = parse_monitor_csv(monitor_path)
    payload = {
        "arm": arm_name,
        "status": "PASSED",
        "batch_size": batch_size,
        "duration_bucketing": bucketed,
        "epochs": int(config["training"]["epochs"]),
        "sample_exposures": sample_exposures,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(config["training"]["learning_rate"]),
        "initial_probe_loss": round(initial_probe, 6),
        "final_probe_loss": round(final_probe, 6),
        "initial_full_training_loss": round(initial_full, 6),
        "final_full_training_loss": round(final_full, 6),
        "full_loss_reduction_percent": round((initial_full - final_full) / initial_full * 100.0, 6) if initial_full else None,
        "probe_curve": probe_curve,
        "gradient_norm": {
            "min": round(min(grad_norms), 6),
            "max": round(max(grad_norms), 6),
            "final": round(grad_norms[-1], 6),
        },
        "delta_norm_curve": delta_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "gpu_monitor": monitor,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "selection": selection.__dict__,
        "trainable_parameter_count": selection.effective_trainable_parameters,
        "pre_merge_integrity": pre_merge,
        "integrity": merged_integrity,
        "restored_checkpoint_integrity": restored_integrity,
        "checkpoint_local_sha256": file_sha256(checkpoint_out),
        "delta_artifact_local_sha256": file_sha256(delta_path),
        "runtime": runtime,
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def verify_checkpoint(config: dict[str, Any], *, arm_name: str) -> dict[str, Any]:
    configure_torch()
    base = restore_base_model(config)
    base_state = state_dict_cpu(base)
    selection = None
    from slaif_asr.prompt_column import derive_prompt_column_selection

    selection = derive_prompt_column_selection(base, "sl-SI")
    del base
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    import nemo.collections.asr as nemo_asr

    checkpoint = run_dir(config) / arm_name / "artifacts" / f"{arm_name}.nemo"
    restored = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0").cuda().eval()
    restored_state = state_dict_cpu(restored)
    report = evaluate_prompt_column_integrity(base_state, restored_state, selection=selection)
    report["checkpoint_restored"] = True
    write_json(run_dir(config) / arm_name / "restore-integrity.local.json", report)
    return report


def selected_batch_or_none(config: dict[str, Any]) -> int | None:
    path = run_dir(config) / "batch-benchmark" / "summary.local.json"
    if not path.exists():
        raise RuntimeError("batch benchmark summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    selected = summary.get("selection", {}).get("selected_batch_size")
    return None if selected is None else int(selected)


def evaluate_all(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_diagnostic_certificate(config_path)
    verify_runtime_identities(check_gpu=True)
    splits = {
        "selected_training": load_synthetic_eval_records(config, "selected_training"),
        "synthetic_holdout": load_synthetic_eval_records(config, "synthetic_holdout"),
        "fleurs_v2": load_real_gate_eval_records(config, "fleurs_v2"),
        "artur_j": load_real_gate_eval_records(config, "artur_j"),
    }
    models: dict[str, Path] = {"base": checkpoint_path()}
    for arm_name in ("reference_batch1", "a100_batched"):
        ckpt = run_dir(config) / arm_name / "artifacts" / f"{arm_name}.nemo"
        if ckpt.exists():
            models[arm_name] = ckpt
    output: dict[str, Any] = {
        "status": "PASSED",
        "models": {},
        "evaluation_policy": {
            "batch_size": 1,
            "duration_bucketing": False,
            "att_context_size": ATT_CONTEXT_SIZE,
            "target_lang": config["evaluation"]["target_lang"],
            "normalizer": NORMALIZER_VERSION,
        },
    }
    for model_name, ckpt in models.items():
        output["models"][model_name] = {"checkpoint_sha256": file_sha256(ckpt), "splits": {}}
        for split_name, records in splits.items():
            arm = run_evaluation_arm(
                records=records,
                checkpoint=ckpt,
                run_dir=run_dir(config) / "evaluation" / model_name / split_name,
                python_executable=Path(sys.executable),
            )
            output["models"][model_name]["splits"][split_name] = {
                "rows": int(arm["rows"]),
                "prediction_count": int(arm["prediction_count"]),
                "audio_duration_seconds": arm["audio_duration_seconds"],
                "wall_time_seconds": arm["execution"]["wall_time_seconds"],
                "real_time_factor": arm["end_to_end_real_time_factor"],
                "metrics": arm["metrics"],
                "gpu_monitor": arm["execution"]["monitor"],
            }
    write_json(run_dir(config) / "evaluation" / "summary.local.json", output)
    return output


def summarize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    auth = verify_diagnostic_certificate(config_path)
    input_integrity = verify_all_input_identities(config, check_gpu=False)
    benchmark_path = run_dir(config) / "batch-benchmark" / "summary.local.json"
    training: dict[str, Any] = {}
    for arm_name in ("reference_batch1", "a100_batched"):
        path = run_dir(config) / arm_name / "training-summary.local.json"
        if path.exists():
            arm = json.loads(path.read_text(encoding="utf-8"))
            training[arm_name] = {
                key: arm[key]
                for key in (
                    "status",
                    "batch_size",
                    "duration_bucketing",
                    "epochs",
                    "sample_exposures",
                    "optimizer_steps",
                    "learning_rate",
                    "initial_probe_loss",
                    "final_probe_loss",
                    "initial_full_training_loss",
                    "final_full_training_loss",
                    "full_loss_reduction_percent",
                    "probe_curve",
                    "gradient_norm",
                    "delta_norm_curve",
                    "wall_time_seconds",
                    "examples_per_second",
                    "audio_seconds_per_wall_second",
                    "gpu_monitor",
                    "peak_allocated_mib",
                    "peak_reserved_mib",
                    "selection",
                    "trainable_parameter_count",
                    "pre_merge_integrity",
                    "integrity",
                    "restored_checkpoint_integrity",
                )
                if key in arm
            }
    evaluation = json.loads((run_dir(config) / "evaluation" / "summary.local.json").read_text(encoding="utf-8"))
    valid_arms = [name for name in ("reference_batch1", "a100_batched") if name in evaluation["models"]]
    public = {
        "schema_version": "1.0",
        "experiment_id": "corpus-v2-prompt-column-diagnostic-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "authorization": {
            "status": auth["certificate"]["status"],
            "sha256": file_sha256(DIAGNOSTIC_CERTIFICATE_PATH),
            "work_order_id": auth["certificate"]["work_order_id"],
            "tracked_before_training": auth["tracked"],
        },
        "runtime": runtime_summary(),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "input_integrity": {
            "selected_training": input_integrity["selected_training"],
            "synthetic_holdout_audio_manifest_sha256": input_integrity["synthetic_holdout_audio_manifest_sha256"],
            "synthetic_holdout_rows": input_integrity["synthetic_holdout_rows"],
            "candidate_holdout_overlap_counts": input_integrity["candidate_holdout_overlap_counts"],
        },
        "batch_benchmark": json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None,
        "training": training,
        "evaluation": evaluation,
        "decisions": {},
        "accepted_parent": "none",
        "limitations": [
            "Single-voice synthetic training.",
            "No real training or calibration speech.",
            "Synthetic holdout is not real-generalization evidence.",
            "Development real gates are not a final blind test.",
        ],
    }
    public["decisions"]["scientific"] = classify_scientific(evaluation, valid_arms)
    public["decisions"]["batching"] = classify_batching(public)
    assert_public_report_safe(public)
    json_path = Path("docs/experiments/0008-corpus-v2-prompt-column-diagnostic.json")
    md_path = Path("docs/experiments/0008-corpus-v2-prompt-column-diagnostic.md")
    write_json(json_path, public)
    write_markdown_report(md_path, public)
    result = {
        "status": "PASSED",
        "json_sha256": file_sha256(json_path),
        "markdown_sha256": file_sha256(md_path),
        "scientific_classification": public["decisions"]["scientific"]["classification"],
        "batching_classification": public["decisions"]["batching"]["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corpus-v2 prompt-column diagnostic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--arm")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.stage in {"verify", "benchmark-batch", "benchmark-one", "train-reference", "train-batched", "evaluate", "verify-checkpoint"}:
        require_env()
    if args.stage == "verify":
        payload = verify_diagnostic_certificate(args.config)
        payload["gpu_runtime"] = verify_runtime_identities(check_gpu=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.stage == "benchmark-one":
        if args.batch_size is None:
            parser.error("--batch-size is required for benchmark-one")
        payload = benchmark_one(config, args.batch_size)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if payload["status"] == "PASSED" else 1
    if args.stage == "benchmark-batch":
        payload = run_benchmark(config, args.config)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.stage == "train-reference":
        payload = train_arm(config, args.config, arm_name="reference_batch1", batch_size=1, bucketed=False)
        print(json.dumps({"status": payload["status"], "arm": payload["arm"]}, indent=2, sort_keys=True))
        return 0
    if args.stage == "train-batched":
        selected = selected_batch_or_none(config)
        if selected is None or selected <= 1:
            print(json.dumps({"status": "SKIPPED", "reason": "no valid batch size above 1"}, indent=2, sort_keys=True))
            return 0
        payload = train_arm(config, args.config, arm_name="a100_batched", batch_size=selected, bucketed=True)
        print(json.dumps({"status": payload["status"], "arm": payload["arm"], "batch_size": selected}, indent=2, sort_keys=True))
        return 0
    if args.stage == "verify-checkpoint":
        if not args.arm:
            parser.error("--arm is required for verify-checkpoint")
        payload = verify_checkpoint(config, arm_name=args.arm)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if payload.get("passed") else 1
    if args.stage == "evaluate":
        payload = evaluate_all(config, args.config)
        print(json.dumps({"status": payload["status"], "models": sorted(payload["models"])}, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        summarize(config, args.config)
        return 0
    parser.error(f"unsupported stage: {args.stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
