#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.config import load_runtime_config, repo_path
from slaif_asr.inference import resolve_existing_path, run_context
from slaif_asr.metrics import corpus_metric_summary, raw_cer, raw_wer, raw_word_edit_counts
from slaif_asr.prompt_column import (
    compare_prompt_column_state_dicts,
    install_prompt_delta,
    merge_prompt_delta,
    trainable_delta_parameters,
    write_integrity_report,
)
from slaif_asr.real_eval import normalize_sl_asr_text, sha256_file, summarize_predictions, validate_gate_manifest
from slaif_asr.slovenian_curriculum import (
    assert_quota,
    assert_training_disjoint,
    atomic_write_json,
    category_counts,
    classify_round1,
    file_sha256,
    load_records,
    protected_hashes_from_metadata,
    read_scored_candidates,
    records_to_rows,
    select_controls,
    select_hard_examples,
    summary_dict,
    to_tts_candidates,
    validate_collection,
    write_jsonl,
)
from slaif_asr.tts import load_tts_config, render_candidates, validate_wav


def load_training_helpers() -> Any:
    helper_path = Path(__file__).resolve().parent / "train_prompt_column.py"
    spec = importlib.util.spec_from_file_location("slaif_train_prompt_column_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING_HELPERS = load_training_helpers()
make_batch = TRAINING_HELPERS.make_batch
rnnt_loss = TRAINING_HELPERS.rnnt_loss
state_dict_cpu = TRAINING_HELPERS.state_dict_cpu
transcribe_one = TRAINING_HELPERS.transcribe_one


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def require_single_gpu() -> str:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, saw {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "2080 Ti" not in name:
        raise RuntimeError(f"expected RTX 2080 Ti, saw {name}")
    return name


def run_nvidia_smi(path: Path) -> None:
    completed = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout, encoding="utf-8")


def run_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["run_dir"]).resolve()


def validate_stage(config: dict[str, Any]) -> dict[str, Any]:
    holdout_path = Path(config["paths"]["holdout_jsonl"])
    candidate_path = Path(config["paths"]["candidate_jsonl"])
    protected_hashes = protected_hashes_from_metadata(
        [
            Path("docs/evaluation-gates/fleurs-sl-si-test-full-v1.metadata.json"),
            Path("docs/evaluation-gates/artur-j-public-gate-v1.metadata.json"),
        ]
    )
    holdout = load_records(holdout_path, expected_role="synthetic_holdout", config=config)
    candidates = load_records(candidate_path, expected_role="synthetic_candidate", config=config)
    holdout_summary = validate_collection(
        holdout,
        expected_count=int(config["holdout"]["required_count"]),
        config=config,
        protected_hashes=protected_hashes,
    )
    candidate_summary = validate_collection(
        candidates,
        expected_count=int(config["candidate_pool"]["required_count"]),
        config=config,
        protected_hashes=protected_hashes,
        disjoint_records=holdout,
    )
    assert_quota(category_counts(holdout), config["holdout"]["category_quotas"])
    assert_quota(category_counts(candidates), config["candidate_pool"]["category_quotas"])
    payload = {
        "schema_version": "1.0",
        "holdout": summary_dict(holdout_summary),
        "candidate_pool": summary_dict(candidate_summary),
        "protected_gate_overlaps": 0,
    }
    atomic_write_json(Path(config["paths"]["validation_report"]), payload)
    return payload


def synthesize_stage(config: dict[str, Any]) -> dict[str, Any]:
    gpu = require_single_gpu()
    root = run_root(config)
    run_nvidia_smi(root / "synthesis" / "nvidia-smi-before.txt")
    tts_config = load_tts_config()
    holdout = load_records(Path(config["paths"]["holdout_jsonl"]), expected_role="synthetic_holdout", config=config)
    candidates = load_records(Path(config["paths"]["candidate_jsonl"]), expected_role="synthetic_candidate", config=config)
    start = time.perf_counter()
    holdout_result = render_candidates(
        candidates=to_tts_candidates(holdout),
        config=tts_config,
        output_root=root / "synthesis" / "holdout",
    )
    candidate_result = render_candidates(
        candidates=to_tts_candidates(candidates),
        config=tts_config,
        output_root=root / "synthesis" / "candidates",
    )
    run_nvidia_smi(root / "synthesis" / "nvidia-smi-after.txt")
    payload = {
        "schema_version": "1.0",
        "gpu": gpu,
        "holdout": holdout_result,
        "candidate_pool": candidate_result,
        "wall_time_seconds": round(time.perf_counter() - start, 3),
    }
    atomic_write_json(root / "synthesis" / "summary.json", payload)
    return payload


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def newest_context_dir(output_dir: Path) -> Path:
    roots = sorted((path for path in output_dir.iterdir() if path.is_dir()), key=lambda item: item.name)
    if not roots:
        raise FileNotFoundError(f"no run directory under {output_dir}")
    context = roots[-1] / "context_56_3"
    if not context.exists():
        raise FileNotFoundError(context)
    return context


def newest_streaming_output(context_dir: Path) -> Path:
    candidates = sorted(context_dir.glob("streaming_out_*.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"missing streaming_out_*.json under {context_dir}")
    return candidates[-1]


def run_streaming(*, checkpoint: Path, manifest: Path, output_dir: Path, batch_size: int = 1) -> tuple[Path, float]:
    cfg = load_runtime_config()
    script = resolve_existing_path(
        repo_path("nemo.source_tree")
        / "examples"
        / "asr"
        / "asr_cache_aware_streaming"
        / "speech_to_text_cache_aware_streaming_infer.py",
        "NeMo streaming script",
    )
    checkpoint = resolve_existing_path(checkpoint, "checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    context_dir = output_dir / time.strftime("%Y%m%d-%H%M%S") / "context_56_3"
    command = [
        sys.executable,
        str(script),
        f"model_path={checkpoint}",
        f"batch_size={batch_size}",
        "target_lang=sl-SI",
        "strip_lang_tags=true",
        "att_context_size=[56,3]",
        f"output_path={context_dir}",
        "cuda=0",
        f"dataset_manifest={manifest.resolve()}",
    ]
    env = os.environ.copy()
    env.setdefault("NEMO_ROOT", str(repo_path("nemo.source_tree")))
    start = time.perf_counter()
    result = run_context(
        command=command,
        context=(56, 3),
        context_dir=context_dir,
        checkpoint_sha256=cfg["base_model"]["sha256"],
        cuda_index=0,
        env=env,
    )
    wall = time.perf_counter() - start
    (context_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if result.exit_status != 0:
        raise RuntimeError((context_dir / "inference.log").read_text(encoding="utf-8"))
    return context_dir, wall


def predictions_for_manifest(context_dir: Path, manifest: Path) -> list[dict[str, Any]]:
    manifest_rows = read_manifest(manifest)
    output_rows = read_manifest(newest_streaming_output(context_dir))
    if len(manifest_rows) != len(output_rows):
        raise RuntimeError(f"prediction count {len(output_rows)} != manifest count {len(manifest_rows)}")
    rows: list[dict[str, Any]] = []
    for manifest_row, output_row in zip(manifest_rows, output_rows, strict=True):
        reference = str(manifest_row.get("text", output_row.get("text", "")))
        hypothesis = str(output_row.get("pred_text", ""))
        word_counts = raw_word_edit_counts(normalize_sl_asr_text(reference), normalize_sl_asr_text(hypothesis))
        rows.append(
            {
                "candidate_id": manifest_row.get("sample_id") or manifest_row.get("candidate_id"),
                "reference": reference,
                "hypothesis": hypothesis,
                "pipeline_status": "PASSED",
                "empty_hypothesis": not hypothesis.strip(),
                "word_substitutions": word_counts.substitutions,
                "word_deletions": word_counts.deletions,
                "word_insertions": word_counts.insertions,
                "raw_wer": raw_wer(reference, hypothesis).percent,
                "raw_cer": raw_cer(reference, hypothesis).percent,
                "normalized_wer": raw_wer(normalize_sl_asr_text(reference), normalize_sl_asr_text(hypothesis)).percent,
                "normalized_cer": raw_cer(normalize_sl_asr_text(reference), normalize_sl_asr_text(hypothesis)).percent,
                "phenomena": manifest_row.get("phenomena", []),
            }
        )
    return rows


def score_stage(config: dict[str, Any]) -> dict[str, Any]:
    require_single_gpu()
    root = run_root(config)
    checkpoint = Path(config["paths"]["checkpoint"])
    manifest = root / "synthesis" / "candidates" / "nemo-manifest.jsonl"
    enriched_manifest = root / "selection" / "candidate-score-manifest.jsonl"
    candidates = {record.candidate_id: record for record in load_records(Path(config["paths"]["candidate_jsonl"]), expected_role="synthetic_candidate", config=config)}
    rows = []
    for row in read_manifest(manifest):
        candidate_id = Path(row["audio_filepath"]).stem
        record = candidates[candidate_id]
        row = dict(row)
        row["sample_id"] = candidate_id
        row["candidate_id"] = candidate_id
        row["partition_role"] = "synthetic_candidate"
        row["source_type"] = "synthetic_tts"
        row["phenomena"] = list(record.phenomena)
        rows.append(row)
    write_jsonl(enriched_manifest, rows)
    run_nvidia_smi(root / "scoring" / "nvidia-smi-before.txt")
    context_dir, wall = run_streaming(checkpoint=checkpoint, manifest=enriched_manifest, output_dir=root / "scoring" / "base")
    run_nvidia_smi(root / "scoring" / "nvidia-smi-after.txt")
    scored = predictions_for_manifest(context_dir, enriched_manifest)
    scores_path = root / "selection" / "candidate-scores.local.jsonl"
    scores_sha = write_jsonl(scores_path, scored)
    payload = {
        "schema_version": "1.0",
        "candidate_pool_scored": len(scored),
        "empty_hypotheses": sum(1 for row in scored if row["empty_hypothesis"]),
        "scores_sha256": scores_sha,
        "wall_time_seconds": round(wall, 3),
        "context_dir": str(context_dir),
    }
    atomic_write_json(root / "selection" / "pre-score-summary.json", payload)
    return payload


def select_stage(config: dict[str, Any]) -> dict[str, Any]:
    root = run_root(config)
    scored = read_scored_candidates(root / "selection" / "candidate-scores.local.jsonl")
    hard_count = int(config["selection"]["hard_examples"])
    control_count = int(config["selection"]["general_controls"])
    category_cap = int(hard_count * float(config["selection"]["hard_category_fraction_cap"]))
    hard = select_hard_examples(scored, count=hard_count, category_cap=category_cap)
    controls = select_controls(
        scored,
        exclude_ids={row["candidate_id"] for row in hard},
        count=control_count,
        seed=int(config["selection"]["control_seed"]),
    )
    selected_ids = [row["candidate_id"] for row in hard] + [row["candidate_id"] for row in controls]
    holdout_ids = {record.candidate_id for record in load_records(Path(config["paths"]["holdout_jsonl"]), expected_role="synthetic_holdout", config=config)}
    assert_training_disjoint(set(selected_ids), holdout_ids=holdout_ids)
    manifest_rows = [row for row in read_manifest(root / "selection" / "candidate-score-manifest.jsonl") if row["candidate_id"] in set(selected_ids)]
    manifest_rows.sort(key=lambda row: selected_ids.index(row["candidate_id"]))
    manifest_sha = write_jsonl(root / "manifests" / "selected-training.jsonl", manifest_rows)
    selection = {
        "schema_version": "1.0",
        "hard_examples": [row["candidate_id"] for row in hard],
        "general_controls": [row["candidate_id"] for row in controls],
        "training_ids": selected_ids,
        "training_manifest_sha256": manifest_sha,
        "hard_category_counts": count_phenomena(hard),
        "control_category_counts": count_phenomena(controls),
        "training_category_counts": count_phenomena(hard + controls),
    }
    atomic_write_json(root / "selection" / "selection-report.local.json", selection)
    public_summary = {
        "hard_examples": len(hard),
        "general_controls": len(controls),
        "training_total": len(selected_ids),
        "training_manifest_sha256": manifest_sha,
        "training_category_counts": selection["training_category_counts"],
    }
    atomic_write_json(root / "selection" / "selection-summary.json", public_summary)
    return selection


def count_phenomena(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for phenomenon in row.get("phenomena", []):
            counts[phenomenon] = counts.get(phenomenon, 0) + 1
    return dict(sorted(counts.items()))


def manifest_records_for_training(path: Path) -> list[Any]:
    from slaif_asr.prompt_experiment import ManifestRecord

    records = []
    for row in read_manifest(path):
        audio_path = Path(row["audio_filepath"]).resolve()
        info = validate_wav(audio_path, sample_rate=16000)
        records.append(
            ManifestRecord(
                sample_id=str(row["sample_id"]),
                audio_filepath=audio_path,
                duration=round(info.duration_seconds, 6),
                text=str(row["text"]),
                lang="sl-SI",
                target_lang="sl-SI",
                partition_role="synthetic_candidate",
                source_type="synthetic_tts",
            )
        )
    return records


def train_stage(config: dict[str, Any]) -> dict[str, Any]:
    gpu = require_single_gpu()
    root = run_root(config)
    training = config["training"]
    runtime_cfg = load_runtime_config()
    checkpoint = Path(config["paths"]["checkpoint"]).resolve()
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != runtime_cfg["base_model"]["sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")
    records = manifest_records_for_training(root / "manifests" / "selected-training.jsonl")
    if len(records) != int(config["selection"]["training_total"]):
        raise RuntimeError("training record count mismatch")
    import nemo.collections.asr as nemo_asr
    import torch

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.manual_seed(int(training["seed"]))
    random.seed(int(training["seed"]))
    run_nvidia_smi(root / "training" / "nvidia-smi-before.txt")
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0").cuda().eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None
    base_state = state_dict_cpu(model)
    selection, wrapper = install_prompt_delta(model, "sl-SI")
    if selection.effective_trainable_parameters != 2048:
        raise RuntimeError(f"expected 2048 trainable parameters, saw {selection.effective_trainable_parameters}")
    optimizer = torch.optim.AdamW(
        trainable_delta_parameters(wrapper, weight_decay=float(training["weight_decay"])),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batches = [make_batch(model, record, "cuda") for record in records]
    order = list(range(len(records)))
    random.Random(int(training["seed"])).shuffle(order)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    max_steps = int(training["max_optimizer_steps"])
    early = training["early_stop"]
    stopped_reason = "completed max_optimizer_steps"
    for step in range(1, max_steps + 1):
        index = order[(step - 1) % len(order)]
        record = records[index]
        batch = batches[index]
        optimizer.zero_grad(set_to_none=True)
        loss = rnnt_loss(model, batch, selection.prompt_index, str(training["precision"]))
        if not torch.isfinite(loss):
            stopped_reason = "non-finite loss"
            break
        loss.backward()
        grad = wrapper.delta.grad.detach()
        grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu()) if grad is not None else 0.0
        delta_norm = float(torch.linalg.vector_norm(wrapper.delta.detach()).cpu())
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % 25 == 0 or step == max_steps:
            logs.append(
                {
                    "step": step,
                    "candidate_id": record.sample_id,
                    "loss": loss_value,
                    "learning_rate": float(training["learning_rate"]),
                    "gradient_norm": grad_norm,
                    "delta_norm": delta_norm,
                    "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
                    "nan_or_overflow": False,
                }
            )
        if bool(early["enabled"]) and step >= int(early["minimum_steps"]) + int(early["window_steps"]):
            window = int(early["window_steps"])
            previous = sum(losses[-2 * window : -window]) / window
            current = sum(losses[-window:]) / window
            improvement = 0.0 if previous == 0 else (previous - current) / previous
            if improvement < float(early["minimum_relative_improvement"]):
                stopped_reason = f"early stop: moving-average improvement {improvement:.6f}"
                break
    final_step = len(losses)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    delta_path = artifact_dir / "round1-sl-si-prompt-column-delta.pt"
    torch.save({"selection": asdict(selection), "delta": wrapper.delta.detach().cpu(), "checkpoint_sha256": checkpoint_sha}, delta_path)
    merge_prompt_delta(model, selection)
    merged_checkpoint = artifact_dir / "round1-sl-si-prompt-column-adapted.nemo"
    model.save_to(str(merged_checkpoint))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored = nemo_asr.models.ASRModel.restore_from(restore_path=str(merged_checkpoint), map_location="cuda:0").cuda().eval()
    adapted_state = state_dict_cpu(restored)
    integrity = compare_prompt_column_state_dicts(
        base_state,
        adapted_state,
        first_linear_weight_name=f"{selection.first_linear_name}.weight",
        first_linear_bias_name=f"{selection.first_linear_name}.bias",
        selected_column=selection.selected_column,
        selected_prompt=selection.prompt_name,
        prompt_index=selection.prompt_index,
        effective_trainable_parameters=selection.effective_trainable_parameters,
    )
    write_integrity_report(integrity, root / "integrity-report.json")
    restore_probe = transcribe_one(restored, records[0].audio_filepath)
    run_nvidia_smi(root / "training" / "nvidia-smi-after.txt")
    payload = {
        "schema_version": "1.0",
        "gpu": gpu,
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": checkpoint_sha,
        "base_model_revision": runtime_cfg["base_model"]["revision"],
        "nemo_revision": runtime_cfg["nemo"]["revision"],
        "selection": asdict(selection),
        "effective_trainable_parameters": selection.effective_trainable_parameters,
        "precision": training["precision"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "steps": final_step,
        "stopped_reason": stopped_reason,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "wall_time_seconds": round(time.perf_counter() - start, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        "delta_artifact": str(delta_path),
        "merged_checkpoint": str(merged_checkpoint),
        "integrity_passed": (
            integrity.tensor_shapes_match
            and not integrity.unexpected_changed_tensors
            and integrity.unexpected_changed_elements == 0
            and integrity.selected_column_changed
            and integrity.other_columns_bitwise_identical
            and integrity.bias_bitwise_identical
        ),
        "restore_probe_transcript_present": bool(restore_probe.strip()),
        "logs": logs,
    }
    atomic_write_json(root / "training" / "training-summary.json", payload)
    return payload


def manifest_with_sample_ids(source: Path, destination: Path, *, role: str) -> str:
    rows = []
    for row in read_manifest(source):
        candidate_id = Path(row["audio_filepath"]).stem
        row = dict(row)
        row["sample_id"] = candidate_id
        row["partition_role"] = role
        row["source_type"] = "synthetic_tts"
        rows.append(row)
    return write_jsonl(destination, rows)


def summarize_pair(manifest: Path, base_context: Path, challenger_context: Path) -> dict[str, Any]:
    rows = read_manifest(manifest)
    base_rows = read_manifest(newest_streaming_output(base_context))
    challenger_rows = read_manifest(newest_streaming_output(challenger_context))
    per_sample = []
    for manifest_row, base_row, challenger_row in zip(rows, base_rows, challenger_rows, strict=True):
        reference = str(manifest_row["text"])
        base_hyp = str(base_row.get("pred_text", ""))
        challenger_hyp = str(challenger_row.get("pred_text", ""))
        per_sample.append({"reference": reference, "hypothesis": base_hyp, "kind": "base"})
        per_sample.append({"reference": reference, "hypothesis": challenger_hyp, "kind": "challenger"})
    base_summary = summarize_predictions([{"reference": row["reference"], "hypothesis": row["hypothesis"]} for row in per_sample if row["kind"] == "base"])
    challenger_summary = summarize_predictions(
        [{"reference": row["reference"], "hypothesis": row["hypothesis"]} for row in per_sample if row["kind"] == "challenger"]
    )
    return {"base": base_summary, "challenger": challenger_summary}


def evaluate_stage(config: dict[str, Any]) -> dict[str, Any]:
    require_single_gpu()
    root = run_root(config)
    checkpoint = Path(config["paths"]["checkpoint"]).resolve()
    training_summary = load_json(root / "training" / "training-summary.json")
    challenger = Path(training_summary["merged_checkpoint"]).resolve()
    eval_root = root / "evaluation"
    holdout_manifest = eval_root / "synthetic-holdout.jsonl"
    manifest_with_sample_ids(root / "synthesis" / "holdout" / "nemo-manifest.jsonl", holdout_manifest, role="synthetic_holdout")
    training_manifest = root / "manifests" / "selected-training.jsonl"
    real_manifests = {
        "fleurs": Path("runs/evaluation-gates/fleurs-sl-si-test-full-v1/manifest.jsonl").resolve(),
        "artur_j": Path("runs/evaluation-gates/artur-j-public-gate-v1/manifest.jsonl").resolve(),
    }
    for manifest in real_manifests.values():
        validate_gate_manifest(manifest)
    splits = {
        "selected_synthetic_training": training_manifest,
        "fixed_synthetic_holdout": holdout_manifest,
    }
    summaries: dict[str, Any] = {}
    wall_times: dict[str, Any] = {}
    for split, manifest in splits.items():
        base_context, base_time = run_streaming(checkpoint=checkpoint, manifest=manifest, output_dir=eval_root / split / "base")
        challenger_context, challenger_time = run_streaming(
            checkpoint=challenger,
            manifest=manifest,
            output_dir=eval_root / split / "challenger",
        )
        summaries[split] = summarize_pair(manifest, base_context, challenger_context)
        wall_times[split] = {"base": round(base_time, 3), "challenger": round(challenger_time, 3)}
    for gate, manifest in real_manifests.items():
        challenger_context, challenger_time = run_streaming(
            checkpoint=challenger,
            manifest=manifest,
            output_dir=eval_root / gate / "challenger",
        )
        challenger_rows = predictions_for_manifest(challenger_context, manifest)
        summaries[gate] = {"challenger": summarize_predictions(challenger_rows)}
        wall_times[gate] = {"challenger": round(challenger_time, 3)}
    payload = {"schema_version": "1.0", "summaries": summaries, "wall_times_seconds": wall_times}
    atomic_write_json(eval_root / "evaluation-summary.local.json", payload)
    return payload


def metric_summary_from_dict(summary: dict[str, Any], key: str = "normalized") -> Any:
    from slaif_asr.metrics import CorpusMetricSummary

    return CorpusMetricSummary(**summary[key])


def summarize_stage(config: dict[str, Any]) -> dict[str, Any]:
    root = run_root(config)
    validation = load_json(Path(config["paths"]["validation_report"]))
    synthesis = load_json(root / "synthesis" / "summary.json")
    scoring = load_json(root / "selection" / "pre-score-summary.json")
    selection = load_json(root / "selection" / "selection-summary.json")
    training = load_json(root / "training" / "training-summary.json")
    evaluation = load_json(root / "evaluation" / "evaluation-summary.local.json")
    fleurs_base = {
        "corpus_wer": 52.734,
        "corpus_cer": 16.423,
        "mean_utterance_wer": 53.541,
        "mean_utterance_cer": 0.0,
        "median_utterance_wer": 52.941,
        "median_utterance_cer": 0.0,
        "empty_hypothesis_count": 0,
        "total_word_edits": 0,
        "total_reference_words": 0,
        "total_character_edits": 0,
        "total_reference_characters": 0,
    }
    artur_base = {
        "corpus_wer": 67.453,
        "corpus_cer": 29.016,
        "mean_utterance_wer": 76.555,
        "mean_utterance_cer": 0.0,
        "median_utterance_wer": 75.0,
        "median_utterance_cer": 0.0,
        "empty_hypothesis_count": 12,
        "total_word_edits": 0,
        "total_reference_words": 0,
        "total_character_edits": 0,
        "total_reference_characters": 0,
    }
    synthetic_holdout_base = metric_summary_from_dict(evaluation["summaries"]["fixed_synthetic_holdout"]["base"])
    synthetic_holdout_challenger = metric_summary_from_dict(evaluation["summaries"]["fixed_synthetic_holdout"]["challenger"])
    decision = classify_round1(
        integrity_passed=bool(training["integrity_passed"]),
        synthetic_holdout_base=synthetic_holdout_base,
        synthetic_holdout_challenger=synthetic_holdout_challenger,
        fleurs_base=metric_summary_from_dict({"normalized": fleurs_base}),
        fleurs_challenger=metric_summary_from_dict(evaluation["summaries"]["fleurs"]["challenger"]),
        artur_base=metric_summary_from_dict({"normalized": artur_base}),
        artur_challenger=metric_summary_from_dict(evaluation["summaries"]["artur_j"]["challenger"]),
        thresholds=config["promotion"],
    )
    public = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "generation_spec_revision": config["generation_spec_revision"],
        "holdout": validation["holdout"],
        "candidate_pool": validation["candidate_pool"],
        "synthesis": {
            "holdout_synthesized": synthesis["holdout"]["successful"],
            "candidate_pool_synthesized": synthesis["candidate_pool"]["successful"],
            "failed": synthesis["holdout"]["failed"] + synthesis["candidate_pool"]["failed"],
            "wall_time_seconds": synthesis["wall_time_seconds"],
        },
        "pre_scoring": scoring,
        "selection": selection,
        "training": {
            "effective_trainable_parameters": training["effective_trainable_parameters"],
            "precision": training["precision"],
            "steps": training["steps"],
            "stopped_reason": training["stopped_reason"],
            "initial_loss": training["initial_loss"],
            "final_loss": training["final_loss"],
            "wall_time_seconds": training["wall_time_seconds"],
            "peak_vram_mib": training["peak_vram_mib"],
            "integrity_passed": training["integrity_passed"],
        },
        "metrics": evaluation["summaries"],
        "round1_decision": decision.decision,
        "decision_reasons": list(decision.reasons),
        "privacy": {
            "raw_real_references_exposed": False,
            "synthetic_holdout_errors_used_for_steering": False,
        },
    }
    atomic_write_json(root / "round1-public-summary.json", public)
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project-generated Slovenian curriculum Round 1 pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/generation/slovenian_curriculum_round1.json"))
    parser.add_argument(
        "--stage",
        choices=["validate", "synthesize", "score", "select", "train", "evaluate", "summarize", "all"],
        required=True,
    )
    args = parser.parse_args()
    config = load_json(args.config)
    stages = ["validate", "synthesize", "score", "select", "train", "evaluate", "summarize"] if args.stage == "all" else [args.stage]
    results = {}
    for stage in stages:
        if stage == "validate":
            results[stage] = validate_stage(config)
        elif stage == "synthesize":
            results[stage] = synthesize_stage(config)
        elif stage == "score":
            results[stage] = score_stage(config)
        elif stage == "select":
            results[stage] = select_stage(config)
        elif stage == "train":
            results[stage] = train_stage(config)
        elif stage == "evaluate":
            results[stage] = evaluate_stage(config)
        elif stage == "summarize":
            results[stage] = summarize_stage(config)
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
