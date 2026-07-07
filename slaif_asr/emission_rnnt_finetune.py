from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.batched_streaming import StreamingRecord, file_sha256, load_gate_records, metrics_for, resolve_manifest_audio_path
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_training import TrainingRecord
from slaif_asr.data_quality import sha256_file
from slaif_asr.real_eval import atomic_write_json, atomic_write_jsonl
from slaif_asr.scale2000_corpus import burden as real_regression_burden
from slaif_asr.tts import validate_wav


ARM_NAME = "scale2000_augmented_decoder_joint_rnnt"
EXPECTED_TEXT_SHA256 = "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14"
EXPECTED_ALL_VIEWS_SHA256 = "9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca"
EXPECTED_SCHEDULE_SHA256 = "6757018f3306839ce8564ba758e13e231ab4784bf98049b65701b963b55e5842"
BASE_DIRECTIONAL_METRICS = {
    "piper_synthetic_holdout": {"wer": 86.025, "cer": 46.762, "empty": 17},
    "supertonic_heldout_voice_holdout": {"wer": 58.307, "cer": 27.712, "empty": 32},
    "fleurs_v2": {"wer": 52.685, "cer": 16.406, "empty": 1},
    "artur_j": {"wer": 67.322, "cer": 28.620, "empty": 12},
}
SCALE2000_JOINT_ADAPTER_METRICS = {
    "piper_synthetic_holdout": {"wer": 55.435, "cer": 20.073, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 27.407, "cer": 7.597, "empty": 0},
    "fleurs_v2": {"wer": 51.589, "cer": 16.238, "empty": 0},
    "artur_j": {"wer": 60.114, "cer": 20.630, "empty": 0},
}
PUBLIC_FORBIDDEN_KEYS = {
    "audio_filepath",
    "candidate_id",
    "candidate_ids",
    "holdout_id",
    "holdout_ids",
    "hypothesis",
    "hypotheses",
    "local_path",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "semantic_key",
    "selected_training_id",
    "text",
    "target_text",
    "spoken_text",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "gamsv",
    "gams9holdout-",
    "fleurs-sl-si-test-occ-",
    "artur-j-public-",
    "/" + "home" + "/",
    "/" + "tmp" + "/",
    "/" + "data-nvme" + "/",
    ".wav",
)


@dataclass(frozen=True)
class TrainableSurfaceSummary:
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]
    forbidden_modules_present: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def local_runs_root() -> Path:
    override = os.environ.get("SLAIF_ASR_RUNS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs"


def local_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "runs":
        return local_runs_root().joinpath(*parts[1:])
    return REPO_ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_config(path: str | Path = "configs/experiments/scale2000_decoder_joint_rnnt_v1.json") -> dict[str, Any]:
    config = read_json(repo_path(path))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0030":
        raise ValueError("work_order_id must be 0030")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    data = config["data"]
    expected_data = {
        "semantic_rows": 16000,
        "view_records": 320000,
        "clean_files": 144000,
        "augmented_files": 176000,
        "fixed_text_sha256": EXPECTED_TEXT_SHA256,
        "all_views_sha256": EXPECTED_ALL_VIEWS_SHA256,
        "exposure_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected!r}")
    training = config["training"]
    expected_training = {
        "semantic_rows": 16000,
        "sample_exposures": 320000,
        "effective_batch_size": 8,
        "optimizer_steps": 40000,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_clipping": "none",
        "seed": 1234,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("physical_microbatch_candidates") != [8, 4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [8, 4, 2, 1]")
    surface = config["trainable_surface"]
    if tuple(surface.get("allowed_prefixes", ())) != ("decoder.", "joint."):
        raise ValueError("trainable surface must be decoder+joint only")
    if surface.get("text_only_path_allowed") is not False:
        raise ValueError("text-only path must be forbidden")
    evaluation = config["evaluation"]
    if evaluation.get("batch_size") != 32 or evaluation.get("duration_bucketing") is not True:
        raise ValueError("directional evaluation must use batch size 32 with duration bucketing")
    if evaluation.get("canonical") is not False or evaluation.get("promotion_eligible") is not False:
        raise ValueError("evaluation must remain noncanonical and promotion-ineligible")


def run_dir(config: dict[str, Any]) -> Path:
    return local_path(config["local_outputs"]["run_root"])


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def protected_file_fingerprints(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for path_text in config["protected_files"]:
        path = repo_path(path_text)
        rel = path.relative_to(REPO_ROOT).as_posix()
        completed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
        output[path_text] = {"git_blob_sha": completed.stdout.strip(), "byte_sha256": file_sha256(path)}
    return output


def verify_protected_file_fingerprints(config: dict[str, Any], expected: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    current = protected_file_fingerprints(config)
    changed = [path for path, fingerprint in expected.items() if current.get(path) != fingerprint]
    if changed:
        raise RuntimeError(f"protected files changed: {changed}")
    return current


def verify_committed_scale2000_evidence(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    audio_cert_path = repo_path(data["audio_certificate"])
    experiment_cert_path = repo_path(data["experiment_certificate"])
    report_path = repo_path(data["experiment_report"])
    audio_cert = read_json(audio_cert_path)
    experiment_cert = read_json(experiment_cert_path)
    report = read_json(report_path)
    if audio_cert.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError("scale-2000 audio certificate is not AUDIO_ACCEPTED")
    if experiment_cert.get("status") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("scale-2000 experiment certificate is not DIAGNOSTIC_ONLY")
    if report.get("directional_evaluation", {}).get("decision", {}).get("classification") != "SCALE2000_TEXT_REAL_GAIN_DIRECTIONAL":
        raise RuntimeError("Experiment 0014 classification mismatch")
    if report.get("accepted_parent") != "none":
        raise RuntimeError("Experiment 0014 accepted_parent mismatch")
    if audio_cert.get("fixed_text_sha256") != EXPECTED_TEXT_SHA256:
        raise RuntimeError("scale-2000 text SHA mismatch in audio certificate")
    if audio_cert.get("hashes", {}).get("all_views_sha256") != EXPECTED_ALL_VIEWS_SHA256:
        raise RuntimeError("scale-2000 all-views SHA mismatch in audio certificate")
    if audio_cert.get("schedule", {}).get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("scale-2000 schedule SHA mismatch in audio certificate")
    metrics = report["directional_evaluation"]["metric_table"]["scale2000_joint_adapter"]
    for split, expected in SCALE2000_JOINT_ADAPTER_METRICS.items():
        for metric in ("wer", "cer", "empty"):
            if round(float(metrics[split][metric]), 3) != float(expected[metric]):
                raise RuntimeError(f"Experiment 0014 {split}.{metric} mismatch")
    return {
        "audio_certificate_sha256": file_sha256(audio_cert_path),
        "experiment_certificate_sha256": file_sha256(experiment_cert_path),
        "experiment_report_sha256": file_sha256(report_path),
        "classification": report["directional_evaluation"]["decision"]["classification"],
        "accepted_parent": report["accepted_parent"],
    }


def verify_local_scale2000_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    fixed_text = local_path(data["fixed_text"])
    all_views = local_path(data["all_views"])
    schedule = local_path(data["exposure_schedule"])
    validation = local_path(data["audio_validation"])
    for path in (fixed_text, all_views, schedule, validation):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(fixed_text) != EXPECTED_TEXT_SHA256:
        raise RuntimeError("local scale-2000 fixed text SHA mismatch")
    if sha256_file(all_views) != EXPECTED_ALL_VIEWS_SHA256:
        raise RuntimeError("local scale-2000 all-views SHA mismatch")
    if sha256_file(schedule) != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("local scale-2000 exposure schedule SHA mismatch")
    validation_payload = read_json(validation)
    if validation_payload.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError("local scale-2000 audio validation is not AUDIO_ACCEPTED")
    rows = {"fixed_text_rows": len(read_jsonl(fixed_text)), "all_view_rows": len(read_jsonl(all_views)), "schedule_rows": len(read_jsonl(schedule))}
    if rows != {"fixed_text_rows": 16000, "all_view_rows": 320000, "schedule_rows": 320000}:
        raise RuntimeError(f"local scale-2000 row counts mismatch: {rows}")
    return {
        **rows,
        "fixed_text_sha256": EXPECTED_TEXT_SHA256,
        "all_views_sha256": EXPECTED_ALL_VIEWS_SHA256,
        "exposure_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }


def verify_all_inputs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "committed_evidence": verify_committed_scale2000_evidence(config),
        "local_artifacts": verify_local_scale2000_artifacts(config),
        "protected_file_fingerprints": protected_file_fingerprints(config),
    }


def configure_decoder_joint_trainable(model: Any) -> TrainableSurfaceSummary:
    forbidden = [name for name, _ in model.named_modules() if "adapter" in name.lower() or "lm_head" in name.lower() or "lm_adapter" in name.lower()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    decoder_count = 0
    joint_count = 0
    for name, parameter in model.named_parameters():
        if "adapter_layer" in name or "lm_adapter" in name or "lm_head" in name:
            raise RuntimeError(f"forbidden training module present: {name}")
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            decoder_count += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            joint_count += parameter.numel()
    if decoder_count <= 0 or joint_count <= 0:
        raise RuntimeError("decoder and joint parameters must both be present")
    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad and not (name.startswith("decoder.") or name.startswith("joint."))]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected[:10]}")
    frozen_count = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    return TrainableSurfaceSummary(
        trainable_parameter_count=decoder_count + joint_count,
        decoder_parameter_count=decoder_count,
        joint_parameter_count=joint_count,
        frozen_parameter_count=frozen_count,
        trainable_prefixes=("decoder.", "joint."),
        forbidden_modules_present=tuple(forbidden),
    )


def trainable_parameters(model: Any) -> list[Any]:
    return [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and (name.startswith("decoder.") or name.startswith("joint."))]


def verify_optimizer_scope(optimizer: Any, model: Any) -> None:
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in trainable_parameters(model)}
    if actual != expected:
        raise RuntimeError("optimizer contains parameters outside decoder+joint")


def optimizer_scope_summary(model: Any) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "trainable_count": len(names),
        "all_decoder_or_joint": all(name.startswith("decoder.") or name.startswith("joint.") for name in names),
        "contains_text_only_lm": any("lm_head" in name or "lm_adapter" in name for name in names),
        "contains_adapter": any("adapter_layer" in name for name in names),
        "sample_trainable_names": names[:20],
    }


def finite_grad_norm(parameters: Sequence[Any]) -> tuple[float, bool]:
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


def changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    changed = []
    for name in sorted(set(before) & set(after)):
        left = before[name]
        right = after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    allowed = [name for name in changed if name.startswith("decoder.") or name.startswith("joint.")]
    disallowed = [name for name in changed if name not in allowed]
    return {
        "changed_tensor_count": len(changed),
        "allowed_changed_tensor_count": len(allowed),
        "unexpected_changed_tensors": disallowed,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "encoder_unchanged": not any(name.startswith("encoder.") for name in changed),
        "prompt_kernel_unchanged": not any("prompt" in name.lower() for name in changed),
        "only_decoder_joint_changed": not missing and not unexpected and not disallowed,
        "sample_changed_tensors": changed[:50],
    }


def has_forbidden_text_only_modules(model: Any) -> bool:
    names = [name for name, _ in model.named_modules()]
    return any("lm_head" in name or "lm_adapter" in name or "decoder_lm" in name for name in names)


def rnnt_audio_loss(model: Any, batch: tuple[Any, Any, Any, Any], prompt_index: int, *, frozen_encoder_no_grad: bool):
    import torch

    signal, signal_len, transcript, transcript_len = batch
    prompt_indices = torch.full((signal.shape[0],), prompt_index, dtype=torch.long, device=signal.device)
    if frozen_encoder_no_grad:
        with torch.no_grad():
            encoded, encoded_len = model.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                prompt_indices=prompt_indices,
            )
        encoded = encoded.detach()
        encoded_len = encoded_len.detach()
    else:
        encoded, encoded_len = model.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            prompt_indices=prompt_indices,
        )
    decoder, target_length, _ = model.decoder(targets=transcript, target_length=transcript_len)
    if model.joint.fuse_loss_wer:
        loss_value, _, _, _ = model.joint(
            encoder_outputs=encoded,
            decoder_outputs=decoder,
            encoder_lengths=encoded_len,
            transcripts=transcript,
            transcript_lengths=target_length,
            compute_wer=False,
        )
    else:
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = model.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
    return model.add_auxiliary_losses(loss_value)


def microbatch_plan(physical_microbatch: int) -> dict[str, int]:
    if physical_microbatch not in {1, 2, 4, 8}:
        raise ValueError("physical microbatch must be one of 1, 2, 4, 8")
    return {"physical_microbatch": physical_microbatch, "gradient_accumulation_steps": 8 // physical_microbatch, "effective_batch_size": 8}


def validate_microbatch_selection(candidates: Sequence[int], outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if list(candidates) != [8, 4, 2, 1]:
        raise ValueError("microbatch candidates must be [8, 4, 2, 1]")
    for candidate in candidates:
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "ENVIRONMENT_BLOCKED",
        "reason": "physical microbatch 1 failed" if outcomes.get(1, {}).get("status") == "FAILED" else "no passing physical microbatch",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def _load_text_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(local_path(config["data"]["fixed_text"]))
    if len(rows) != 16000:
        raise RuntimeError(f"expected 16000 fixed text rows, found {len(rows)}")
    return {str(row["candidate_id"]): row for row in rows}


def _view_lookup(config: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows = read_jsonl(local_path(config["data"]["all_views"]))
    if len(rows) != 320000:
        raise RuntimeError(f"expected 320000 all-view rows, found {len(rows)}")
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"]))
        if key in lookup:
            raise RuntimeError(f"duplicate view key: {key}")
        lookup[key] = row
    return lookup


def training_record_from_view(text_row: dict[str, Any], view_row: dict[str, Any], *, reason: str) -> TrainingRecord:
    path = Path(str(view_row["audio_filepath"]))
    if not path.exists():
        raise FileNotFoundError(path)
    text = str(text_row["target_text"])
    if str(view_row["target_text_sha256"]) != stable_sha256(text):
        raise RuntimeError("scale-2000 text/audio text-hash mismatch")
    return TrainingRecord(
        selected_training_id=str(text_row["candidate_id"]),
        audio_filepath=str(path),
        duration=float(view_row["duration_seconds"]),
        text=text,
        text_sha256=str(view_row["target_text_sha256"]),
        audio_sha256=str(view_row["audio_sha256"]),
        selection_reason=reason,
        selection_rank=int(text_row["generation"]["prompt_cell"].removeprefix("cell")) if str(text_row["generation"]["prompt_cell"]).startswith("cell") else 0,
    )


def load_scheduled_round_records(config: dict[str, Any]) -> tuple[dict[int, list[TrainingRecord]], dict[str, dict[str, Any]], dict[str, Any]]:
    text_by_id = _load_text_by_id(config)
    views = _view_lookup(config)
    schedule = read_jsonl(local_path(config["data"]["exposure_schedule"]))
    if len(schedule) != 320000:
        raise RuntimeError(f"expected 320000 exposure schedule rows, found {len(schedule)}")
    rounds: dict[int, list[TrainingRecord]] = defaultdict(list)
    meta_by_audio: dict[str, dict[str, Any]] = {}
    seen = set()
    for item in schedule:
        semantic_key = str(item["semantic_key"])
        view_key = (semantic_key, str(item["view_type"]), str(item["voice"]), str(item["profile_id"]))
        if view_key not in views:
            raise RuntimeError(f"schedule references missing view: {view_key}")
        if str(item["voice"]) in {"supertonic-M5", "supertonic-F5", "M5", "F5"}:
            raise RuntimeError("held-out Supertonic voice leaked into training schedule")
        round_key = (int(item["round"]), semantic_key)
        if round_key in seen:
            raise RuntimeError(f"duplicate semantic item in round {round_key[0]}")
        seen.add(round_key)
        record = training_record_from_view(text_by_id[semantic_key], views[view_key], reason="scale2000_decoder_joint_rnnt")
        rounds[int(item["round"])].append(record)
        meta_by_audio[record.audio_filepath] = {"voice": item["voice"], "profile_id": item["profile_id"], "view_type": item["view_type"], "spec_augment": bool(item.get("spec_augment", False))}
    for round_index in range(1, 21):
        if len(rounds[round_index]) != 16000:
            raise RuntimeError(f"round {round_index} has {len(rounds[round_index])} rows, expected 16000")
    return rounds, meta_by_audio, {"schedule_sha256": sha256_file(local_path(config["data"]["exposure_schedule"]))}


def probe_records(config: dict[str, Any]) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    from slaif_asr.corpus_v2_training import select_probe_records

    text_by_id = _load_text_by_id(config)
    views = _view_lookup(config)
    inherited = []
    all_clean = []
    for semantic_key in sorted(text_by_id):
        view = views[(semantic_key, "clean", "piper-sl_SI-artur-medium", "clean")]
        record = training_record_from_view(text_by_id[semantic_key], view, reason="scale2000_decoder_joint_probe")
        all_clean.append(record)
        if semantic_key.startswith("gamsv3-"):
            inherited.append(record)
    if len(inherited) != 1600:
        raise RuntimeError("anchor probe requires 1600 inherited scale-200 rows")
    return select_probe_records(inherited, 32), select_probe_records(all_clean, 320)


def directional_suite(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]]]:
    # Reuse the already-tested scale-8000 directional-suite helper when present,
    # because Work Orders 0028 and 0030 use the same fixed validation suite.
    from slaif_asr.scale8000_clean_training import directional_suite as scale8000_directional_suite

    suite, split_records = scale8000_directional_suite(
        {
            "data": {
                "piper_synthetic_holdout_manifest": "runs/data-quality/sl-corpus-v2-independent-holdout-v1/audio-manifest.local.jsonl",
                "piper_synthetic_holdout_rows": 96,
                "supertonic_heldout_manifest": "runs/data-quality/sl-corpus-v2-supertonic3-multivoice-v1/audio-manifest.local.jsonl",
                "supertonic_heldout_rows": 192,
                "fleurs_v2_manifest": "runs/gates/fleurs-sl-si-test-full-v2/manifest.local.jsonl",
                "fleurs_v2_manifest_sha256": "8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513",
                "fleurs_v2_rows": 834,
                "artur_j_manifest": "runs/gates/artur-j-public-gate-v1/manifest.local.jsonl",
                "artur_j_manifest_sha256": "66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763",
                "artur_j_rows": 256,
            }
        }
    )
    return suite, split_records


def metric_row(split_summary: dict[str, Any]) -> dict[str, Any]:
    normalized = split_summary["metrics"]["normalized"]
    raw = split_summary["metrics"]["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "raw_wer": round(float(raw["corpus_wer"]), 3),
        "raw_cer": round(float(raw["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
    }


def classify_decoder_joint_rnnt(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    piper_gain = (
        float(metrics["piper_synthetic_holdout"]["wer"]) < BASE_DIRECTIONAL_METRICS["piper_synthetic_holdout"]["wer"]
        or float(metrics["piper_synthetic_holdout"]["cer"]) < BASE_DIRECTIONAL_METRICS["piper_synthetic_holdout"]["cer"]
    )
    supertonic_gain = (
        float(metrics["supertonic_heldout_voice_holdout"]["wer"]) < BASE_DIRECTIONAL_METRICS["supertonic_heldout_voice_holdout"]["wer"]
        or float(metrics["supertonic_heldout_voice_holdout"]["cer"]) < BASE_DIRECTIONAL_METRICS["supertonic_heldout_voice_holdout"]["cer"]
    )
    real_burden = real_regression_burden(metrics, BASE_DIRECTIONAL_METRICS)
    real_improvements = 0
    no_real_metric_worse = True
    real_within_half = True
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            delta = float(metrics[split][metric]) - float(SCALE2000_JOINT_ADAPTER_METRICS[split][metric])
            if delta < 0:
                real_improvements += 1
            if delta > 0.5:
                no_real_metric_worse = False
            if abs(delta) > 0.5:
                real_within_half = False
    synthetic_no_more_than_one_worse = True
    synthetic_within_one = True
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout"):
        for metric in ("wer", "cer"):
            delta = float(metrics[split][metric]) - float(SCALE2000_JOINT_ADAPTER_METRICS[split][metric])
            if delta > 1.0:
                synthetic_no_more_than_one_worse = False
            if abs(delta) > 1.0:
                synthetic_within_one = False
    if piper_gain and supertonic_gain and real_burden == 0.0 and real_improvements >= 2 and no_real_metric_worse and synthetic_no_more_than_one_worse:
        classification = "DECODER_JOINT_RNNT_BEATS_SCALE2000_DIRECTIONAL"
    elif piper_gain and supertonic_gain and real_burden == 0.0 and real_within_half and synthetic_within_one:
        classification = "DECODER_JOINT_RNNT_MATCHES_SCALE2000_DIRECTIONAL"
    elif piper_gain and supertonic_gain and real_burden == 0.0:
        classification = "DECODER_JOINT_RNNT_BEATS_BASE_BUT_NOT_SCALE2000"
    else:
        classification = "DECODER_JOINT_RNNT_SYNTHETIC_ONLY_OR_REGRESSES"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_holdout_gain": piper_gain,
        "supertonic_holdout_gain": supertonic_gain,
        "real_burden": real_burden,
        "real_metrics_improved_vs_scale2000": real_improvements,
        "no_real_metric_more_than_half_point_worse_than_scale2000": no_real_metric_worse,
        "synthetic_holdouts_no_more_than_one_point_worse_than_scale2000": synthetic_no_more_than_one_worse,
    }


def assert_public_report_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in PUBLIC_FORBIDDEN_KEYS:
                    raise ValueError(f"public report contains forbidden key: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    for marker in PUBLIC_FORBIDDEN_MARKERS:
        if marker in serialized:
            raise ValueError(f"public report contains forbidden marker: {marker}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, rows)
