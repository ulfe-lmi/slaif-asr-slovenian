from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.batched_streaming import StreamingRecord, file_sha256
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, sha256_file
from slaif_asr.emission_rnnt_finetune import (
    BASE_DIRECTIONAL_METRICS,
    EXPECTED_ALL_VIEWS_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPECTED_TEXT_SHA256,
    SCALE2000_JOINT_ADAPTER_METRICS,
    read_json,
    read_jsonl,
    repo_path,
    resolve_local_artifact_path,
    stable_sha256,
    training_record_from_view,
    verify_committed_scale2000_evidence,
    verify_local_scale2000_artifacts,
    protected_file_fingerprints,
)
from slaif_asr.s6tts_augmentation import EXPECTED_PROFILE_IDS, load_profiles
from slaif_asr.s6tts_tts import (
    S6Paths,
    S6TextRow,
    load_s6_config,
    local_path as s6_local_path,
    s6_paths,
    synthesize_worker_status,
    write_manifest_pair,
)
from slaif_asr.scale2000_corpus import burden as real_regression_burden
from slaif_asr.transcript_preserving_augmentation import parameters_for_profile, render_augmented_file
from slaif_asr.tts import validate_wav, write_jsonl


ARM_NAME = "scale2000_s6tts_hardvoice20_decoder_joint_rnnt"
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "scale2000_decoder_joint_rnnt_s6tts_hardvoice20.json"
SCHEDULE_CERTIFICATE = REPO_ROOT / "docs" / "data-certificates" / "sl-corpus-v4-s6tts-hardvoice20-schedule-v1.json"
EXPERIMENT_CERTIFICATE = REPO_ROOT / "docs" / "data-certificates" / "sl-corpus-v4-decoder-joint-rnnt-s6tts-hardvoice20-diagnostic-v1.json"
REPORT_JSON = REPO_ROOT / "docs" / "experiments" / "0023-scale2000-decoder-joint-rnnt-s6tts-hardvoice20.json"
REPORT_MD = REPO_ROOT / "docs" / "experiments" / "0023-scale2000-decoder-joint-rnnt-s6tts-hardvoice20.md"

S6_CLEAN_SHA = "355a85134e81d9e3ea4089ea9a941f62fb101902b4e151c394eaaf1d1de416d5"
S6_AUG_SHA = "8d39606dc276a7730e032e83c1811f6c71ece3de6f0b68aa1bd5f4c0a8f50251"
S6_AUG_PROV_SHA = "d18a2c8245e75d94d18c97ae02a3194ddcdc8fdf864bc44cc21fcec8941603ec"
S6_VOICE = "s6tts-sl-si-s6-vintage"
S6_REPLACEMENT_ROUNDS = (5, 10, 15, 20)
HOLDOUT_TEXT_SHA = "078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5"
HOLDOUT_CERTIFICATE = REPO_ROOT / "docs" / "data-certificates" / "sl-corpus-v2-s6tts-hardvoice-holdout-v1.json"
HOLDOUT_REPORT_JSON = REPO_ROOT / "docs" / "data-reports" / "0022-s6tts-hardvoice-holdout-admission.json"
HOLDOUT_REPORT_MD = REPO_ROOT / "docs" / "data-reports" / "0022-s6tts-hardvoice-holdout-admission.md"

PR36_DECODER_JOINT_METRICS = {
    "piper_synthetic_holdout": {"wer": 34.317, "cer": 13.765, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 14.752, "cer": 4.682, "empty": 0},
    "fleurs_v2": {"wer": 46.195, "cer": 15.604, "empty": 0},
    "artur_j": {"wer": 56.793, "cer": 20.177, "empty": 0},
}
PR39_ROUND6_METRICS = {
    "piper_synthetic_holdout": {"wer": 36.788, "cer": 13.918, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 17.137, "cer": 5.344, "empty": 0},
    "fleurs_v2": {"wer": 48.781, "cer": 16.845, "empty": 0},
    "artur_j": {"wer": 58.258, "cer": 22.246, "empty": 0},
}

PUBLIC_FORBIDDEN_KEYS = {
    "audio_filepath",
    "candidate_id",
    "candidate_ids",
    "hypothesis",
    "hypotheses",
    "local_path",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "semantic_key",
    "selected_training_id",
    "spoken_text",
    "target_text",
    "text",
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


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(repo_path(path))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0036":
        raise ValueError("work_order_id must be 0036")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    data = config["data"]
    expected_data = {
        "semantic_rows": 16000,
        "fixed_text_sha256": EXPECTED_TEXT_SHA256,
        "base_scale2000_all_views_sha256": EXPECTED_ALL_VIEWS_SHA256,
        "base_scale2000_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "s6_clean_manifest_sha256": S6_CLEAN_SHA,
        "s6_augmented_manifest_sha256": S6_AUG_SHA,
        "s6_augmented_provenance_manifest_sha256": S6_AUG_PROV_SHA,
        "s6tts_revision": "6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c",
        "s6tts_runtime_data_hash": "2e71b1ed2df53b7959fa748d9cd1366478895202a874d884ed3986abb581e6dc",
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected!r}")
    schedule = config["schedule"]
    expected_schedule = {
        "schedule_id": "scale2000_plus_s6tts_hardvoice_20pct_v1",
        "semantic_rows": 16000,
        "exposures_per_semantic_row": 20,
        "total_exposures": 320000,
        "original_scale2000_exposures_per_row": 16,
        "original_scale2000_exposures": 256000,
        "s6tts_exposures_per_row": 4,
        "s6tts_total_exposures": 64000,
        "s6tts_share": 0.2,
        "s6tts_clean_exposures_per_row": 1,
        "s6tts_augmented_exposures_per_row": 3,
    }
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise ValueError(f"schedule.{key} must be {expected!r}")
    if tuple(schedule.get("s6tts_replacement_rounds", ())) != S6_REPLACEMENT_ROUNDS:
        raise ValueError("unexpected S6TTS replacement rounds")
    training = config["training"]
    expected_training = {
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
        "early_stopping": True,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("physical_microbatch_candidates") != [8, 4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [8, 4, 2, 1]")
    if training.get("retain_per_round_checkpoints") is not True:
        raise ValueError("per-round checkpoints must be retained locally for controller-dev run-control")
    if training.get("early_stopping_partition") != "artur-controller-dev-v1":
        raise ValueError("early stopping must use artur-controller-dev-v1")
    surface = config["trainable_surface"]
    if tuple(surface.get("allowed_prefixes", ())) != ("decoder.", "joint."):
        raise ValueError("trainable surface must be decoder+joint only")
    if surface.get("text_only_path_allowed") is not False:
        raise ValueError("text-only path must be forbidden")
    evaluation = config["evaluation"]
    if evaluation.get("batch_size") != 32 or evaluation.get("duration_bucketing") is not True:
        raise ValueError("directional evaluation must use batch-32 with bucketing")
    if evaluation.get("canonical") is not False or evaluation.get("promotion_eligible") is not False:
        raise ValueError("evaluation must be directional and promotion-ineligible")
    controller = config["controller_dev"]
    if controller.get("partition_id") != "artur-controller-dev-v1":
        raise ValueError("controller-dev partition must be artur-controller-dev-v1")
    if controller.get("manifest_sha256") != "7944cbd82107e4aa8cfd3c5ca991d652e4ec3450ba8805efbc98e7c3aeec34f9":
        raise ValueError("controller-dev manifest SHA mismatch in config")
    if controller.get("batch_size") != 1 or controller.get("duration_bucketing") is not False:
        raise ValueError("controller-dev run-control must use batch-1 without bucketing")
    if controller.get("allowed_for") != "aggregate_run_control_and_early_stopping_only":
        raise ValueError("controller-dev use must be aggregate run-control only")
    early_stop = config["early_stop_rule"]
    if early_stop.get("operational_stop_rule") != "stop_after_three_evaluated_rounds_without_new_raw_best_controller_dev_wer":
        raise ValueError("unexpected S6 hardvoice early-stop rule")
    if int(early_stop.get("patience_rounds_without_new_raw_best", 0)) != 3:
        raise ValueError("S6 hardvoice controller-dev patience must be 3 rounds")


def should_stop_for_controller_dev(config: dict[str, Any], rows: Sequence[dict[str, Any]]) -> bool:
    post = [row for row in rows if int(row["round"]) > 0]
    min_rounds = int(config["early_stop_rule"].get("min_rounds_before_stop", 0))
    patience = int(config["early_stop_rule"]["patience_rounds_without_new_raw_best"])
    if len(post) < min_rounds:
        return False
    best_wer = float("inf")
    best_position = -1
    for position, row in enumerate(post):
        wer = float(row["wer"])
        if wer < best_wer:
            best_wer = wer
            best_position = position
    return len(post) - best_position - 1 >= patience


def run_dir(config: dict[str, Any]) -> Path:
    return s6_local_path(config["local_outputs"]["run_root"])


def schedule_path(config: dict[str, Any]) -> Path:
    return s6_local_path(config["local_outputs"]["schedule"])


def schedule_summary_path(config: dict[str, Any]) -> Path:
    return s6_local_path(config["local_outputs"]["schedule_summary"])


def holdout_root() -> Path:
    return s6_local_path("runs/data-quality/sl-corpus-v2-s6tts-hardvoice-holdout-v1")


def holdout_clean_paths() -> S6Paths:
    base = s6_paths(load_s6_config())
    root = holdout_root() / "clean"
    return S6Paths(
        source_dir=base.source_dir,
        build_dir=base.build_dir,
        cli_path=base.cli_path,
        runtime_ini=base.runtime_ini,
        run_root=root,
        audio_manifest=root / "audio-manifest.local.jsonl",
        provenance_manifest=root / "provenance.local.jsonl",
        validation=root / "audio-validation.local.json",
        summary=root / "summary.local.json",
        logs_dir=root / "logs",
    )


def holdout_augmented_manifest() -> Path:
    return holdout_root() / "augmented" / "audio-manifest.local.jsonl"


def holdout_augmented_provenance() -> Path:
    return holdout_root() / "augmented" / "provenance.local.jsonl"


def holdout_summary_path() -> Path:
    return holdout_root() / "summary.local.json"


def _small_distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
    }


def _holdout_text_path() -> Path:
    configured = s6_local_path("runs/data-quality/sl-corpus-v2-independent-synthetic-holdout-v1/accepted-holdout.local.jsonl")
    if configured.exists():
        return configured
    fallback = REPO_ROOT / "runs" / "data-quality" / "sl-corpus-v2-independent-synthetic-holdout-v1" / "accepted-holdout.local.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("accepted independent synthetic holdout text is unavailable")


def _load_holdout_text_rows() -> list[dict[str, Any]]:
    path = _holdout_text_path()
    if sha256_file(path) != HOLDOUT_TEXT_SHA:
        raise RuntimeError("independent synthetic holdout text SHA mismatch")
    rows = read_jsonl(path)
    if len(rows) != 96:
        raise RuntimeError("expected 96 independent synthetic holdout rows")
    return rows


def _holdout_s6_rows() -> list[S6TextRow]:
    rows = []
    for index, row in enumerate(_load_holdout_text_rows()):
        target_text = str(row["target_text"])
        rows.append(
            S6TextRow(
                index=index,
                source_id=str(row.get("source_id", f"s6-hardvoice-holdout-source-{index:05d}")),
                source_family_id=str(row.get("source_family_id", "s6-hardvoice-holdout")),
                utterance_family_id=str(row.get("utterance_family_id", f"s6-hardvoice-holdout-{index:05d}")),
                text_hash=stable_sha256(target_text),
                spoken_text=str(row.get("spoken_text", target_text)),
                target_text=target_text,
                partition_role="synthetic_diagnostic_holdout",
            )
        )
    return rows


def prepare_s6_hardvoice_holdout(config: dict[str, Any]) -> dict[str, Any]:
    if config["hardvoice_holdout"]["source_text_sha256"] != HOLDOUT_TEXT_SHA:
        raise RuntimeError("hardvoice holdout source SHA mismatch in config")
    rows = _holdout_s6_rows()
    paths = holdout_clean_paths()
    clean_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        status = synthesize_worker_status((paths, row, False))
        if status.get("ok"):
            local_row = dict(status["row"])
            local_row["view_id"] = "sl-corpus-v2-s6tts-hardvoice-clean-holdout-v1"
            local_row["partition_role"] = "synthetic_diagnostic_holdout"
            clean_rows.append(local_row)
        else:
            failures.append(status["failure"])
    if failures:
        write_jsonl(holdout_root() / "failures.local.jsonl", failures)
        raise RuntimeError(f"S6 hardvoice clean holdout synthesis failed for {len(failures)} rows")
    write_manifest_pair(paths, clean_rows)
    profiles = load_profiles()
    augmented_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    aug_root = holdout_root() / "augmented"
    for clean_index, clean_row in enumerate(clean_rows):
        for profile_index, profile in enumerate(profiles, start=1):
            profile_id = str(profile["profile_id"])
            safe_key = f"s6-hardvoice-holdout-{clean_index:05d}__aug{profile_index:02d}"
            shard = f"{clean_index // 1000:02d}"
            output = aug_root / "wav" / f"aug{profile_index:02d}-{profile_id}" / shard / f"{safe_key}.wav"
            parameters = parameters_for_profile(profile, semantic_key=safe_key)
            parameter_seed = f"{safe_key}:{profile_id}"
            if output.exists():
                info = validate_wav(output, sample_rate=16000)
                details = {"reused_existing_audio": True}
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                details = render_augmented_file(
                    source_audio_path=Path(clean_row["audio_filepath"]),
                    output_audio_path=output,
                    profile_id=profile_id,
                    parameters=parameters,
                    seed_text=parameter_seed,
                )
                info = validate_wav(output, sample_rate=16000)
            if info.peak_ratio <= 0.001:
                raise RuntimeError(f"near-empty S6 hardvoice augmented holdout output: {safe_key}")
            row = {
                "schema_version": "1.0",
                "view_id": "sl-corpus-v2-s6tts-hardvoice-augmented-holdout-v1",
                "safe_key": safe_key,
                "row_index": clean_index,
                "source_clean_view_id": "sl-corpus-v2-s6tts-hardvoice-clean-holdout-v1",
                "source_clean_audio_sha256": str(clean_row["audio_sha256"]),
                "profile_id": profile_id,
                "profile_index": profile_index,
                "parameter_seed": parameter_seed,
                "parameters": parameters,
                "transform_details": details,
                "text_hash": str(clean_row["text_hash"]),
                "audio_filepath": str(info.path.resolve()),
                "audio_relative_path": str(info.path.relative_to(aug_root)),
                "audio_sha256": info.sha256,
                "duration_seconds": round(info.duration_seconds, 6),
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "sample_width": info.sample_width,
                "frames": info.frames,
                "peak_ratio": round(info.peak_ratio, 6),
                "target_lang": "sl-SI",
                "source_type": "synthetic_tts_augmented",
                "partition_role": "synthetic_diagnostic_holdout",
            }
            augmented_rows.append(row)
            provenance_rows.append({key: value for key, value in row.items() if key != "audio_filepath"})
    write_jsonl(holdout_augmented_manifest(), augmented_rows)
    write_jsonl(holdout_augmented_provenance(), provenance_rows)
    summary = {
        "status": "S6TTS_HARDVOICE_HOLDOUT_AUDIO_ACCEPTED",
        "source_holdout": config["hardvoice_holdout"]["source_holdout"],
        "source_text_sha256": HOLDOUT_TEXT_SHA,
        "rows": len(rows),
        "s6tts_clean_files": len(clean_rows),
        "s6tts_augmented_files": len(augmented_rows),
        "clean_manifest_sha256": sha256_file(paths.audio_manifest),
        "clean_provenance_manifest_sha256": sha256_file(paths.provenance_manifest),
        "augmented_manifest_sha256": sha256_file(holdout_augmented_manifest()),
        "augmented_provenance_manifest_sha256": sha256_file(holdout_augmented_provenance()),
        "duration_distribution": _small_distribution([float(row["duration_seconds"]) for row in [*clean_rows, *augmented_rows]]),
        "peak_distribution": _small_distribution([float(row["peak_ratio"]) for row in [*clean_rows, *augmented_rows]]),
        "duplicate_path_count": 0,
        "synthesis_or_augmentation_failure_count": 0,
        "generated_audio_committed": False,
        "local_manifest_committed": False,
        "raw_text_committed": False,
    }
    atomic_write_json(holdout_summary_path(), summary)
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-s6tts-hardvoice-holdout-v1",
        "status": summary["status"],
        "source_holdout": summary["source_holdout"],
        "source_text_sha256": HOLDOUT_TEXT_SHA,
        "rows": summary["rows"],
        "s6tts_clean_files": summary["s6tts_clean_files"],
        "s6tts_augmented_files": summary["s6tts_augmented_files"],
        "clean_manifest_sha256": summary["clean_manifest_sha256"],
        "augmented_manifest_sha256": summary["augmented_manifest_sha256"],
        "s6tts_revision": config["data"]["s6tts_revision"],
        "s6tts_runtime_data_hash": config["data"]["s6tts_runtime_data_hash"],
        "duration_distribution": summary["duration_distribution"],
        "peak_distribution": summary["peak_distribution"],
        "generated_audio_committed": False,
        "local_manifest_committed": False,
        "raw_text_committed": False,
        "allowed_uses": ["synthetic-only hard-voice diagnostic evaluation for Work Order 0036"],
        "forbidden_uses": ["TRAINING_ELIGIBLE", "checkpoint acceptance", "model release claim", "public audio release"],
    }
    assert_public_report_safe(certificate)
    atomic_write_json(HOLDOUT_CERTIFICATE, certificate)
    report = {
        "schema_version": "1.0",
        "report_id": "0022-s6tts-hardvoice-holdout-admission",
        **certificate,
    }
    assert_public_report_safe(report)
    atomic_write_json(HOLDOUT_REPORT_JSON, report)
    lines = [
        "# S6TTS Hard-Voice Holdout Admission",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Source holdout: `{summary['source_holdout']}`",
        f"- Rows: {summary['rows']}",
        f"- S6 clean files: {summary['s6tts_clean_files']}",
        f"- S6 augmented files: {summary['s6tts_augmented_files']}",
        f"- Clean manifest SHA256: `{summary['clean_manifest_sha256']}`",
        f"- Augmented manifest SHA256: `{summary['augmented_manifest_sha256']}`",
        "",
        "No generated audio, local manifest, raw text, prediction, checkpoint, or model artifact is committed.",
    ]
    HOLDOUT_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    (HOLDOUT_REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def load_s6_hardvoice_holdout_records(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]], dict[str, Any]]:
    if not holdout_summary_path().exists():
        prepare_s6_hardvoice_holdout(config)
    summary = read_json(holdout_summary_path())
    clean_rows = read_jsonl(holdout_clean_paths().audio_manifest)
    augmented_rows = read_jsonl(holdout_augmented_manifest())
    text_rows = _load_holdout_text_rows()
    if len(clean_rows) != 96 or len(augmented_rows) != 1056:
        raise RuntimeError("S6 hardvoice holdout local row counts mismatch")
    if sha256_file(holdout_clean_paths().audio_manifest) != summary["clean_manifest_sha256"]:
        raise RuntimeError("S6 hardvoice clean holdout manifest SHA mismatch")
    if sha256_file(holdout_augmented_manifest()) != summary["augmented_manifest_sha256"]:
        raise RuntimeError("S6 hardvoice augmented holdout manifest SHA mismatch")
    splits: dict[str, list[StreamingRecord]] = {"s6tts_clean_holdout": [], "s6tts_augmented_holdout": []}
    for index, row in enumerate(clean_rows):
        text = str(text_rows[int(row["row_index"])]["target_text"])
        splits["s6tts_clean_holdout"].append(
            StreamingRecord(
                sample_id=f"s6-hardvoice-clean-{index:05d}",
                audio_filepath=str(resolve_local_artifact_path(str(row["audio_filepath"]))),
                duration=float(row["duration_seconds"]),
                reference=text,
                original_index=index,
                row={"split": "s6tts_clean_holdout", "source_order": index},
            )
        )
    for index, row in enumerate(augmented_rows):
        text = str(text_rows[int(row["row_index"])]["target_text"])
        splits["s6tts_augmented_holdout"].append(
            StreamingRecord(
                sample_id=f"s6-hardvoice-augmented-{index:05d}",
                audio_filepath=str(resolve_local_artifact_path(str(row["audio_filepath"]))),
                duration=float(row["duration_seconds"]),
                reference=text,
                original_index=96 + index,
                row={"split": "s6tts_augmented_holdout", "source_order": index},
            )
        )
    suite = [*splits["s6tts_clean_holdout"], *splits["s6tts_augmented_holdout"]]
    return suite, splits, summary


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


def verify_s6_public_evidence(config: dict[str, Any]) -> dict[str, Any]:
    clean = read_json(repo_path(config["data"]["s6_clean_certificate"]))
    augmented = read_json(repo_path(config["data"]["s6_augmented_certificate"]))
    if clean.get("status") != "S6TTS_SCALE2000_CLEAN_VIEW_AUDIO_ACCEPTED":
        raise RuntimeError("S6 clean view is not accepted")
    if augmented.get("status") != "S6TTS_AUGMENTED_VIEW_AUDIO_ACCEPTED":
        raise RuntimeError("S6 augmented view is not accepted")
    if clean.get("audio_manifest_sha256") != config["data"]["s6_clean_manifest_sha256"]:
        raise RuntimeError("S6 clean manifest SHA mismatch in certificate")
    if augmented.get("augmented_audio_manifest_sha256") != config["data"]["s6_augmented_manifest_sha256"]:
        raise RuntimeError("S6 augmented manifest SHA mismatch in certificate")
    if augmented.get("augmented_provenance_manifest_sha256") != config["data"]["s6_augmented_provenance_manifest_sha256"]:
        raise RuntimeError("S6 augmented provenance SHA mismatch in certificate")
    if clean.get("unexplained_duplicate_audio_hash_count") != 0 or augmented.get("unexplained_duplicate_audio_hash_count") != 0:
        raise RuntimeError("S6 evidence contains unexplained duplicate audio hashes")
    return {
        "s6_clean_certificate_sha256": sha256_file(repo_path(config["data"]["s6_clean_certificate"])),
        "s6_augmented_certificate_sha256": sha256_file(repo_path(config["data"]["s6_augmented_certificate"])),
        "s6_clean_status": clean["status"],
        "s6_augmented_status": augmented["status"],
    }


def verify_local_s6_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    clean_manifest = s6_local_path(config["data"]["s6_clean_manifest"])
    aug_manifest = s6_local_path(config["data"]["s6_augmented_manifest"])
    aug_prov = s6_local_path(config["data"]["s6_augmented_provenance_manifest"])
    for path in (clean_manifest, aug_manifest, aug_prov):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(clean_manifest) != config["data"]["s6_clean_manifest_sha256"]:
        raise RuntimeError("local S6 clean manifest SHA mismatch")
    if sha256_file(aug_manifest) != config["data"]["s6_augmented_manifest_sha256"]:
        raise RuntimeError("local S6 augmented manifest SHA mismatch")
    if sha256_file(aug_prov) != config["data"]["s6_augmented_provenance_manifest_sha256"]:
        raise RuntimeError("local S6 augmented provenance SHA mismatch")
    clean_rows = read_jsonl(clean_manifest)
    aug_rows = read_jsonl(aug_manifest)
    if len(clean_rows) != 16000 or len(aug_rows) != 176000:
        raise RuntimeError("local S6 manifest row counts mismatch")
    return {
        "s6_clean_manifest_sha256": sha256_file(clean_manifest),
        "s6_augmented_manifest_sha256": sha256_file(aug_manifest),
        "s6_augmented_provenance_manifest_sha256": sha256_file(aug_prov),
        "s6_clean_rows": len(clean_rows),
        "s6_augmented_rows": len(aug_rows),
    }


def verify_all_inputs(config: dict[str, Any]) -> dict[str, Any]:
    scale_config = {
        "data": {
            "audio_certificate": config["data"]["base_scale2000_audio_certificate"],
            "experiment_certificate": "docs/data-certificates/sl-corpus-v4-scale2000-joint-adapter-diagnostic-v1.json",
            "experiment_report": "docs/experiments/0014-gams16000-scale2000-text-only-directional.json",
            "fixed_text": config["data"]["fixed_text"],
            "all_views": config["data"]["base_all_views"],
            "exposure_schedule": config["data"]["base_exposure_schedule"],
            "audio_validation": config["data"]["base_audio_validation"],
        }
    }
    return {
        "committed_scale2000_evidence": verify_committed_scale2000_evidence(scale_config),
        "local_scale2000_artifacts": verify_local_scale2000_artifacts(scale_config),
        "committed_s6_evidence": verify_s6_public_evidence(config),
        "local_s6_artifacts": verify_local_s6_artifacts(config),
        "protected_file_fingerprints": protected_file_fingerprints(config),
    }


def _text_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(s6_local_path(config["data"]["fixed_text"]))
    if len(rows) != 16000:
        raise RuntimeError("expected 16000 fixed text rows")
    return rows


def _text_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in _text_rows(config)}


def _view_lookup(config: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows = read_jsonl(s6_local_path(config["data"]["base_all_views"]))
    if len(rows) != 320000:
        raise RuntimeError("expected 320000 base scale-2000 view rows")
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"]))
        lookup[key] = row
    return lookup


def _s6_clean_by_semantic(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    texts = _text_rows(config)
    rows = read_jsonl(s6_local_path(config["data"]["s6_clean_manifest"]))
    output = {}
    for row in rows:
        semantic_key = str(texts[int(row["row_index"])]["candidate_id"])
        output[semantic_key] = row
    if len(output) != 16000:
        raise RuntimeError("S6 clean semantic mapping is incomplete")
    return output


def _s6_aug_by_semantic_profile(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    texts = _text_rows(config)
    rows = read_jsonl(s6_local_path(config["data"]["s6_augmented_manifest"]))
    output = {}
    for row in rows:
        semantic_key = str(texts[int(row["row_index"])]["candidate_id"])
        output[(semantic_key, str(row["profile_id"]))] = row
    if len(output) != 176000:
        raise RuntimeError("S6 augmented semantic/profile mapping is incomplete")
    return output


def _s6_aug_profile_for_position(position: int, slot: int) -> str:
    return EXPECTED_PROFILE_IDS[(position * 3 + slot) % len(EXPECTED_PROFILE_IDS)]


def build_hardvoice_schedule(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_schedule = read_jsonl(s6_local_path(config["data"]["base_exposure_schedule"]))
    if len(base_schedule) != 320000:
        raise RuntimeError("base scale-2000 schedule row count mismatch")
    output = []
    for row in base_schedule:
        round_index = int(row["round"])
        position = int(row["semantic_position"])
        semantic_key = str(row["semantic_key"])
        if round_index == 5:
            replacement = {
                "round": round_index,
                "semantic_position": position,
                "semantic_key": semantic_key,
                "voice": S6_VOICE,
                "profile_id": "clean",
                "view_type": "clean",
                "source_schedule": "s6tts",
                "s6_slot": "clean",
                "batch_order_seed": stable_sha256(f"s6-hard20:{round_index}:{semantic_key}:clean"),
                "spec_augment": False,
            }
        elif round_index in (10, 15, 20):
            slot = {10: 0, 15: 1, 20: 2}[round_index]
            profile = _s6_aug_profile_for_position(position, slot)
            replacement = {
                "round": round_index,
                "semantic_position": position,
                "semantic_key": semantic_key,
                "voice": S6_VOICE,
                "profile_id": profile,
                "view_type": "augmented",
                "source_schedule": "s6tts",
                "s6_slot": f"augmented_{slot + 1}",
                "batch_order_seed": stable_sha256(f"s6-hard20:{round_index}:{semantic_key}:{profile}"),
                "spec_augment": False,
            }
        else:
            replacement = {**row, "source_schedule": "scale2000"}
        output.append(replacement)
    summary = validate_hardvoice_schedule(output)
    return output, summary


def validate_hardvoice_schedule(schedule: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(schedule) != 320000:
        raise ValueError("hardvoice schedule must contain exactly 320000 exposures")
    by_round: dict[int, set[str]] = defaultdict(set)
    source_counts = Counter(str(row["source_schedule"]) for row in schedule)
    voice_counts = Counter(str(row["voice"]) for row in schedule)
    profile_counts = Counter(
        str(row["profile_id"])
        for row in schedule
        if str(row.get("source_schedule")) == "s6tts" and str(row.get("view_type")) == "augmented"
    )
    s6_view_counts = Counter(str(row["view_type"]) for row in schedule if str(row.get("source_schedule")) == "s6tts")
    heldout = {"supertonic-M5", "supertonic-F5", "M5", "F5"}
    for row in schedule:
        round_index = int(row["round"])
        semantic_key = str(row["semantic_key"])
        if semantic_key in by_round[round_index]:
            raise ValueError(f"duplicate semantic key in round {round_index}: {semantic_key}")
        by_round[round_index].add(semantic_key)
        if str(row["voice"]) in heldout:
            raise ValueError("held-out Supertonic voice leaked into hardvoice schedule")
    if sorted(by_round) != list(range(1, 21)):
        raise ValueError("schedule must contain rounds 1..20")
    for round_index, keys in by_round.items():
        if len(keys) != 16000:
            raise ValueError(f"round {round_index} contains {len(keys)} semantic rows")
    if source_counts["s6tts"] != 64000 or source_counts["scale2000"] != 256000:
        raise ValueError("hardvoice schedule source counts mismatch")
    if s6_view_counts["clean"] != 16000 or s6_view_counts["augmented"] != 48000:
        raise ValueError("S6 clean/augmented exposure counts mismatch")
    if max(profile_counts.values()) - min(profile_counts.values()) > 1:
        raise ValueError("S6 augmented profile distribution is not balanced")
    return {
        "status": "PASSED",
        "schedule_id": "scale2000_plus_s6tts_hardvoice_20pct_v1",
        "semantic_rows": 16000,
        "total_exposures": 320000,
        "original_scale2000_exposures": source_counts["scale2000"],
        "s6tts_exposures": source_counts["s6tts"],
        "s6tts_share": round(source_counts["s6tts"] / len(schedule), 6),
        "s6_clean_exposures": s6_view_counts["clean"],
        "s6_augmented_exposures": s6_view_counts["augmented"],
        "profile_distribution": dict(sorted(profile_counts.items())),
        "voice_counts": dict(sorted(voice_counts.items())),
        "heldout_voice_exposures": {
            "supertonic-M5": voice_counts.get("supertonic-M5", 0),
            "supertonic-F5": voice_counts.get("supertonic-F5", 0),
        },
        "optimizer_steps": 40000,
    }


def prepare_schedule(config: dict[str, Any]) -> dict[str, Any]:
    verify_all_inputs(config)
    rows, summary = build_hardvoice_schedule(config)
    path = schedule_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, rows)
    schedule_sha = sha256_file(path)
    summary = {**summary, "schedule_sha256": schedule_sha}
    atomic_write_json(schedule_summary_path(config), summary)
    return summary


def _s6_record_from_row(text_row: dict[str, Any], row: dict[str, Any], *, reason: str):
    from slaif_asr.corpus_v2_training import TrainingRecord

    text = str(text_row["target_text"])
    expected_text_hash = stable_sha256(text)
    if str(row.get("text_hash")) != expected_text_hash:
        raise RuntimeError("S6 text/audio text-hash mismatch")
    path = resolve_local_artifact_path(str(row["audio_filepath"]))
    return TrainingRecord(
        selected_training_id=str(text_row["candidate_id"]),
        audio_filepath=str(path),
        duration=float(row["duration_seconds"]),
        text=text,
        text_sha256=expected_text_hash,
        audio_sha256=str(row["audio_sha256"]),
        selection_reason=reason,
        selection_rank=int(str(text_row["generation"]["prompt_cell"]).removeprefix("cell")) if str(text_row["generation"]["prompt_cell"]).startswith("cell") else 0,
    )


def load_hardvoice_round_records(config: dict[str, Any]):
    if not schedule_path(config).exists():
        prepare_schedule(config)
    text_by_id = _text_by_id(config)
    base_views = _view_lookup(config)
    s6_clean = _s6_clean_by_semantic(config)
    s6_aug = _s6_aug_by_semantic_profile(config)
    schedule = read_jsonl(schedule_path(config))
    rounds: dict[int, list[Any]] = defaultdict(list)
    meta_by_audio: dict[str, dict[str, Any]] = {}
    seen = set()
    for item in schedule:
        semantic_key = str(item["semantic_key"])
        round_key = (int(item["round"]), semantic_key)
        if round_key in seen:
            raise RuntimeError(f"duplicate semantic item in round {round_key[0]}")
        seen.add(round_key)
        text_row = text_by_id[semantic_key]
        if item["source_schedule"] == "s6tts":
            if item["view_type"] == "clean":
                source = s6_clean[semantic_key]
            else:
                source = s6_aug[(semantic_key, str(item["profile_id"]))]
            record = _s6_record_from_row(text_row, source, reason="s6tts_hardvoice20_decoder_joint_rnnt")
        else:
            key = (semantic_key, str(item["view_type"]), str(item["voice"]), str(item["profile_id"]))
            record = training_record_from_view(text_row, base_views[key], reason="s6tts_hardvoice20_retained_scale2000")
        rounds[int(item["round"])].append(record)
        meta_by_audio[record.audio_filepath] = {
            "voice": item["voice"],
            "profile_id": item["profile_id"],
            "view_type": item["view_type"],
            "source_schedule": item["source_schedule"],
            "spec_augment": bool(item.get("spec_augment", False)),
        }
    for round_index in range(1, 21):
        if len(rounds[round_index]) != 16000:
            raise RuntimeError(f"round {round_index} has {len(rounds[round_index])} records")
    return rounds, meta_by_audio, {"schedule_sha256": sha256_file(schedule_path(config))}


def probe_records(config: dict[str, Any]):
    from slaif_asr.corpus_v2_training import select_probe_records
    from slaif_asr.emission_rnnt_finetune import probe_records as original_probe_records

    anchor, scale = original_probe_records(
        {
            "data": {
                "fixed_text": config["data"]["fixed_text"],
                "all_views": config["data"]["base_all_views"],
            }
        }
    )
    text_rows = _text_rows(config)
    text_by_id = {str(row["candidate_id"]): row for row in text_rows}
    s6_clean = _s6_clean_by_semantic(config)
    s6_aug = _s6_aug_by_semantic_profile(config)
    ordered_keys = sorted(text_by_id, key=lambda key: stable_sha256(key))
    clean_records = [_s6_record_from_row(text_by_id[key], s6_clean[key], reason="s6_clean_probe") for key in ordered_keys]
    aug_records = []
    for index, key in enumerate(ordered_keys):
        profile = EXPECTED_PROFILE_IDS[index % len(EXPECTED_PROFILE_IDS)]
        aug_records.append(_s6_record_from_row(text_by_id[key], s6_aug[(key, profile)], reason="s6_augmented_probe"))
    return anchor, scale, select_probe_records(clean_records, 32), select_probe_records(aug_records, 320)


def classify_hardvoice(
    metrics: dict[str, dict[str, Any]],
    *,
    base_s6_metrics: dict[str, dict[str, Any]],
    pr36_s6_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    s6_clean_gain = (
        float(metrics["s6tts_clean_holdout"]["wer"]) < float(base_s6_metrics["s6tts_clean_holdout"]["wer"])
        and float(metrics["s6tts_clean_holdout"]["cer"]) < float(base_s6_metrics["s6tts_clean_holdout"]["cer"])
    )
    s6_aug_gain = (
        float(metrics["s6tts_augmented_holdout"]["wer"]) < float(base_s6_metrics["s6tts_augmented_holdout"]["wer"])
        and float(metrics["s6tts_augmented_holdout"]["cer"]) < float(base_s6_metrics["s6tts_augmented_holdout"]["cer"])
    )
    real_vs_base_safe = all(
        float(metrics[split][metric]) <= float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in ("fleurs_v2", "artur_j")
        for metric in ("wer", "cer")
    )
    real_better_than_base = real_vs_base_safe
    real_within_pr36 = all(
        float(metrics[split][metric]) - float(PR36_DECODER_JOINT_METRICS[split][metric]) <= (0.75 if metric == "wer" else 0.35)
        for split in ("fleurs_v2", "artur_j")
        for metric in ("wer", "cer")
    )
    empty_safe = all(int(metrics[split]["empty"]) <= int(PR36_DECODER_JOINT_METRICS[split]["empty"]) for split in ("fleurs_v2", "artur_j"))
    pr36_s6_improved = None
    if pr36_s6_metrics is not None:
        pr36_s6_improved = all(
            float(metrics[split][metric]) < float(pr36_s6_metrics[split][metric])
            for split in ("s6tts_clean_holdout", "s6tts_augmented_holdout")
            for metric in ("wer", "cer")
        )
    if s6_clean_gain and s6_aug_gain and real_better_than_base and real_within_pr36 and empty_safe and (pr36_s6_improved is not False):
        classification = "S6TTS_HARDVOICE_IMPACT_POSITIVE_REAL_SAFE"
    elif s6_clean_gain and s6_aug_gain and real_vs_base_safe:
        classification = "S6TTS_HARDVOICE_IMPROVES_S6_SYNTHETIC_ONLY"
    elif not real_vs_base_safe or any(int(metrics[split]["empty"]) > int(BASE_DIRECTIONAL_METRICS[split]["empty"]) for split in ("fleurs_v2", "artur_j")):
        classification = "S6TTS_HARDVOICE_REAL_REGRESSION"
    elif not (s6_clean_gain and s6_aug_gain):
        classification = "S6TTS_HARDVOICE_NO_S6_GAIN"
    else:
        classification = "S6TTS_HARDVOICE_IMPROVES_S6_SYNTHETIC_ONLY"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "s6tts_clean_holdout_gain": s6_clean_gain,
        "s6tts_augmented_holdout_gain": s6_aug_gain,
        "base_s6_metrics": base_s6_metrics,
        "real_burden": real_regression_burden(metrics, BASE_DIRECTIONAL_METRICS),
        "real_safe_vs_base": real_vs_base_safe,
        "real_within_pr36_thresholds": real_within_pr36,
        "empty_safe_vs_pr36": empty_safe,
        "pr36_s6_holdout_comparison": pr36_s6_improved if pr36_s6_metrics is not None else "PR36_S6_HOLDOUT_COMPARISON_NOT_RUN_CHECKPOINT_UNAVAILABLE",
    }


def write_schedule_certificate(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-s6tts-hardvoice20-schedule-v1",
        "status": "DIAGNOSTIC_ONLY",
        "corpus_id": config["data"]["corpus_id"],
        "fixed_text_sha256": config["data"]["fixed_text_sha256"],
        "base_scale2000_all_views_sha256": config["data"]["base_scale2000_all_views_sha256"],
        "s6_clean_manifest_sha256": config["data"]["s6_clean_manifest_sha256"],
        "s6_augmented_manifest_sha256": config["data"]["s6_augmented_manifest_sha256"],
        "schedule_id": config["schedule"]["schedule_id"],
        "semantic_rows": summary["semantic_rows"],
        "exposures_per_semantic_row": config["schedule"]["exposures_per_semantic_row"],
        "total_exposures": summary["total_exposures"],
        "original_scale2000_exposures": summary["original_scale2000_exposures"],
        "s6tts_exposures": summary["s6tts_exposures"],
        "s6tts_share": summary["s6tts_share"],
        "profile_distribution": summary["profile_distribution"],
        "duplicate_path_count": 0,
        "missing_audio_count": 0,
        "schedule_sha256": summary["schedule_sha256"],
        "local_manifest_committed": False,
        "generated_audio_committed": False,
        "training_eligible_issued": False,
        "allowed_uses": ["internal synthetic diagnostic training by Work Order 0036", "aggregate reporting"],
        "forbidden_uses": ["TRAINING_ELIGIBLE", "checkpoint acceptance", "model release claim", "public audio release"],
        "limitations": [
            "This balanced schedule admits a diagnostic training recipe only.",
            "It does not authorize S6TTS audio or trained models for public release.",
            "Synthetic-only training cannot accept a checkpoint.",
        ],
    }
    assert_public_report_safe(certificate)
    atomic_write_json(SCHEDULE_CERTIFICATE, certificate)
    return certificate
