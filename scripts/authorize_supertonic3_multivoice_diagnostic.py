#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.batched_streaming import file_sha256
from slaif_asr.corpus_v2_training import (
    assert_public_report_safe,
    read_json,
    verify_all_input_identities,
    verify_selected_training_certificate,
)
from slaif_asr.real_eval import atomic_write_json
from slaif_asr.slovenian_joint_adapter import load_adapter_spec, repo_path
from slaif_asr.supertonic3_tts import ALL_STYLES, HELD_OUT_STYLES, TRAINING_STYLES, read_jsonl


CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v2-supertonic3-joint-adapter-diagnostic-v1.json")
DIAGNOSTIC_STATUS = "DIAGNOSTIC_ONLY"


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _verify_audio_manifest(path: Path, *, expected_sha256: str, expected_rows: int, expected_voices: tuple[str, ...], partition_role: str) -> dict[str, Any]:
    actual = file_sha256(path)
    _require_equal(actual, expected_sha256, f"{partition_role} Supertonic audio manifest SHA256")
    rows = read_jsonl(path)
    _require_equal(len(rows), expected_rows, f"{partition_role} Supertonic audio manifest row count")
    voices = sorted({str(row["voice_style_id"]) for row in rows})
    _require_equal(tuple(voices), tuple(sorted(expected_voices)), f"{partition_role} Supertonic voice styles")
    if any(str(row["partition_role"]) != partition_role for row in rows):
        raise RuntimeError(f"{partition_role} Supertonic audio manifest contains another partition role")
    return {"sha256": actual, "rows": len(rows), "voices": voices}


def build_certificate(*, work_order_id: str, experiment_config_path: Path) -> dict[str, Any]:
    if work_order_id != "0023":
        raise ValueError("Supertonic diagnostic certificate is authorized only for work order 0023")
    config = read_json(repo_path(experiment_config_path))
    if config.get("work_order_id") != "0023":
        raise ValueError("experiment config must belong to work order 0023")
    training = config.get("training", {})
    required_training = {
        "arm": "supertonic3_multivoice_joint_adapter_dim32",
        "seed": 1234,
        "epochs": 12,
        "rows_per_epoch": 160,
        "sample_exposures": 1920,
        "batch_size": 8,
        "optimizer_steps": 240,
        "optimizer": "AdamW",
        "scheduler": "none",
        "gradient_accumulation": "none",
        "gradient_clipping": "none",
        "precision": "fp32",
        "tf32": False,
        "spec_augment": False,
        "waveform_augmentation": False,
    }
    for key, expected in required_training.items():
        _require_equal(training.get(key), expected, f"training.{key}")
    _require_equal(float(training.get("learning_rate", -1.0)), 0.001, "training.learning_rate")
    _require_equal(float(training.get("weight_decay", -1.0)), 0.0, "training.weight_decay")

    comparison = config["comparison_report"]
    comparison_path = repo_path(comparison["path"])
    _require_equal(file_sha256(comparison_path), comparison["sha256"], "Experiment 0010 comparison report SHA256")

    adapter_config = repo_path(config["adapter"]["config"])
    adapter_spec = load_adapter_spec(adapter_config)
    _require_equal(adapter_spec.name, "sl-si-joint-adapter-v1", "adapter name")
    _require_equal(adapter_spec.bottleneck_dim, 32, "adapter bottleneck")

    selected = verify_selected_training_certificate(repo_path(config["data"]["selected_training_certificate"]))
    identities = verify_all_input_identities(config, check_gpu=False)

    audio_certificate_path = repo_path(config["supertonic_audio"]["audio_certificate"])
    audio_certificate_sha256 = file_sha256(audio_certificate_path)
    audio_certificate = read_json(audio_certificate_path)
    _require_equal(audio_certificate.get("status"), "AUDIO_ACCEPTED", "Supertonic audio certificate status")
    _require_equal(audio_certificate.get("counts", {}).get("final_training_files"), 1280, "Supertonic final training files")
    _require_equal(audio_certificate.get("counts", {}).get("final_holdout_files"), 192, "Supertonic final holdout files")
    _require_equal(tuple(audio_certificate.get("voice_styles", {}).get("training", [])), TRAINING_STYLES, "Supertonic training voices")
    _require_equal(tuple(audio_certificate.get("voice_styles", {}).get("held_out", [])), HELD_OUT_STYLES, "Supertonic held-out voices")
    _require_equal(tuple(audio_certificate.get("voice_styles", {}).get("available", [])), ALL_STYLES, "Supertonic available voices")

    audio_hashes = audio_certificate["hashes"]
    combined_manifest = repo_path(config["supertonic_audio"]["audio_manifest"])
    training_manifest = repo_path(config["supertonic_audio"]["training_audio_manifest"])
    holdout_manifest = repo_path(config["supertonic_audio"]["holdout_audio_manifest"])
    _require_equal(file_sha256(combined_manifest), audio_hashes["audio_manifest_sha256"], "combined Supertonic audio manifest SHA256")
    training_audio = _verify_audio_manifest(
        training_manifest,
        expected_sha256=audio_hashes["training_audio_manifest_sha256"],
        expected_rows=1280,
        expected_voices=TRAINING_STYLES,
        partition_role="selected_training",
    )
    holdout_audio = _verify_audio_manifest(
        holdout_manifest,
        expected_sha256=audio_hashes["holdout_audio_manifest_sha256"],
        expected_rows=192,
        expected_voices=HELD_OUT_STYLES,
        partition_role="synthetic_holdout",
    )

    voice_counts = audio_certificate["voice_styles"]["counts"]
    if set(voice_counts.get("selected_training", {})) & set(HELD_OUT_STYLES):
        raise RuntimeError("held-out Supertonic voice leaked into training")
    if set(voice_counts.get("synthetic_holdout", {})) & set(TRAINING_STYLES):
        raise RuntimeError("training Supertonic voice leaked into holdout")

    tts_config_path = repo_path(config["tts_config"])
    license_path = repo_path(config["license_assessment"])
    experiment_config_sha256 = file_sha256(repo_path(experiment_config_path))
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-supertonic3-joint-adapter-diagnostic-v1",
        "status": DIAGNOSTIC_STATUS,
        "decision_date": "2026-06-25",
        "work_order_id": work_order_id,
        "named_exception": "corpus-v2 Supertonic 3 multi-voice frozen-base joint-adapter diagnostic",
        "human_approved_exception_statement": (
            "This certificate authorizes only the named Work Order 0023 diagnostic. "
            "It permits internal training of one frozen-base RNNT joint adapter on Supertonic 3 synthetic audio, "
            "does not issue TRAINING_ELIGIBLE, and cannot be reused by another experiment."
        ),
        "selected_training_certificate": {
            "sha256": selected["sha256"],
            "status": selected["certificate"]["status"],
        },
        "source_partitions": {
            "selected_training_manifest_sha256": identities["selected_training"]["selected_manifest_sha256"],
            "selected_training_audio_manifest_sha256": identities["selected_training"]["selected_audio_manifest_sha256"],
            "selected_rows": identities["selected_training"]["rows"],
            "hard_rows": identities["selected_training"]["hard"],
            "control_rows": identities["selected_training"]["control"],
            "holdout_partition_sha256": config["data"]["synthetic_holdout_text_sha256"],
            "holdout_audio_manifest_sha256": identities["synthetic_holdout_audio_manifest_sha256"],
            "holdout_rows": identities["synthetic_holdout_rows"],
        },
        "source_partition_exclusion_evidence": identities["candidate_holdout_overlap_counts"],
        "supertonic": {
            "package": audio_certificate["tts"]["package"],
            "package_version": audio_certificate["tts"]["package_version"],
            "package_license": audio_certificate["tts"]["package_license"],
            "model_repository": audio_certificate["tts"]["model_repository"],
            "model_revision": audio_certificate["tts"]["model_revision"],
            "model_license": audio_certificate["tts"]["model_license"],
            "asset_tree_sha256": audio_certificate["tts"]["asset_tree_sha256"],
            "voice_style_hashes": audio_certificate["voice_styles"]["voice_style_hashes"],
            "available_voice_styles": list(ALL_STYLES),
            "training_voice_styles": list(TRAINING_STYLES),
            "held_out_voice_styles": list(HELD_OUT_STYLES),
            "runtime_packages": audio_certificate.get("synthesis", {}).get("runtime_packages", {}),
            "wheel_record_hashes": audio_certificate.get("synthesis", {}).get("wheel_record_hashes", {}),
        },
        "supertonic_audio": {
            "audio_certificate_sha256": audio_certificate_sha256,
            "status": audio_certificate["status"],
            "combined_audio_manifest_sha256": audio_hashes["audio_manifest_sha256"],
            "training_audio_manifest_sha256": training_audio["sha256"],
            "holdout_audio_manifest_sha256": holdout_audio["sha256"],
            "training_final_files": audio_certificate["counts"]["final_training_files"],
            "holdout_final_files": audio_certificate["counts"]["final_holdout_files"],
            "training_probe_manifest_sha256": audio_hashes["training_probe_manifest_sha256"],
            "exposure_schedule_sha256": audio_hashes["exposure_schedule_sha256"],
        },
        "license_assessment_sha256": file_sha256(license_path),
        "tts_config_sha256": file_sha256(tts_config_path),
        "adapter_config_sha256": file_sha256(adapter_config),
        "experiment_config_sha256": experiment_config_sha256,
        "model": config["model"],
        "nemo_revision": config["model"]["nemo_revision"],
        "adapter": {
            "name": adapter_spec.name,
            "target_module": "model.joint",
            "type": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "strategy": "ResidualAddAdapterStrategy",
            "bottleneck_dimension": adapter_spec.bottleneck_dim,
            "activation": adapter_spec.activation,
            "normalization_position": adapter_spec.norm_position,
            "dropout": adapter_spec.dropout,
            "input_dimension": "derive_at_runtime_from_model.joint.joint_hidden",
            "trainable_parameter_formula": "2 * joint_hidden * 32 + 2 * joint_hidden",
        },
        "fixed_training_protocol": config["training"],
        "permitted_evaluation_sets": [
            "supertonic_training_voice_probe",
            "piper_selected_training",
            "piper_synthetic_holdout",
            "supertonic_heldout_voice_holdout",
            "fleurs_v2",
            "artur_j",
        ],
        "authorized_actions": config["authorization"]["authorized_actions"],
        "prohibited_actions": config["authorization"]["prohibited_actions"],
        "scientific_limitations": [
            "Data status is DIAGNOSTIC_ONLY, not TRAINING_ELIGIBLE.",
            "All training audio is synthetic Supertonic 3 preset voice-style output.",
            "Preset voice styles are not verified real speakers or demographic evidence.",
            "Generated audio and trained artifacts are not authorized for publication or distribution.",
            "Synthetic diagnostics are not real-speech generalization evidence.",
            "No checkpoint or adapter from this experiment may become an accepted parent.",
        ],
    }
    assert_public_report_safe(certificate)
    atomic_write_json(repo_path(CERTIFICATE_PATH), certificate)
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize the Supertonic 3 multi-voice joint-adapter diagnostic.")
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/corpus_v2_supertonic3_multivoice_v1.json"))
    parser.add_argument("--require-status", required=True)
    args = parser.parse_args()
    certificate = build_certificate(work_order_id=args.work_order_id, experiment_config_path=args.experiment_config)
    status = str(certificate.get("status"))
    result = {
        "status": status,
        "certificate": str(CERTIFICATE_PATH),
        "certificate_sha256": file_sha256(repo_path(CERTIFICATE_PATH)),
        "work_order_id": certificate.get("work_order_id"),
        "audio_certificate_sha256": certificate.get("supertonic_audio", {}).get("audio_certificate_sha256"),
        "experiment_config_sha256": certificate.get("experiment_config_sha256"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == args.require_status == DIAGNOSTIC_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
