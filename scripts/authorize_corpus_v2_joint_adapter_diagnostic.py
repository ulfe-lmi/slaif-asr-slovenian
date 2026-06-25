#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.batched_streaming import file_sha256
from slaif_asr.corpus_v2_training import assert_public_report_safe, read_json, verify_all_input_identities, verify_selected_training_certificate
from slaif_asr.real_eval import atomic_write_json
from slaif_asr.slovenian_joint_adapter import load_adapter_spec, repo_path


CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v2-joint-adapter-diagnostic-v1.json")
DIAGNOSTIC_STATUS = "DIAGNOSTIC_ONLY"


def build_certificate(*, work_order_id: str, selected_certificate: Path, adapter_config: Path, experiment_config: Path) -> dict[str, object]:
    if work_order_id != "0022":
        raise ValueError("joint-adapter diagnostic certificate is authorized only for work order 0022")
    config = read_json(repo_path(experiment_config))
    if config.get("work_order_id") != "0022":
        raise ValueError("experiment config must belong to work order 0022")
    if config.get("trainable_surface", {}).get("type") != "sl-si-rnnt-joint-linear-residual-adapter":
        raise ValueError("experiment config has wrong trainable surface")
    training = config.get("training", {})
    required_training = {
        "batch_size": 8,
        "epochs": 12,
        "sample_exposures": 1920,
        "optimizer_steps": 240,
        "seed": 1234,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected}")
    if float(training.get("learning_rate", -1.0)) != 0.001:
        raise ValueError("joint-adapter learning rate must be 0.001")
    if training.get("precision") != "fp32" or training.get("tf32") is not False:
        raise ValueError("training must use FP32 with TF32 disabled")

    spec = load_adapter_spec(adapter_config)
    selected = verify_selected_training_certificate(repo_path(selected_certificate))
    identities = verify_all_input_identities(config, check_gpu=False)
    adapter_sha = file_sha256(repo_path(adapter_config))
    config_sha = file_sha256(repo_path(experiment_config))
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-joint-adapter-diagnostic-v1",
        "status": DIAGNOSTIC_STATUS,
        "decision_date": "2026-06-25",
        "work_order_id": work_order_id,
        "named_exception": "corpus-v2 frozen-base Slovenian joint-adapter diagnostic",
        "human_approved_exception_statement": (
            "This certificate authorizes only the named Work Order 0022 diagnostic. "
            "It permits training one NeMo native RNNT joint adapter on the single-voice synthetic selected-training partition, "
            "does not issue TRAINING_ELIGIBLE, and cannot be reused by another experiment."
        ),
        "selected_training_certificate": {
            "path": str(repo_path(selected_certificate).relative_to(Path.cwd())),
            "sha256": selected["sha256"],
            "status": selected["certificate"]["status"],
        },
        "selected_training": {
            "manifest_sha256": identities["selected_training"]["selected_manifest_sha256"],
            "audio_manifest_sha256": identities["selected_training"]["selected_audio_manifest_sha256"],
            "rows": identities["selected_training"]["rows"],
            "hard_rows": identities["selected_training"]["hard"],
            "control_rows": identities["selected_training"]["control"],
        },
        "holdout": {
            "text_sha256": config["data"]["synthetic_holdout_text_sha256"],
            "audio_manifest_sha256": identities["synthetic_holdout_audio_manifest_sha256"],
            "rows": identities["synthetic_holdout_rows"],
        },
        "candidate_holdout_exclusion_evidence": identities["candidate_holdout_overlap_counts"],
        "candidate_and_holdout_certificates": {
            "candidate_audio_certificate_sha256": selected["certificate"]["candidate_source_hashes"]["audio_certificate_sha256"],
            "holdout_audio_certificate_sha256": selected["certificate"]["holdout_hashes"]["audio_certificate_sha256"],
        },
        "generator_revisions": {
            "candidate_source": "cjvt/GaMS3-12B-Instruct@1d0b27af5748784482600d24779409e7e1dc9adc",
            "synthetic_holdout": "cjvt/GaMS-9B-Instruct@292744023fa0b7ccc7ae2c3c885a67468e49fa03",
        },
        "piper": {
            "engine": "OHF-Voice/piper1-gpl",
            "engine_revision": "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6",
            "voice": "sl_SI-artur-medium",
            "voice_revision": "217ddc79818708b078d0d14a8fae9608b9d77141",
            "voice_concentration": "single voice synthetic audio",
        },
        "model": config["model"],
        "nemo_revision": config["model"]["nemo_revision"],
        "adapter": {
            "name": spec.name,
            "target_module": "model.joint",
            "type": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "strategy": "ResidualAddAdapterStrategy",
            "bottleneck_dimension": spec.bottleneck_dim,
            "activation": spec.activation,
            "normalization_position": spec.norm_position,
            "dropout": spec.dropout,
            "input_dimension": "derive_at_runtime_from_model.joint.joint_hidden",
            "trainable_parameter_formula": "2 * joint_hidden * 32 + 2 * joint_hidden",
        },
        "adapter_config_sha256": adapter_sha,
        "experiment_config_sha256": config_sha,
        "fixed_training_protocol": config["training"],
        "permitted_evaluation_sets": ["selected_training", "synthetic_holdout", "fleurs_v2", "artur_j"],
        "authorized_actions": config["authorization"]["authorized_actions"],
        "prohibited_actions": config["authorization"]["prohibited_actions"],
        "scientific_limitations": [
            "Data status is DIAGNOSTIC_ONLY, not TRAINING_ELIGIBLE.",
            "Selected training and synthetic holdout are single-voice synthetic Piper audio.",
            "Synthetic holdout improvement is diagnostic only and is not real-speech generalization evidence.",
            "No checkpoint or adapter from this experiment may become an accepted parent.",
        ],
    }
    assert_public_report_safe(certificate)
    atomic_write_json(CERTIFICATE_PATH, certificate)
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize the corpus-v2 Slovenian joint-adapter diagnostic.")
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--selected-certificate", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--require-status", required=True)
    args = parser.parse_args()
    certificate = build_certificate(
        work_order_id=args.work_order_id,
        selected_certificate=args.selected_certificate,
        adapter_config=args.adapter_config,
        experiment_config=args.experiment_config,
    )
    status = str(certificate.get("status"))
    result = {
        "status": status,
        "certificate": str(CERTIFICATE_PATH),
        "certificate_sha256": file_sha256(CERTIFICATE_PATH),
        "work_order_id": certificate.get("work_order_id"),
        "adapter_config_sha256": certificate.get("adapter_config_sha256"),
        "experiment_config_sha256": certificate.get("experiment_config_sha256"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == args.require_status == DIAGNOSTIC_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
