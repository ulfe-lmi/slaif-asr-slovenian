from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.artur_controller_dev import PARTITION_ID, assert_public_payload_safe, select_earliest_within_tolerance
from slaif_asr.batched_streaming import StreamingRecord, read_jsonl, resolve_manifest_audio_path
from slaif_asr.emission_rnnt_finetune import BASE_DIRECTIONAL_METRICS, SCALE2000_JOINT_ADAPTER_METRICS
from slaif_asr.scale2000_corpus import burden as real_regression_burden
from slaif_asr.tts import validate_wav


EXPERIMENT_ID = "scale2000-decoder-joint-rnnt-artur-earlystop-v1"
WORK_ORDER_ID = "0032"
ARM_NAME = "scale2000_augmented_decoder_joint_rnnt_artur_earlystop"
CERTIFICATE_ID = "sl-corpus-v4-decoder-joint-rnnt-artur-earlystop-diagnostic-v1"
PR36_DECODER_JOINT_METRICS = {
    "piper_synthetic_holdout": {"wer": 34.317, "cer": 13.765, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 14.752, "cer": 4.682, "empty": 0},
    "fleurs_v2": {"wer": 46.195, "cer": 15.604, "empty": 0},
    "artur_j": {"wer": 56.793, "cer": 20.177, "empty": 0},
}


@dataclass(frozen=True)
class RoundCheckpoint:
    round: int
    checkpoint_sha256: str | None
    optimizer_step: int
    exposures_seen: int
    train_loss: float | None
    synthetic_anchor_probe_loss: float | None
    synthetic_scale_probe_loss: float | None
    artur_controller_dev_wer: float | None
    artur_controller_dev_cer: float | None
    empty_count: int | None
    delete: float | None
    insert: float | None
    substitute: float | None
    available: bool
    selected_by_rule: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_agents_controller_dev_exception(text: str) -> bool:
    required = [
        "artur-controller-dev-v1",
        "ADR 0008",
        "aggregate run-control",
        "early stopping",
        "Immutable gates and final blind tests remain unavailable",
    ]
    return all(item in text for item in required)


def validate_earlystop_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if config.get("work_order_id") != WORK_ORDER_ID:
        raise ValueError("work_order_id must be 0032")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    training = config["training"]
    expected = {
        "semantic_rows": 16000,
        "sample_exposures": 320000,
        "effective_batch_size": 8,
        "max_optimizer_steps": 40000,
        "max_rounds": 20,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_clipping": "none",
        "precision": "fp32",
        "tf32": False,
        "objective": "audio_conditioned_rnnt_loss",
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"training.{key} must be {value!r}")
    if training.get("trainable_surface") != ["decoder.", "joint."]:
        raise ValueError("trainable surface must be decoder+joint only")
    if training.get("forbid_text_only_path") is not True:
        raise ValueError("text-only path must be explicitly forbidden")
    controller = config["controller_dev"]
    if controller.get("partition_id") != PARTITION_ID:
        raise ValueError(f"controller_dev.partition_id must be {PARTITION_ID}")
    if controller.get("batch_size") != 1 or controller.get("duration_bucketing") is not False:
        raise ValueError("controller-dev evaluation must use batch size 1 without bucketing")
    if controller.get("allowed_for") != "aggregate_run_control_and_early_stopping_only":
        raise ValueError("controller-dev use must be bounded to aggregate run-control")
    directional = config["post_selection_directional"]
    if directional.get("batch_size") != 32 or directional.get("duration_bucketing") is not True:
        raise ValueError("post-selection directional evaluation must use batch size 32 with bucketing")
    if directional.get("canonical") is not False or directional.get("promotion_eligible") is not False:
        raise ValueError("post-selection directional evaluation must be noncanonical and promotion-ineligible")


def load_controller_dev_records(manifest: Path, *, expected_sha256: str, expected_rows: int) -> list[StreamingRecord]:
    from slaif_asr.batched_streaming import file_sha256

    actual = file_sha256(manifest)
    if actual != expected_sha256:
        raise ValueError(f"{PARTITION_ID}: manifest SHA256 mismatch: {actual} != {expected_sha256}")
    rows = read_jsonl(manifest)
    if len(rows) != expected_rows:
        raise ValueError(f"{PARTITION_ID}: expected {expected_rows} rows, found {len(rows)}")
    records: list[StreamingRecord] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("dataset") != PARTITION_ID:
            raise ValueError(f"{PARTITION_ID}: row {index} dataset mismatch")
        if row.get("partition_role") != "controller_development_real":
            raise ValueError(f"{PARTITION_ID}: row {index} must be controller development")
        if row.get("target_lang") != "sl-SI":
            raise ValueError(f"{PARTITION_ID}: row {index} target_lang must be sl-SI")
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"{PARTITION_ID}: duplicate sample_id")
        seen_ids.add(sample_id)
        audio_path = resolve_manifest_audio_path(manifest, str(row["audio_filepath"]))
        validate_wav(audio_path, sample_rate=16000)
        records.append(
            StreamingRecord(
                sample_id=sample_id,
                audio_filepath=str(audio_path),
                duration=float(row["duration"]),
                reference=str(row["text"]),
                original_index=index,
                row=row,
            )
        )
    return records


def select_round(rounds: Sequence[dict[str, Any]], *, base_empty_count: int) -> dict[str, Any] | None:
    return select_earliest_within_tolerance(rounds, base_empty_count=base_empty_count)


def mark_selected(rounds: Sequence[RoundCheckpoint], selected_round: int | None) -> list[dict[str, Any]]:
    return [
        {**row.public_dict(), "selected_by_rule": selected_round is not None and row.round == selected_round}
        for row in rounds
    ]


def classify_artur_earlystop(
    *,
    selected_round: int | None,
    max_round: int,
    controller_rows: Sequence[dict[str, Any]],
    selected_directional_metrics: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if selected_round is None:
        classification = "ARTUR_EARLYSTOP_NO_REAL_DEV_GAIN"
    elif selected_round == 0:
        classification = "ARTUR_EARLYSTOP_NO_REAL_DEV_GAIN"
    elif selected_round >= max_round:
        classification = "ARTUR_EARLYSTOP_SELECTS_ROUND20_OR_MAXROUND"
    else:
        if selected_directional_metrics is None:
            classification = "ARTUR_EARLYSTOP_FINDS_EFFICIENT_EARLIER_CHECKPOINT"
        else:
            burden = real_regression_burden(selected_directional_metrics, BASE_DIRECTIONAL_METRICS)
            close_or_better = True
            for split in ("fleurs_v2", "artur_j"):
                if float(selected_directional_metrics[split]["wer"]) > PR36_DECODER_JOINT_METRICS[split]["wer"] + 0.75:
                    close_or_better = False
                if float(selected_directional_metrics[split]["cer"]) > PR36_DECODER_JOINT_METRICS[split]["cer"] + 0.35:
                    close_or_better = False
            classification = (
                "ARTUR_EARLYSTOP_FINDS_EFFICIENT_EARLIER_CHECKPOINT"
                if burden == 0.0 and close_or_better
                else "ARTUR_EARLYSTOP_SELECTED_CHECKPOINT_DIRECTIONAL_REGRESSES"
            )
    return {
        "classification": classification,
        "accepted_parent": "none",
        "selected_round": selected_round,
        "max_round": max_round,
        "controller_dev_rows_evaluated": sum(1 for row in controller_rows if row.get("available")),
    }


def assert_post_selection_does_not_change_selection(before: int | None, after: int | None) -> None:
    if before != after:
        raise RuntimeError("post-selection directional metrics must not change selected_round")


def assert_no_raw_report_material(payload: Any) -> None:
    assert_public_payload_safe(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden = ("raw_reference", "raw_hypothesis", "reference_text", "hypothesis_text", "audio_filepath", ".wav")
    for marker in forbidden:
        if marker in serialized:
            raise ValueError(f"public report contains forbidden material: {marker}")


def redacted_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": int(row["round"]),
        "checkpoint_sha256": row.get("checkpoint_sha256") or row.get("sha256"),
        "optimizer_step": int(row.get("optimizer_step", 0)),
        "exposures_seen": int(row.get("exposures_seen", 0)),
        "available": bool(row.get("available", False)),
        "selected_by_rule": bool(row.get("selected_by_rule", False)),
    }


def concurrent_gpu_contract(training_gpu: str, validation_gpu: str, *, sequential: bool) -> dict[str, Any]:
    if sequential:
        return {"mode": "sequential", "status": "PASSED", "reason": "SECOND_GPU_UNAVAILABLE_SEQUENTIAL_VALIDATION_USED"}
    if training_gpu == validation_gpu:
        raise ValueError("training and validation GPU selectors must differ in concurrent mode")
    return {"mode": "concurrent", "status": "PASSED", "training_gpu": training_gpu, "validation_gpu": validation_gpu}
