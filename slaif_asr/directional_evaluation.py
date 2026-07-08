from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.batched_streaming import StreamingRecord, file_sha256, metrics_for, run_batched_arm
from slaif_asr.config import REPO_ROOT
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json, atomic_write_jsonl
from slaif_asr.supertonic3_tts import HELD_OUT_STYLES, load_holdout_items, load_supertonic_config, read_jsonl, supertonic_paths
from slaif_asr.tts import validate_wav


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
    "selected_training_id",
    "source_candidate_id",
    "text",
}
PUBLIC_FORBIDDEN_MARKERS = ("gamsv2-", "gams9holdout-", "/" + "home" + "/", "/" + "mnt" + "/", "/" + "tmp" + "/")


@dataclass(frozen=True)
class DirectionalModel:
    model_id: str
    checkpoint: Path
    expected_sha256: str | None = None
    source: str = "local"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_local_run_artifact(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "runs" not in parts:
        raise FileNotFoundError(path_text)
    index = parts.index("runs")
    suffix = Path(*parts[index:])
    roots = [
        Path(item).expanduser()
        for item in [os.environ.get("SLAIF_ASR_RUNS_ROOT", ""), *os.environ.get("SLAIF_ASR_EXTRA_RUNS_ROOTS", "").split(os.pathsep)]
        if item
    ]
    candidates = [root / Path(*parts[index + 1 :]) for root in roots]
    candidates.append(REPO_ROOT / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path_text)


def git_blob_sha(path: Path) -> str:
    rel = str(path.relative_to(REPO_ROOT))
    completed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def verify_protected_training_files(config: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for entry in config["protected_training_files"]:
        path = repo_path(entry["path"])
        blob = git_blob_sha(path)
        byte_sha = file_sha256(path)
        if blob != entry["git_blob_sha"]:
            raise RuntimeError(f"protected training file Git blob changed: {entry['path']}")
        if byte_sha != entry["byte_sha256"]:
            raise RuntimeError(f"protected training file byte SHA changed: {entry['path']}")
        results[entry["path"]] = {"git_blob_sha": blob, "byte_sha256": byte_sha}
    return results


def verify_historical_reports(config: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for name, report in config["historical_reports"].items():
        path = repo_path(report["path"])
        actual = file_sha256(path)
        if actual != report["sha256"]:
            raise RuntimeError(f"historical report SHA mismatch for {name}: {actual}")
        results[name] = {"path": report["path"], "sha256": actual}
    return results


def verify_model_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for model in directional_models(config, include_replay=False):
        if model.model_id == "base":
            continue
        if not model.checkpoint.exists():
            raise FileNotFoundError(model.checkpoint)
        actual = file_sha256(model.checkpoint)
        if model.expected_sha256 and actual != model.expected_sha256:
            raise RuntimeError(f"model artifact SHA mismatch for {model.model_id}: {actual}")
        models[model.model_id] = {"sha256": actual, "source": model.source}
    return models


def directional_models(config: dict[str, Any], *, include_replay: bool) -> list[DirectionalModel]:
    from slaif_asr.corpus_v2_scoring import checkpoint_path

    artifacts = config["historical_artifacts"]
    models = [
        DirectionalModel("base", checkpoint_path(), None, "untouched_base"),
        DirectionalModel(
            "piper_joint_adapter",
            repo_path(artifacts["piper_joint_adapter"]["checkpoint"]),
            artifacts["piper_joint_adapter"]["sha256"],
            "experiment_0010",
        ),
        DirectionalModel(
            "supertonic3_joint_adapter",
            repo_path(artifacts["supertonic3_joint_adapter"]["checkpoint"]),
            artifacts["supertonic3_joint_adapter"]["sha256"],
            "experiment_0011",
        ),
    ]
    if include_replay:
        replay = config["replay_artifact"]
        models.append(
            DirectionalModel(
                "batched_replay_joint_adapter",
                repo_path(replay["checkpoint"]),
                replay.get("sha256"),
                "work_order_0024",
            )
        )
    return models


def _safe_suite_record(split: str, index: int, row: StreamingRecord) -> StreamingRecord:
    return StreamingRecord(
        sample_id=f"{split}:{index:04d}",
        audio_filepath=row.audio_filepath,
        duration=row.duration,
        reference=row.reference,
        original_index=index,
        row={"split": split, "source_order": row.original_index},
    )


def load_supertonic_heldout_records(config: dict[str, Any]) -> list[StreamingRecord]:
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    holdout_text = {item.source_key: item for item in load_holdout_items(tts_config)}
    rows = read_jsonl(supertonic_paths(tts_config).holdout_audio_manifest)
    output: list[StreamingRecord] = []
    for row in rows:
        voice = str(row["voice_style_id"])
        if voice not in HELD_OUT_STYLES:
            raise RuntimeError("training voice leaked into Supertonic held-out evaluation")
        source_key = str(row["source_key"])
        item = holdout_text[source_key]
        path = resolve_local_run_artifact(str(row["audio_filepath"]))
        validate_wav(path, sample_rate=16000)
        if file_sha256(path) != str(row["audio_sha256"]):
            raise RuntimeError("Supertonic held-out audio hash mismatch")
        output.append(
            StreamingRecord(
                sample_id=f"{source_key}.{voice}",
                audio_filepath=str(path),
                duration=float(row["duration_seconds"]),
                reference=item.text,
                original_index=len(output),
                row={"split": "supertonic_heldout_voice_holdout", "voice_style_id": voice},
            )
        )
    if len(output) != 192:
        raise RuntimeError(f"expected 192 Supertonic held-out records, found {len(output)}")
    return output


def load_directional_suite(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]]]:
    from slaif_asr.corpus_v2_training import load_real_gate_eval_records, load_synthetic_eval_records

    split_records = {
        "piper_synthetic_holdout": load_synthetic_eval_records(config, "synthetic_holdout"),
        "supertonic_heldout_voice_holdout": load_supertonic_heldout_records(config),
        "fleurs_v2": load_real_gate_eval_records(config, "fleurs_v2"),
        "artur_j": load_real_gate_eval_records(config, "artur_j"),
    }
    suite: list[StreamingRecord] = []
    remapped: dict[str, list[StreamingRecord]] = {}
    for split_name, records in split_records.items():
        remapped_rows = [_safe_suite_record(split_name, index, row) for index, row in enumerate(records)]
        for row in remapped_rows:
            suite.append(
                StreamingRecord(
                    sample_id=row.sample_id,
                    audio_filepath=row.audio_filepath,
                    duration=row.duration,
                    reference=row.reference,
                    original_index=len(suite),
                    row=row.row,
                )
            )
        remapped[split_name] = remapped_rows
    if len(suite) != 1378:
        raise RuntimeError(f"expected 1378 directional suite rows, found {len(suite)}")
    return suite, remapped


def split_predictions(
    suite_records: Sequence[StreamingRecord],
    split_records: dict[str, list[StreamingRecord]],
    predictions: dict[str, str],
) -> dict[str, dict[str, str]]:
    expected = {row.sample_id for row in suite_records}
    if set(predictions) != expected:
        missing = sorted(expected - set(predictions))
        unexpected = sorted(set(predictions) - expected)
        raise RuntimeError(f"directional prediction mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {
        split: {row.sample_id: predictions[row.sample_id] for row in records}
        for split, records in split_records.items()
    }


def run_directional_model(
    *,
    config: dict[str, Any],
    model: DirectionalModel,
    suite_records: Sequence[StreamingRecord],
    split_records: dict[str, list[StreamingRecord]],
    run_dir: Path,
    python_executable: Path,
) -> dict[str, Any]:
    from slaif_asr.corpus_v2_scoring import nemo_streaming_script, runtime_environment

    if model.expected_sha256 and file_sha256(model.checkpoint) != model.expected_sha256:
        raise RuntimeError(f"checkpoint SHA mismatch for {model.model_id}")
    arm = run_batched_arm(
        records=suite_records,
        batch_size=int(config["directional_evaluation"]["batch_size"]),
        bucketed=bool(config["directional_evaluation"]["duration_bucketing"]),
        run_dir=run_dir / model.model_id,
        python_executable=python_executable,
        nemo_script=nemo_streaming_script(),
        checkpoint=model.checkpoint,
        context=config["directional_evaluation"]["att_context_size"],
        env=runtime_environment(),
        physical_gpu_index="1",
        monitor_interval_seconds=0.2,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"directional evaluation failed for {model.model_id}: {arm.get('status')}")
    prediction_rows = read_jsonl(run_dir / model.model_id / "predictions.local.jsonl")
    predictions = {str(row["sample_id"]): str(row["hypothesis"]) for row in prediction_rows}
    by_split_predictions = split_predictions(suite_records, split_records, predictions)
    split_summaries = {}
    for split, records in split_records.items():
        metrics = metrics_for(records, by_split_predictions[split])
        split_summaries[split] = {
            "rows": len(records),
            "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            "metrics": metrics,
        }
    summary = {
        "model_id": model.model_id,
        "source": model.source,
        "checkpoint_sha256": file_sha256(model.checkpoint),
        "suite": {
            "rows": int(arm["rows"]),
            "prediction_count": int(arm["prediction_count"]),
            "audio_duration_seconds": arm["audio_duration_seconds"],
            "wall_time_seconds": arm["execution"]["wall_time_seconds"],
            "real_time_factor": arm["end_to_end_real_time_factor"],
            "rows_per_second": arm["utterances_per_second"],
            "audio_seconds_per_wall_second": arm["end_to_end_audio_seconds_per_wall_second"],
            "layout": arm["layout"],
            "gpu_monitor": arm["execution"]["monitor"],
        },
        "splits": split_summaries,
    }
    atomic_write_json(run_dir / model.model_id / "directional-summary.local.json", summary)
    return summary


def normalized_metric_row(split: dict[str, Any]) -> dict[str, Any]:
    normalized = split["metrics"]["normalized"]
    raw = split["metrics"]["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "raw_wer": round(float(raw["corpus_wer"]), 3),
        "raw_cer": round(float(raw["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
    }


def real_regression_burden(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> float:
    burden = 0.0
    for split in ("fleurs_v2", "artur_j"):
        burden += max(0.0, metrics[split]["wer"] - base[split]["wer"])
        burden += max(0.0, metrics[split]["cer"] - base[split]["cer"])
    return round(burden, 6)


def real_non_regression(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> bool:
    for split in ("fleurs_v2", "artur_j"):
        if metrics[split]["wer"] - base[split]["wer"] > 1.0:
            return False
        if metrics[split]["cer"] - base[split]["cer"] > 1.5:
            return False
        if metrics[split]["empty"] > base[split]["empty"]:
            return False
    return True


def real_improvement(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> bool:
    for split in ("fleurs_v2", "artur_j"):
        if metrics[split]["wer"] - base[split]["wer"] <= -1.0:
            return True
        if metrics[split]["cer"] - base[split]["cer"] <= -1.5:
            return True
    return False


def heldout_synthetic_gain(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> bool:
    split = "supertonic_heldout_voice_holdout"
    return metrics[split]["wer"] < base[split]["wer"] or metrics[split]["cer"] < base[split]["cer"]


def classify_directional(models: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    base = models["base"]
    piper = models["piper_joint_adapter"]
    canonical = models["supertonic3_joint_adapter"]
    replay = models["batched_replay_joint_adapter"]
    piper_burden = real_regression_burden(piper, base)
    canonical_burden = real_regression_burden(canonical, base)
    replay_burden = real_regression_burden(replay, base)
    canonical_gain = heldout_synthetic_gain(canonical, base)
    replay_gain = heldout_synthetic_gain(replay, base)
    canonical_reduction = (piper_burden - canonical_burden) / piper_burden * 100.0 if piper_burden else 0.0
    replay_reduction = (piper_burden - replay_burden) / piper_burden * 100.0 if piper_burden else 0.0
    replay_non_regression = real_non_regression(replay, base)
    replay_improvement = real_improvement(replay, base)
    canonical_non_regression = real_non_regression(canonical, base)
    if replay_non_regression and replay_improvement and replay_gain:
        classification = "FAST_DIRECTIONAL_REPLAY_CHANGES_CONCLUSION_POSITIVELY"
    elif (not replay_gain) or replay_reduction < 20.0:
        classification = "FAST_DIRECTIONAL_REPLAY_CHANGES_CONCLUSION_NEGATIVELY"
    elif canonical_gain and replay_gain and canonical_reduction >= 20.0 and replay_reduction >= 20.0 and not canonical_non_regression and not replay_non_regression:
        classification = "FAST_DIRECTIONAL_REPLAY_CONFIRMS_CONCLUSION"
    else:
        classification = "FAST_DIRECTIONAL_REPLAY_INCONCLUSIVE"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_burden": round(piper_burden, 6),
        "canonical_supertonic_burden": round(canonical_burden, 6),
        "replay_supertonic_burden": round(replay_burden, 6),
        "canonical_burden_reduction_percent": round(canonical_reduction, 6),
        "replay_burden_reduction_percent": round(replay_reduction, 6),
        "canonical_supertonic_heldout_gain": canonical_gain,
        "replay_supertonic_heldout_gain": replay_gain,
        "canonical_real_non_regression": canonical_non_regression,
        "replay_real_non_regression": replay_non_regression,
        "replay_real_improvement": replay_improvement,
    }


def suite_plan_hash(records: Sequence[StreamingRecord]) -> str:
    lines = [f"{row.sample_id}\t{row.duration:.6f}\t{row.row.get('split')}" for row in records]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def privacy_safe_public_report(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public directional report contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, child_key)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public directional report contains raw IDs or local paths")


def write_privacy_safe_suite_manifest(path: Path, records: Sequence[StreamingRecord]) -> str:
    rows = [
        {
            "suite_sample_id": row.sample_id,
            "split": row.row.get("split"),
            "source_order": row.row.get("source_order"),
            "duration": row.duration,
        }
        for row in records
    ]
    atomic_write_jsonl(path, rows)
    return file_sha256(path)


def metric_table_from_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        model_id: {
            split: normalized_metric_row(split_summary)
            for split, split_summary in model_summary["splits"].items()
        }
        for model_id, model_summary in summaries.items()
    }
