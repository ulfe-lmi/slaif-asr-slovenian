from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import (
    CleanAudioRecord,
    MicrobatchTask,
    load_augmentation_config_from_payload,
    sha256_file,
    validate_public_payload,
)
from slaif_asr.scale200_corpus import load_augmentation_config


EXPERIMENT_ID = "surface08-scale8000-otf-augmented-v1"
WORK_ORDER_ID = "0045"
CORPUS_ID = "sl-corpus-v5-scale8000-training-v1"
SEMANTIC_ROWS = 64_000
CLEAN_FILES = 576_000
FIXED_TEXT_SHA256 = "e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd"
PIPER_RESTORED_MANIFEST_SHA256 = "ef601b45552c6f15fc40568b3ba277908946ff43234bdd84763acdbcdc029dbd"
SUPERTONIC_RESTORED_MANIFEST_SHA256 = "3e1a4bf4cf72ed6f32f64335d3efc146102f5dec86ff71ac3ee006cc1438c4af"
PIPER_COMMITTED_MANIFEST_SHA256 = "290c733207ee08a58d6a707b241b65ed25121d19da11472caed99f562c0fde9f"
SUPERTONIC_COMMITTED_MANIFEST_SHA256 = "c6b3152ee2d8925dfbeb4c7acd2c6f6656f8aa26a32f65570f425a3c5004d5a4"
PROFILE_CONFIG_SHA256 = "ee93dce63a0b5f47b4bc2426d932fe04ccc4a161e13d40349578afabe4fdce40"
AUGMENTATION_KEY = "scale8000-otf-transcript-preserving-v1"
ALGORITHM_VERSION = "otf-transcript-preserving-v1"
SURFACE_ID = "SURFACE_08_FULL_ENCODER"
VOICES = ("piper-artur", "F1", "F2", "F3", "F4", "M1", "M2", "M3", "M4")


@dataclass(frozen=True)
class Scale8000CleanPool:
    records_by_semantic_key: dict[str, tuple[CleanAudioRecord, ...]]
    semantic_order: tuple[str, ...]
    fixed_text_sha256: str
    piper_manifest_sha256: str
    supertonic_manifest_sha256: str

    @property
    def semantic_rows(self) -> int:
        return len(self.semantic_order)

    @property
    def clean_files(self) -> int:
        return sum(len(rows) for rows in self.records_by_semantic_key.values())


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return payload


def _stable_digest(key: str, *parts: str) -> str:
    message = "\x1f".join(parts).encode("utf-8")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def scale8000_runs_root() -> Path:
    value = os.environ.get("SLAIF_ASR_SCALE8000_RUNS_ROOT")
    if not value:
        raise RuntimeError("SLAIF_ASR_SCALE8000_RUNS_ROOT is required")
    return Path(value).expanduser().resolve()


def auxiliary_runs_root() -> Path:
    value = os.environ.get("SLAIF_ASR_AUX_RUNS_ROOT")
    if not value:
        raise RuntimeError("SLAIF_ASR_AUX_RUNS_ROOT is required for probes and evaluation")
    return Path(value).expanduser().resolve()


def resolve_under(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    path.relative_to(root.resolve())
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0" or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected scale-8000 OTF experiment identity")
    if config.get("work_order_id") != WORK_ORDER_ID:
        raise ValueError(f"work_order_id must be {WORK_ORDER_ID}")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("experiment must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if config.get("training_eligible") is not False:
        raise ValueError("TRAINING_ELIGIBLE is forbidden")

    model = config.get("model", {})
    if model.get("checkpoint_sha256") != "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74":
        raise ValueError("training must start from the untouched base")
    if model.get("initialization") != "untouched_base":
        raise ValueError("prior adapted checkpoint initialization is forbidden")
    if model.get("target_lang") != "sl-SI" or model.get("streaming_context") != [56, 3]:
        raise ValueError("target language or streaming context drifted")

    data = config.get("data", {})
    expected_data = {
        "corpus_id": CORPUS_ID,
        "semantic_rows": SEMANTIC_ROWS,
        "clean_files": CLEAN_FILES,
        "combined_text_sha256": FIXED_TEXT_SHA256,
        "piper_restored_manifest_sha256": PIPER_RESTORED_MANIFEST_SHA256,
        "supertonic_restored_manifest_sha256": SUPERTONIC_RESTORED_MANIFEST_SHA256,
        "piper_committed_manifest_sha256": PIPER_COMMITTED_MANIFEST_SHA256,
        "supertonic_committed_manifest_sha256": SUPERTONIC_COMMITTED_MANIFEST_SHA256,
        "training_source": "scale8000_clean_otf_only",
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected!r}")
    serialized_data = json.dumps(data, sort_keys=True).lower()
    if any(marker in serialized_data for marker in ("s6tts", "scale2000_training", "real_speech")):
        raise ValueError("scale-2000, S6TTS, and real speech are forbidden training sources")

    augmentation = config.get("otf_augmentation", {})
    expected_augmentation = {
        "augmentation_key": AUGMENTATION_KEY,
        "algorithm_version": ALGORITHM_VERSION,
        "profile_config_sha256": PROFILE_CONFIG_SHA256,
        "profile_count": 11,
        "workers": 3,
        "prefetch_microbatches": 16,
        "disk_output": False,
        "transcript_preserved": True,
    }
    for key, expected in expected_augmentation.items():
        if augmentation.get(key) != expected:
            raise ValueError(f"otf_augmentation.{key} must be {expected!r}")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE_ID:
        raise ValueError("only SURFACE_08_FULL_ENCODER is authorized")
    if surface.get("encoder_layer_indices") != list(range(24)):
        raise ValueError("Surface08 must train encoder layers 0 through 23")
    if surface.get("fusion_bridge_module") != "prompt_kernel":
        raise ValueError("Surface08 requires the proven prompt_kernel bridge")
    if surface.get("allowed_prefixes") != ["decoder.", "joint.", "encoder.layers.", "prompt_kernel."]:
        raise ValueError("Surface08 trainable prefixes drifted")
    for forbidden_flag in (
        "full_model_allowed",
        "preprocessor_training_allowed",
        "frontend_subsampling_training_allowed",
        "tokenizer_changes_allowed",
        "prompt_identity_changes_allowed",
        "text_only_objective_allowed",
        "temporary_lm_head_allowed",
    ):
        if surface.get(forbidden_flag) is not False:
            raise ValueError(f"trainable_surface.{forbidden_flag} must be false")

    training = config.get("training", {})
    expected_training = {
        "objective": "audio_conditioned_rnnt",
        "effective_batch_size": 8,
        "round_size_exposures": 16_000,
        "steps_per_round": 2_000,
        "primary_exposure_budget": 160_000,
        "max_exposures": 320_000,
        "max_rounds_primary": 10,
        "max_rounds": 20,
        "max_optimizer_steps": 40_000,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "scheduler": "none",
        "precision": "fp32",
        "tf32": False,
        "seed": 1234,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("physical_microbatch_candidates") != [4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [4, 2, 1]")
    if training.get("known_sustained_oom_microbatches") != [4]:
        raise ValueError("the committed Surface08 microbatch-4 sustained OOM must be preserved")
    expected_lrs = {
        "decoder": 0.0005,
        "joint": 0.0005,
        "encoder_all_layers": 0.000005,
        "fusion_bridge": 0.00005,
    }
    if training.get("learning_rates") != expected_lrs:
        raise ValueError("Surface08 learning-rate groups drifted")

    controller = config.get("controller_dev", {})
    if controller.get("partition_id") != "artur-controller-dev-v1":
        raise ValueError("ARTUR controller-dev is required")
    if controller.get("batch_size") != 1 or controller.get("duration_bucketing") is not False:
        raise ValueError("controller-dev must use batch 1 without bucketing")
    if controller.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("immutable gates cannot select the checkpoint")

    evaluation = config.get("evaluation", {})
    if evaluation.get("batch_size") != 32 or evaluation.get("duration_bucketing") is not True:
        raise ValueError("directional evaluation must use batch 32 with bucketing")
    if evaluation.get("canonical") is not False or evaluation.get("promotion_eligible") is not False:
        raise ValueError("evaluation must remain noncanonical and promotion-ineligible")


def load_config(path: Path) -> dict[str, Any]:
    config = _json(path)
    validate_config(config)
    return config


def load_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = REPO_ROOT / str(config["otf_augmentation"]["profile_config"])
    if sha256_file(path) != PROFILE_CONFIG_SHA256:
        raise RuntimeError("augmentation profile config SHA256 mismatch")
    payload = load_augmentation_config(path)
    load_augmentation_config_from_payload(payload)
    return list(payload["augmentation_profiles"])


def verify_committed_scale8000_evidence(config: dict[str, Any]) -> dict[str, Any]:
    certificate_path = REPO_ROOT / str(config["data"]["dataset_certificate"])
    diagnostic_path = REPO_ROOT / str(config["data"]["prior_diagnostic_certificate"])
    certificate = _json(certificate_path)
    diagnostic = _json(diagnostic_path)
    clean = certificate.get("clean_audio", {})
    if certificate.get("corpus_id") != CORPUS_ID or certificate.get("status") != "TEXT_ACCEPTED":
        raise RuntimeError("committed scale-8000 dataset certificate identity mismatch")
    if clean.get("clean_audio_complete") is not True or clean.get("generated_clean_files") != CLEAN_FILES:
        raise RuntimeError("committed scale-8000 clean audio is incomplete")
    if clean["piper"].get("audio_manifest_sha256") != PIPER_COMMITTED_MANIFEST_SHA256:
        raise RuntimeError("committed Piper manifest identity mismatch")
    if clean["supertonic"].get("audio_manifest_sha256") != SUPERTONIC_COMMITTED_MANIFEST_SHA256:
        raise RuntimeError("committed Supertonic manifest identity mismatch")
    if diagnostic.get("classification") != "SCALE8000_CLEAN_BEATS_BASE_BUT_NOT_SCALE2000":
        raise RuntimeError("prior scale-8000 diagnostic evidence mismatch")
    return {
        "dataset_certificate_sha256": sha256_file(certificate_path),
        "prior_diagnostic_certificate_sha256": sha256_file(diagnostic_path),
        "historical_piper_manifest_sha256": PIPER_COMMITTED_MANIFEST_SHA256,
        "historical_supertonic_manifest_sha256": SUPERTONIC_COMMITTED_MANIFEST_SHA256,
    }


def _semantic_order(keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            keys,
            key=lambda value: (
                _stable_digest(AUGMENTATION_KEY, CORPUS_ID, FIXED_TEXT_SHA256, value, "semantic-order"),
                value,
            ),
        )
    )


def load_clean_pool(config: dict[str, Any], *, runs_root: Path) -> Scale8000CleanPool:
    data = config["data"]
    fixed_path = resolve_under(runs_root, str(data["fixed_text_relative_path"]))
    piper_path = resolve_under(runs_root, str(data["piper_manifest_relative_path"]))
    supertonic_path = resolve_under(runs_root, str(data["supertonic_manifest_relative_path"]))
    integrity_path = resolve_under(runs_root, str(data["local_integrity_summary_relative_path"]))
    if sha256_file(fixed_path) != FIXED_TEXT_SHA256:
        raise RuntimeError("scale-8000 fixed text SHA256 mismatch")
    if sha256_file(piper_path) != PIPER_RESTORED_MANIFEST_SHA256:
        raise RuntimeError("restored Piper manifest SHA256 mismatch")
    if sha256_file(supertonic_path) != SUPERTONIC_RESTORED_MANIFEST_SHA256:
        raise RuntimeError("restored Supertonic manifest SHA256 mismatch")
    integrity = _json(integrity_path)
    if (
        integrity.get("status") != "PASSED"
        or integrity.get("semantic_rows") != SEMANTIC_ROWS
        or integrity.get("audio_records") != CLEAN_FILES
        or integrity.get("audio_checks") != {"ok": CLEAN_FILES}
        or integrity.get("target_text_hash_mismatches") != 0
        or integrity.get("duplicate_paths") != 0
    ):
        raise RuntimeError("scale-8000 local full-audio integrity evidence is not passing")

    texts: dict[str, tuple[str, str]] = {}
    with fixed_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["utterance_family_id"])
            transcript = str(row.get("target_text") or row["spoken_text"])
            if key in texts:
                raise RuntimeError("duplicate scale-8000 semantic key")
            texts[key] = (transcript, hashlib.sha256(transcript.encode("utf-8")).hexdigest())
    if len(texts) != SEMANTIC_ROWS:
        raise RuntimeError("scale-8000 fixed text row count mismatch")

    records: dict[str, dict[str, CleanAudioRecord]] = {key: {} for key in texts}

    def consume(path: Path, *, piper: bool) -> int:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row["utterance_family_id"])
                text_row = texts.get(key)
                if text_row is None or row.get("target_text_sha256") != text_row[1]:
                    raise RuntimeError("scale-8000 clean record transcript identity mismatch")
                voice = "piper-artur" if piper else str(row["voice_style_id"])
                if voice not in VOICES or voice in records[key]:
                    raise RuntimeError("scale-8000 clean voice identity mismatch")
                raw_audio_path = Path(str(row["audio_filepath"]))
                audio_path = raw_audio_path if raw_audio_path.is_absolute() else runs_root / raw_audio_path
                audio_path = Path(os.path.normpath(str(audio_path)))
                audio_path.relative_to(runs_root)
                if int(row.get("sample_rate", 0)) != 16_000 or int(row.get("channels", 0)) != 1:
                    raise RuntimeError("scale-8000 clean WAV metadata mismatch")
                source_hash = str(row.get("audio_sha256", ""))
                if len(source_hash) != 64:
                    raise RuntimeError("scale-8000 source audio SHA256 missing")
                records[key][voice] = CleanAudioRecord(
                    semantic_key=key,
                    voice=voice,
                    source_audio_sha256=source_hash,
                    audio_path=audio_path,
                    duration_seconds=float(row["duration_seconds"]),
                    transcript=text_row[0],
                )
                count += 1
        return count

    if consume(piper_path, piper=True) != SEMANTIC_ROWS:
        raise RuntimeError("restored Piper row count mismatch")
    if consume(supertonic_path, piper=False) != SEMANTIC_ROWS * 8:
        raise RuntimeError("restored Supertonic row count mismatch")
    if any(tuple(sorted(rows)) != tuple(sorted(VOICES)) for rows in records.values()):
        raise RuntimeError("not every scale-8000 row has all nine clean voices")

    ordered = {
        key: tuple(records[key][voice] for voice in VOICES)
        for key in records
    }
    return Scale8000CleanPool(
        records_by_semantic_key=ordered,
        semantic_order=_semantic_order(ordered),
        fixed_text_sha256=FIXED_TEXT_SHA256,
        piper_manifest_sha256=PIPER_RESTORED_MANIFEST_SHA256,
        supertonic_manifest_sha256=SUPERTONIC_RESTORED_MANIFEST_SHA256,
    )


def exposure_record(pool: Scale8000CleanPool, virtual_exposure_id: int) -> CleanAudioRecord:
    if virtual_exposure_id < 0:
        raise ValueError("virtual exposure ID must be non-negative")
    semantic_position = virtual_exposure_id % pool.semantic_rows
    coverage_cycle = virtual_exposure_id // pool.semantic_rows
    semantic_key = pool.semantic_order[semantic_position]
    voice_index = (semantic_position + coverage_cycle) % len(VOICES)
    return pool.records_by_semantic_key[semantic_key][voice_index]


def round_exposures(
    pool: Scale8000CleanPool,
    round_index: int,
    *,
    round_size: int = 16_000,
) -> list[tuple[CleanAudioRecord, int]]:
    if round_index < 1:
        raise ValueError("training rounds start at one")
    start = (round_index - 1) * round_size
    return [(exposure_record(pool, exposure_id), exposure_id) for exposure_id in range(start, start + round_size)]


def build_round_tasks(
    pool: Scale8000CleanPool,
    round_index: int,
    *,
    physical_microbatch: int,
    effective_batch: int = 8,
    round_size: int = 16_000,
    duration_bucket_size: int = 256,
) -> list[MicrobatchTask]:
    if effective_batch % physical_microbatch:
        raise ValueError("physical microbatch must divide effective batch")
    if duration_bucket_size % effective_batch:
        raise ValueError("duration bucket size must divide into effective batches")
    exposures = sorted(round_exposures(pool, round_index, round_size=round_size), key=lambda item: item[0].duration_seconds)
    buckets = [exposures[index : index + duration_bucket_size] for index in range(0, len(exposures), duration_bucket_size)]
    buckets.sort(
        key=lambda bucket: _stable_digest(
            AUGMENTATION_KEY,
            CORPUS_ID,
            str(round_index),
            str(bucket[0][1]),
            "duration-bucket-order",
        )
    )
    ordered: list[tuple[CleanAudioRecord, int]] = []
    for bucket_index, bucket in enumerate(buckets):
        bucket.sort(
            key=lambda item: _stable_digest(
                AUGMENTATION_KEY,
                CORPUS_ID,
                str(round_index),
                str(bucket_index),
                str(item[1]),
                "within-bucket-order",
            )
        )
        ordered.extend(bucket)
    if len(ordered) != round_size or len({exposure_id for _record, exposure_id in ordered}) != round_size:
        raise RuntimeError("scale-8000 OTF round schedule lost or duplicated exposures")
    tasks = []
    for index in range(0, len(ordered), physical_microbatch):
        tasks.append(
            MicrobatchTask(
                microbatch_index=index // physical_microbatch,
                exposures=tuple(ordered[index : index + physical_microbatch]),
            )
        )
    return tasks


def diversity_statistics(
    pool: Scale8000CleanPool,
    exposure_count: int,
    *,
    profile_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    semantic_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    for exposure_id in range(exposure_count):
        record = exposure_record(pool, exposure_id)
        semantic_counts[record.semantic_key] += 1
        voice_counts[record.voice] += 1
    values = sorted(semantic_counts.values())
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "total_virtual_exposures_seen": exposure_count,
        "unique_semantic_rows_seen": len(semantic_counts),
        "unique_transcripts_seen": len(semantic_counts),
        "mean_exposures_per_semantic_row": round(exposure_count / len(semantic_counts), 6) if semantic_counts else 0.0,
        "p95_exposures_per_semantic_row": values[p95_index] if values else 0,
        "voice_distribution": dict(sorted(voice_counts.items())),
        "augmentation_profile_distribution": dict(sorted((profile_counts or {}).items())),
    }


def validate_loader_telemetry(payload: dict[str, Any]) -> None:
    required = (
        "round",
        "examples_per_second",
        "otf_fill_rate",
        "consumer_wait_percent",
        "queue_p50",
        "queue_p95",
        "worker_count",
        "microbatch_count",
    )
    if any(key not in payload for key in required):
        raise ValueError("loader telemetry is incomplete")
    if int(payload["worker_count"]) != 3:
        raise ValueError("live OTF loader must use exactly three workers")
    if not 0.0 <= float(payload["otf_fill_rate"]) <= 1.0:
        raise ValueError("OTF fill rate is outside [0, 1]")
    if float(payload["consumer_wait_percent"]) < 0.0:
        raise ValueError("consumer wait percentage cannot be negative")


def validate_public_report(payload: dict[str, Any]) -> None:
    validate_public_payload(payload)
