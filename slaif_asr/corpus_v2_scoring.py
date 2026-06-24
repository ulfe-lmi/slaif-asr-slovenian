from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.acoustic_quality import (
    EXPECTED_CANDIDATE_AUDIO_CERTIFICATE_SHA256,
    EXPECTED_CANDIDATE_AUDIO_MANIFEST_SHA256,
    EXPECTED_HOLDOUT_ACCEPTED_SHA256,
    audio_paths,
    corpus_audio_spec,
    load_audio_generation_config,
    verify_audio_certificate,
)
from slaif_asr.batched_streaming import (
    StreamingRecord,
    ensure_gpu_idle,
    file_sha256,
    load_local_predictions,
    privacy_safe_arm_summary,
    run_batched_arm,
)
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import (
    entity_masked_form,
    fingerprint_hash,
    load_json,
    load_jsonl,
    sha256_text,
)
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.metrics import (
    raw_cer,
    raw_character_edit_counts,
    raw_wer,
    raw_word_edit_counts,
)
from slaif_asr.real_eval import (
    NORMALIZER_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    normalize_sl_asr_text,
    summarize_predictions,
)
from slaif_asr.tts import validate_wav


SCORING_RUN_ID = "sl-corpus-v2-base-scoring-v1"
SCORING_REPORT_SCHEMA_VERSION = "1.0"
SCORING_AUTHORIZATION_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-scoring-authorization-v1.json"
EXPECTED_SCORING_AUTHORIZATION_SHA256 = "42c57975a77594d68cd1b1250a8edc17643bbc254e29642364fc9e4be680664b"
SCORING_AUTHORIZED = "SCORING_AUTHORIZED"
CHECKPOINT_SHA256 = "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"
MODEL_REVISION = "3fc30f3e2ae5d78d462441f3ce89dda694f89bd7"
MODEL_REPOSITORY = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMO_REVISION = "8044a3924bfcfe8ef71d792bb73bf274fe853575"
ATT_CONTEXT_SIZE = [56, 3]
TARGET_LANG = "sl-SI"
EXPECTED_HOLDOUT_AUDIO_MANIFEST_SHA256 = "7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d"
EXPECTED_HOLDOUT_AUDIO_CERTIFICATE_SHA256 = "d5c1660b8b11b8b250d04034dfb2abe14a96dda33d48560875a51d7168865297"

PUBLIC_FORBIDDEN_KEYS = {
    "candidate_id",
    "candidate_ids",
    "selected_training_id",
    "sample_id",
    "sample_ids",
    "text",
    "spoken_text",
    "target_text",
    "reference",
    "hypothesis",
    "audio_filepath",
    "local_path",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "gamsv2-",
    "gams9holdout-",
    "/" + "home" + "/",
    "/" + "mnt" + "/" + "data",
    "/" + "tmp" + "/",
)


@dataclass(frozen=True)
class CorpusScoringSpec:
    corpus_role: str
    public_name: str
    run_subdir: str
    expected_rows: int
    expected_text_sha256: str
    expected_review_sha256: str | None
    expected_audio_manifest_sha256: str
    expected_audio_certificate_sha256: str
    required_text_status: str
    required_audio_status: str


@dataclass(frozen=True)
class CorpusScoringRecord:
    sample_id: str
    audio_filepath: str
    duration: float
    reference: str
    original_index: int
    text_sha256: str
    audio_sha256: str
    source_id: str
    source_family_id: str
    utterance_family_id: str
    discovered_template_family: str
    domain: str
    phenomena: tuple[str, ...]
    prompt_cell: str
    row: dict[str, Any]
    audio_row: dict[str, Any]


def repo_root() -> Path:
    return REPO_ROOT


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_scoring_spec(corpus_role: str) -> CorpusScoringSpec:
    if corpus_role == "synthetic_candidate":
        audio_spec = corpus_audio_spec("synthetic_candidate")
        return CorpusScoringSpec(
            corpus_role=corpus_role,
            public_name="candidate_source",
            run_subdir="candidate-source",
            expected_rows=415,
            expected_text_sha256=audio_spec.expected_accepted_sha256,
            expected_review_sha256=audio_spec.expected_review_sha256,
            expected_audio_manifest_sha256=EXPECTED_CANDIDATE_AUDIO_MANIFEST_SHA256,
            expected_audio_certificate_sha256=EXPECTED_CANDIDATE_AUDIO_CERTIFICATE_SHA256,
            required_text_status="TEXT_ACCEPTED",
            required_audio_status="AUDIO_ACCEPTED",
        )
    if corpus_role == "synthetic_holdout":
        audio_spec = corpus_audio_spec("synthetic_holdout")
        return CorpusScoringSpec(
            corpus_role=corpus_role,
            public_name="synthetic_holdout",
            run_subdir="synthetic-holdout",
            expected_rows=96,
            expected_text_sha256=EXPECTED_HOLDOUT_ACCEPTED_SHA256,
            expected_review_sha256=audio_spec.expected_review_sha256,
            expected_audio_manifest_sha256=EXPECTED_HOLDOUT_AUDIO_MANIFEST_SHA256,
            expected_audio_certificate_sha256=EXPECTED_HOLDOUT_AUDIO_CERTIFICATE_SHA256,
            required_text_status="TEXT_ACCEPTED",
            required_audio_status="AUDIO_ACCEPTED",
        )
    raise ValueError(f"unsupported corpus role: {corpus_role}")


def scoring_root() -> Path:
    return repo_root() / "runs/scoring/sl-corpus-v2-v1"


def role_run_dir(corpus_role: str) -> Path:
    return scoring_root() / corpus_scoring_spec(corpus_role).run_subdir


def scoring_paths(corpus_role: str) -> dict[str, Path]:
    root = role_run_dir(corpus_role)
    return {
        "root": root,
        "manifest": root / "manifest.local.jsonl",
        "predictions": root / "predictions.local.jsonl",
        "per_row": root / "per-row.local.jsonl",
        "summary": root / "scoring.local.json",
        "monitor": root / "gpu-monitor.local.csv",
        "nemo_run": root / "nemo-run",
    }


def runtime_config_path() -> Path:
    return repo_root() / "configs/runtime/nemotron_3_5_asr.json"


def experiment_config_path() -> Path:
    return repo_root() / "configs/experiments/corpus_v2_scoring_selection_v1.json"


def batch_policy_path() -> Path:
    return repo_root() / "configs/evaluation/a100_streaming_batch_policy.json"


def checkpoint_path() -> Path:
    return repo_root() / "models/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo"


def nemo_root() -> Path:
    return repo_root() / ".external/NeMo"


def nemo_streaming_script() -> Path:
    return nemo_root() / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"


def verify_scoring_authorization(*, require_status: str | None = None) -> dict[str, Any]:
    if not SCORING_AUTHORIZATION_PATH.exists():
        raise RuntimeError(f"missing scoring authorization certificate: {SCORING_AUTHORIZATION_PATH}")
    actual_sha = file_sha256(SCORING_AUTHORIZATION_PATH)
    if actual_sha != EXPECTED_SCORING_AUTHORIZATION_SHA256:
        raise RuntimeError(f"scoring authorization SHA256 mismatch: {actual_sha}")
    certificate = read_json(SCORING_AUTHORIZATION_PATH)
    status = str(certificate.get("status", ""))
    if status != SCORING_AUTHORIZED:
        raise RuntimeError(f"scoring authorization status is {status!r}, expected {SCORING_AUTHORIZED}")
    if require_status and status != require_status:
        raise RuntimeError(f"required authorization {require_status}, got {status}")
    return {
        "certificate": certificate,
        "sha256": actual_sha,
        "status": status,
    }


def verify_batch_policy() -> dict[str, Any]:
    policy = read_json(batch_policy_path())
    if int(policy.get("batch_size", -1)) != 1:
        raise RuntimeError("A100 scoring policy must use batch_size=1")
    if bool(policy.get("duration_bucketing")):
        raise RuntimeError("A100 scoring policy must keep duration_bucketing=false")
    reference = policy.get("reference_mode")
    if not isinstance(reference, dict) or int(reference.get("batch_size", -1)) != 1 or bool(reference.get("duration_bucketing")):
        raise RuntimeError("A100 scoring policy must record batch-1 unbucketed reference mode")
    return policy


def verify_runtime_identities(*, check_gpu: bool = True) -> dict[str, Any]:
    checkpoint = checkpoint_path()
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint_sha} != {CHECKPOINT_SHA256}")
    completed = subprocess.run(
        ["git", "-C", str(nemo_root()), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    nemo_revision = completed.stdout.strip()
    if nemo_revision != NEMO_REVISION:
        raise RuntimeError(f"NeMo revision mismatch: {nemo_revision} != {NEMO_REVISION}")
    payload: dict[str, Any] = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha,
        "nemo_revision": nemo_revision,
        "att_context_size": ATT_CONTEXT_SIZE,
        "target_lang": TARGET_LANG,
        "normalizer": NORMALIZER_VERSION,
        "batch_policy": verify_batch_policy(),
    }
    if check_gpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1 for this work order")
        if os.environ.get("NVIDIA_TF32_OVERRIDE") not in {"0", None}:
            raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0 when set")
        gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
        idle = ensure_gpu_idle(physical_gpu_index="1", max_memory_mib=1024, max_utilization_percent=10)
        payload["gpu"] = gpu.to_dict()
        payload["physical_gpu_idle_before_stage"] = idle
    return payload


def load_accepted_text_rows(corpus_role: str) -> list[dict[str, Any]]:
    generation_config = load_audio_generation_config(corpus_role)
    path = audio_paths(generation_config).run_root / corpus_audio_spec(corpus_role).accepted_filename
    return load_jsonl(path)


def load_audio_manifest_rows(corpus_role: str) -> list[dict[str, Any]]:
    generation_config = load_audio_generation_config(corpus_role)
    return load_jsonl(audio_paths(generation_config).audio_manifest)


def prompt_cell(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("prompt_cell", "unknown"))
    return "unknown"


def discovered_template_family(row: dict[str, Any]) -> str:
    text = str(row.get("target_text", ""))
    entities = tuple(item for item in row.get("entities", []) if isinstance(item, dict))
    basis = entity_masked_form(text, entities)
    return f"dtf-{fingerprint_hash(basis)[:16]}"


def verify_audio_manifest_and_text(corpus_role: str) -> list[CorpusScoringRecord]:
    spec = corpus_scoring_spec(corpus_role)
    audio_certificate = verify_audio_certificate(corpus_role)
    if audio_certificate["certificate_sha256"] != spec.expected_audio_certificate_sha256:
        raise RuntimeError(f"{corpus_role}: audio certificate SHA mismatch")
    if audio_certificate["manifest_sha256"] != spec.expected_audio_manifest_sha256:
        raise RuntimeError(f"{corpus_role}: audio manifest SHA mismatch")
    cert = audio_certificate["certificate"]
    if cert.get("status") != spec.required_audio_status:
        raise RuntimeError(f"{corpus_role}: audio status is not {spec.required_audio_status}")
    if cert.get("accepted_text_partition_sha256") != spec.expected_text_sha256:
        raise RuntimeError(f"{corpus_role}: text SHA mismatch in audio certificate")

    text_rows = load_accepted_text_rows(corpus_role)
    audio_rows = load_audio_manifest_rows(corpus_role)
    if len(text_rows) != spec.expected_rows:
        raise RuntimeError(f"{corpus_role}: expected {spec.expected_rows} text rows, saw {len(text_rows)}")
    if len(audio_rows) != spec.expected_rows:
        raise RuntimeError(f"{corpus_role}: expected {spec.expected_rows} audio rows, saw {len(audio_rows)}")
    text_by_id = {str(row["candidate_id"]): row for row in text_rows}
    if len(text_by_id) != len(text_rows):
        raise RuntimeError(f"{corpus_role}: duplicate text candidate IDs")
    records: list[CorpusScoringRecord] = []
    seen_audio_paths: set[str] = set()
    seen_audio_hashes: set[str] = set()
    seen_ids: set[str] = set()
    for index, audio_row in enumerate(audio_rows):
        sample_id = str(audio_row.get("candidate_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise RuntimeError(f"{corpus_role}: duplicate or missing audio candidate ID")
        seen_ids.add(sample_id)
        text_row = text_by_id.get(sample_id)
        if text_row is None:
            raise RuntimeError(f"{corpus_role}: audio row has unexpected candidate ID")
        if text_row.get("partition_role") != corpus_role:
            raise RuntimeError(f"{corpus_role}: text row has wrong partition_role")
        reference = str(text_row["target_text"])
        if reference != str(audio_row.get("text", "")):
            raise RuntimeError(f"{corpus_role}: audio/text reference mismatch")
        text_sha = sha256_text(reference)
        if str(audio_row.get("target_text_sha256", "")) != text_sha:
            raise RuntimeError(f"{corpus_role}: target text hash mismatch for local audio row")
        audio_path = Path(str(audio_row.get("audio_filepath", ""))).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"{corpus_role}: missing audio file for scoring")
        resolved_audio = str(audio_path.resolve())
        if resolved_audio in seen_audio_paths:
            raise RuntimeError(f"{corpus_role}: duplicate audio path")
        seen_audio_paths.add(resolved_audio)
        validate_wav(audio_path, sample_rate=16000)
        audio_sha = file_sha256(audio_path)
        if audio_sha != str(audio_row.get("audio_sha256", "")):
            raise RuntimeError(f"{corpus_role}: audio SHA mismatch")
        if audio_sha in seen_audio_hashes:
            raise RuntimeError(f"{corpus_role}: duplicate audio SHA")
        seen_audio_hashes.add(audio_sha)
        records.append(
            CorpusScoringRecord(
                sample_id=sample_id,
                audio_filepath=resolved_audio,
                duration=float(audio_row["duration_seconds"]),
                reference=reference,
                original_index=index,
                text_sha256=text_sha,
                audio_sha256=audio_sha,
                source_id=str(text_row["source_id"]),
                source_family_id=str(text_row["source_family_id"]),
                utterance_family_id=str(text_row["utterance_family_id"]),
                discovered_template_family=discovered_template_family(text_row),
                domain=str(text_row.get("domain", "unknown")),
                phenomena=tuple(str(item) for item in text_row.get("phenomena", [])),
                prompt_cell=prompt_cell(text_row),
                row=text_row,
                audio_row=audio_row,
            )
        )
    if set(text_by_id) != seen_ids:
        raise RuntimeError(f"{corpus_role}: text/audio ID set mismatch")
    return records


def to_streaming_records(records: Sequence[CorpusScoringRecord]) -> list[StreamingRecord]:
    return [
        StreamingRecord(
            sample_id=record.sample_id,
            audio_filepath=record.audio_filepath,
            duration=record.duration,
            reference=record.reference,
            original_index=record.original_index,
            row={
                "domain": record.domain,
                "phenomena": list(record.phenomena),
                "source_family_id": record.source_family_id,
                "utterance_family_id": record.utterance_family_id,
                "discovered_template_family": record.discovered_template_family,
                "prompt_cell": record.prompt_cell,
                "text_sha256": record.text_sha256,
                "audio_sha256": record.audio_sha256,
            },
        )
        for record in records
    ]


def write_local_scoring_manifest(path: Path, records: Sequence[CorpusScoringRecord], *, corpus_role: str) -> str:
    rows = [
        {
            "sample_id": record.sample_id,
            "audio_filepath": record.audio_filepath,
            "duration": record.duration,
            "text": record.reference,
            "lang": "sl-SI",
            "target_lang": "sl-SI",
            "partition_role": corpus_role,
            "source_type": "synthetic_tts",
            "text_sha256": record.text_sha256,
            "audio_sha256": record.audio_sha256,
        }
        for record in records
    ]
    atomic_write_jsonl(path, rows)
    return file_sha256(path)


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    raw_word_counts = raw_word_edit_counts(reference, hypothesis)
    raw_char_counts = raw_character_edit_counts(reference, hypothesis)
    normalized_reference = normalize_sl_asr_text(reference)
    normalized_hypothesis = normalize_sl_asr_text(hypothesis)
    normalized_word_counts = raw_word_edit_counts(normalized_reference, normalized_hypothesis)
    normalized_char_counts = raw_character_edit_counts(normalized_reference, normalized_hypothesis)
    return {
        "raw_wer": raw_wer(reference, hypothesis).percent,
        "raw_cer": raw_cer(reference, hypothesis).percent,
        "normalized_wer": raw_wer(normalized_reference, normalized_hypothesis).percent,
        "normalized_cer": raw_cer(normalized_reference, normalized_hypothesis).percent,
        "raw_word_edits": raw_word_counts.__dict__,
        "raw_character_edits": raw_char_counts.__dict__,
        "normalized_word_edits": normalized_word_counts.__dict__,
        "normalized_character_edits": normalized_char_counts.__dict__,
    }


def build_per_row_scores(records: Sequence[CorpusScoringRecord], predictions: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = {record.sample_id for record in records}
    actual = set(predictions)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"prediction ID mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for record in sorted(records, key=lambda item: item.original_index):
        hypothesis = predictions[record.sample_id]
        rows.append(
            {
                "sample_id": record.sample_id,
                "reference": record.reference,
                "hypothesis": hypothesis,
                "pipeline_status": "PASSED",
                "empty_hypothesis": not hypothesis.strip(),
                "duration_seconds": record.duration,
                "domain": record.domain,
                "phenomena": list(record.phenomena),
                "prompt_cell": record.prompt_cell,
                "source_id": record.source_id,
                "source_family_id": record.source_family_id,
                "utterance_family_id": record.utterance_family_id,
                "discovered_template_family": record.discovered_template_family,
                "text_sha256": record.text_sha256,
                "audio_sha256": record.audio_sha256,
                **score_pair(record.reference, hypothesis),
            }
        )
    return rows


def metrics_from_per_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return summarize_predictions(
        [
            {
                "reference": str(row["reference"]),
                "hypothesis": str(row["hypothesis"]),
            }
            for row in rows
        ]
    )


def edit_count_totals(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    totals = Counter()
    for row in rows:
        totals.update({name: int(value) for name, value in row[key].items()})
    return {
        "substitutions": totals["substitutions"],
        "deletions": totals["deletions"],
        "insertions": totals["insertions"],
        "distance": totals["substitutions"] + totals["deletions"] + totals["insertions"],
    }


def distribution_bucket(value: float) -> str:
    if value == 0:
        return "0"
    if value <= 25:
        return "0-25"
    if value <= 50:
        return "25-50"
    if value <= 75:
        return "50-75"
    if value <= 100:
        return "75-100"
    return "over-100"


def duration_bucket(value: float) -> str:
    if value < 2.0:
        return "short_lt_2s"
    if value <= 5.0:
        return "medium_2_to_5s"
    return "long_gt_5s"


def bucket_counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(distribution_bucket(float(row[field])) for row in rows).items()))


def duration_bucket_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(duration_bucket(float(row["duration_seconds"])) for row in rows).items()))


def group_metric_summary(rows: Sequence[dict[str, Any]], *, group_field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if group_field == "phenomena":
            values = row.get("phenomena", [])
            for value in values:
                groups[str(value)].append(row)
        else:
            groups[str(row.get(group_field, "unknown"))].append(row)
    output: dict[str, Any] = {}
    for name, group_rows in sorted(groups.items()):
        metrics = metrics_from_per_rows(group_rows)
        output[name] = {
            "count": len(group_rows),
            "duration_seconds": round(sum(float(row["duration_seconds"]) for row in group_rows), 6),
            "raw_corpus_wer": metrics["raw"]["corpus_wer"],
            "raw_corpus_cer": metrics["raw"]["corpus_cer"],
            "normalized_corpus_wer": metrics["normalized"]["corpus_wer"],
            "normalized_corpus_cer": metrics["normalized"]["corpus_cer"],
            "empty_hypotheses": metrics["raw"]["empty_hypothesis_count"],
        }
    return output


def aggregate_scoring_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = metrics_from_per_rows(rows)
    return {
        "rows": len(rows),
        "audio_duration_seconds": round(sum(float(row["duration_seconds"]) for row in rows), 6),
        "metrics": metrics,
        "word_edit_counts": {
            "raw": edit_count_totals(rows, "raw_word_edits"),
            "normalized": edit_count_totals(rows, "normalized_word_edits"),
        },
        "character_edit_counts": {
            "raw": edit_count_totals(rows, "raw_character_edits"),
            "normalized": edit_count_totals(rows, "normalized_character_edits"),
        },
        "wer_distribution_buckets": {
            "raw": bucket_counts(rows, "raw_wer"),
            "normalized": bucket_counts(rows, "normalized_wer"),
        },
        "cer_distribution_buckets": {
            "raw": bucket_counts(rows, "raw_cer"),
            "normalized": bucket_counts(rows, "normalized_cer"),
        },
        "duration_buckets": duration_bucket_counts(rows),
        "domain_aggregates": group_metric_summary(rows, group_field="domain"),
        "phenomenon_aggregates": group_metric_summary(rows, group_field="phenomena"),
        "hard_signal_counts": {
            "empty_hypotheses": sum(1 for row in rows if row["empty_hypothesis"]),
            "normalized_wer_ge_100": sum(1 for row in rows if float(row["normalized_wer"]) >= 100.0),
            "normalized_wer_ge_75": sum(1 for row in rows if float(row["normalized_wer"]) >= 75.0),
            "normalized_cer_ge_50": sum(1 for row in rows if float(row["normalized_cer"]) >= 50.0),
        },
    }


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env.setdefault("NEMO_ROOT", str(nemo_root()))
    return env


def score_corpus_role(corpus_role: str) -> dict[str, Any]:
    verify_scoring_authorization(require_status=SCORING_AUTHORIZED)
    runtime = verify_runtime_identities(check_gpu=True)
    spec = corpus_scoring_spec(corpus_role)
    records = verify_audio_manifest_and_text(corpus_role)
    paths = scoring_paths(corpus_role)
    paths["root"].mkdir(parents=True, exist_ok=True)
    manifest_sha = write_local_scoring_manifest(paths["manifest"], records, corpus_role=corpus_role)
    arm = run_batched_arm(
        records=to_streaming_records(records),
        batch_size=1,
        bucketed=False,
        run_dir=paths["nemo_run"],
        python_executable=Path(sys.executable),
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint_path(),
        context=ATT_CONTEXT_SIZE,
        env=runtime_environment(),
        physical_gpu_index="1",
        monitor_interval_seconds=0.2,
    )
    if arm.get("status") != "PASSED":
        atomic_write_json(paths["summary"], {"status": arm.get("status"), "arm": privacy_safe_arm_summary(arm)})
        raise RuntimeError(f"{corpus_role}: ASR scoring failed with status {arm.get('status')}")
    predictions = load_local_predictions(paths["nemo_run"] / "predictions.local.jsonl")
    local_predictions = [
        {"sample_id": record.sample_id, "hypothesis": predictions[record.sample_id]}
        for record in sorted(records, key=lambda item: item.original_index)
    ]
    atomic_write_jsonl(paths["predictions"], local_predictions)
    per_rows = build_per_row_scores(records, predictions)
    atomic_write_jsonl(paths["per_row"], per_rows)
    monitor_source = paths["nemo_run"] / "gpu-monitor.local.csv"
    if monitor_source.exists():
        shutil.copyfile(monitor_source, paths["monitor"])
    aggregate = aggregate_scoring_summary(per_rows)
    summary = {
        "schema_version": SCORING_REPORT_SCHEMA_VERSION,
        "scoring_run_id": SCORING_RUN_ID,
        "corpus_role": corpus_role,
        "public_name": spec.public_name,
        "corpus_id": corpus_audio_spec(corpus_role).corpus_id,
        "status": "PASSED",
        "repository_commit": git_revision(),
        "runtime": {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
            "att_context_size": ATT_CONTEXT_SIZE,
            "target_lang": TARGET_LANG,
            "precision": "fp32",
            "tf32": False,
            "batch_policy": verify_batch_policy(),
            "gpu": runtime.get("gpu"),
        },
        "input_identity": {
            "row_count": len(records),
            "accepted_text_sha256": spec.expected_text_sha256,
            "accepted_review_sha256": spec.expected_review_sha256,
            "audio_manifest_sha256": spec.expected_audio_manifest_sha256,
            "audio_certificate_sha256": spec.expected_audio_certificate_sha256,
            "local_scoring_manifest_sha256": manifest_sha,
        },
        "execution": privacy_safe_arm_summary(arm).get("execution"),
        "prediction_count": len(predictions),
        "aggregate": aggregate,
        "local_artifacts": {
            "manifest": str(paths["manifest"]),
            "predictions": str(paths["predictions"]),
            "per_row": str(paths["per_row"]),
            "monitor": str(paths["monitor"]),
        },
    }
    atomic_write_json(paths["summary"], summary)
    return summary


def public_partition_summary(local_summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = local_summary["aggregate"]
    execution = local_summary.get("execution") or {}
    return {
        "corpus_id": local_summary["corpus_id"],
        "corpus_role": local_summary["corpus_role"],
        "rows": aggregate["rows"],
        "audio_duration_seconds": aggregate["audio_duration_seconds"],
        "prediction_count": local_summary["prediction_count"],
        "metrics": aggregate["metrics"],
        "word_edit_counts": aggregate["word_edit_counts"],
        "character_edit_counts": aggregate["character_edit_counts"],
        "wer_distribution_buckets": aggregate["wer_distribution_buckets"],
        "cer_distribution_buckets": aggregate["cer_distribution_buckets"],
        "duration_buckets": aggregate["duration_buckets"],
        "domain_aggregates": aggregate["domain_aggregates"],
        "phenomenon_aggregates": aggregate["phenomenon_aggregates"],
        "hard_signal_counts": aggregate["hard_signal_counts"],
        "wall_time_seconds": execution.get("wall_time_seconds"),
        "active_wall_time_seconds": execution.get("active_wall_time_seconds"),
        "real_time_factor": local_summary.get("aggregate", {}).get("audio_duration_seconds")
        and execution.get("wall_time_seconds")
        and round(float(execution["wall_time_seconds"]) / float(aggregate["audio_duration_seconds"]), 6),
        "gpu_monitor": execution.get("monitor", {}),
    }


def assert_public_scoring_payload_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public scoring payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public scoring payload contains row IDs or local paths")


def summarize_scoring() -> dict[str, Any]:
    authorization = verify_scoring_authorization(require_status=SCORING_AUTHORIZED)
    summaries = {
        "candidate_source": read_json(scoring_paths("synthetic_candidate")["summary"]),
        "synthetic_holdout": read_json(scoring_paths("synthetic_holdout")["summary"]),
    }
    public = {
        "schema_version": SCORING_REPORT_SCHEMA_VERSION,
        "report": "corpus-v2-asr-scoring",
        "status": "PASSED",
        "repository_commit": git_revision(),
        "scoring_run_id": SCORING_RUN_ID,
        "authorization": {
            "certificate_id": authorization["certificate"]["certificate_id"],
            "status": authorization["status"],
            "sha256": authorization["sha256"],
        },
        "runtime": {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
            "att_context_size": ATT_CONTEXT_SIZE,
            "target_lang": TARGET_LANG,
            "normalizer": NORMALIZER_VERSION,
            "batch_policy": verify_batch_policy(),
            "precision": "fp32",
            "tf32": False,
        },
        "partitions": {
            name: public_partition_summary(summary)
            for name, summary in summaries.items()
        },
        "limitations": [
            "This is untouched-base ASR scoring of single-voice synthetic audio.",
            "Synthetic holdout metrics are diagnostic only and are not real-speech generalization evidence.",
            "No selected-training data is TRAINING_ELIGIBLE in this report.",
            "Raw generated text, hypotheses, candidate IDs, audio paths, local manifests, and monitor CSV files remain ignored local artifacts.",
        ],
    }
    assert_public_scoring_payload_safe(public)
    json_path = repo_root() / "docs/data-reports/0008-corpus-v2-asr-scoring.json"
    md_path = repo_root() / "docs/data-reports/0008-corpus-v2-asr-scoring.md"
    atomic_write_json(json_path, public)
    write_scoring_markdown(md_path, public)
    return {
        "public_report": public,
        "json_path": str(json_path.relative_to(repo_root())),
        "json_sha256": file_sha256(json_path),
        "markdown_path": str(md_path.relative_to(repo_root())),
        "markdown_sha256": file_sha256(md_path),
    }


def write_scoring_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Corpus-v2 ASR Scoring",
        "",
        "Status: `PASSED`",
        "",
        "This privacy-safe report records untouched-base ASR scoring for the accepted single-voice synthetic candidate source and independent synthetic holdout. It contains aggregate metrics only and does not authorize model training.",
        "",
        "## Runtime",
        "",
        f"- Model: `{MODEL_REPOSITORY}`",
        f"- Model revision: `{MODEL_REVISION}`",
        f"- Checkpoint SHA256: `{CHECKPOINT_SHA256}`",
        f"- NeMo revision: `{NEMO_REVISION}`",
        f"- Context: `{ATT_CONTEXT_SIZE}`",
        "- Batch policy: batch size 1, no duration bucketing, FP32, TF32 disabled",
        "",
        "## Aggregate Metrics",
        "",
        "| Partition | Rows | Normalized WER | Normalized CER | Raw WER | Raw CER | Empty hypotheses | RTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, partition in payload["partitions"].items():
        metrics = partition["metrics"]
        rtf = partition.get("real_time_factor")
        lines.append(
            f"| {name} | {partition['rows']} | {metrics['normalized']['corpus_wer']} | "
            f"{metrics['normalized']['corpus_cer']} | {metrics['raw']['corpus_wer']} | "
            f"{metrics['raw']['corpus_cer']} | {metrics['raw']['empty_hypothesis_count']} | {rtf} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_all_inputs(*, require_authorization: str | None = None, check_gpu: bool = True) -> dict[str, Any]:
    authorization = verify_scoring_authorization(require_status=require_authorization)
    runtime = verify_runtime_identities(check_gpu=check_gpu)
    partition_inputs = {}
    for role in ("synthetic_candidate", "synthetic_holdout"):
        records = verify_audio_manifest_and_text(role)
        spec = corpus_scoring_spec(role)
        partition_inputs[role] = {
            "rows": len(records),
            "accepted_text_sha256": spec.expected_text_sha256,
            "audio_manifest_sha256": spec.expected_audio_manifest_sha256,
            "audio_certificate_sha256": spec.expected_audio_certificate_sha256,
        }
    return {
        "authorization": {
            "status": authorization["status"],
            "sha256": authorization["sha256"],
        },
        "runtime": runtime,
        "partitions": partition_inputs,
    }
