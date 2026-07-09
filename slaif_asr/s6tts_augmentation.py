from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.s6tts_tts import (
    ABSOLUTE_PATH_PATTERN,
    duplicate_audio_hash_groups,
    distribution,
    local_path,
    validate_public_payload,
)
from slaif_asr.transcript_preserving_augmentation import parameters_for_profile, render_augmented_file
from slaif_asr.tts import atomic_write_json, atomic_write_text, sha256_file, validate_wav, write_jsonl


CONFIG_PATH = REPO_ROOT / "configs" / "augmentation" / "s6tts_transcript_preserving_11_views_v1.json"
BASE_AUGMENTATION_CONFIG = REPO_ROOT / "configs" / "augmentation" / "scale200_transcript_preserving_v1.json"
CLEAN_CERTIFICATE = REPO_ROOT / "docs" / "data-certificates" / "sl-corpus-v4-s6tts-clean-view-v1.json"
AUGMENTED_CERTIFICATE = REPO_ROOT / "docs" / "data-certificates" / "sl-corpus-v4-s6tts-augmented-view-v1.json"
AUGMENTED_REPORT_JSON = REPO_ROOT / "docs" / "data-reports" / "0021-s6tts-vintage-augmentation-admission.json"
AUGMENTED_REPORT_MD = REPO_ROOT / "docs" / "data-reports" / "0021-s6tts-vintage-augmentation-admission.md"

EXPECTED_PROFILE_IDS = [
    "coupled_speed_pitch_resampling",
    "tempo_preserving_pitch",
    "mild_pitch_formant_vtlp_proxy",
    "procedural_room_impulse_response",
    "environmental_background_noise",
    "coloured_electrical_noise",
    "microphone_channel_filtering",
    "codec_sample_rate_simulation",
    "gain_dynamic_range_variation",
    "timing_silence_variation",
    "compound_realistic_condition",
]


@dataclass(frozen=True)
class S6AugPaths:
    run_root: Path
    audio_manifest: Path
    provenance_manifest: Path
    validation: Path
    summary: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_s6_augmentation_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_json(path)
    clean_certificate = load_json(REPO_ROOT / config["source_clean_certificate"])
    base_config = load_json(REPO_ROOT / config["inherits_profile_config"])
    validate_s6_augmentation_config(config, clean_certificate=clean_certificate, base_config=base_config)
    return config


def validate_s6_augmentation_config(
    config: dict[str, Any],
    *,
    clean_certificate: dict[str, Any],
    base_config: dict[str, Any],
) -> None:
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported S6 augmentation config schema")
    if config.get("policy_id") != "s6tts-transcript-preserving-11-views-v1":
        raise ValueError("unexpected S6 augmentation policy id")
    if clean_certificate.get("status") != "S6TTS_SCALE2000_CLEAN_VIEW_AUDIO_ACCEPTED":
        raise ValueError("source S6 clean-view certificate is not accepted")
    required_pairs = {
        "corpus_id": "corpus_id",
        "source_clean_view_id": "view_id",
        "fixed_text_sha256": "fixed_text_sha256",
        "semantic_rows": "semantic_rows",
        "source_clean_audio_manifest_sha256": "audio_manifest_sha256",
        "source_clean_provenance_manifest_sha256": "provenance_manifest_sha256",
        "s6tts_runtime_data_hash": "s6tts_runtime_data_hash",
        "tts_engine_revision": "tts_engine_revision",
    }
    for config_key, cert_key in required_pairs.items():
        if config.get(config_key) != clean_certificate.get(cert_key):
            raise ValueError(f"S6 augmentation config mismatch for {config_key}")
    if int(config.get("source_clean_files", 0)) != int(clean_certificate.get("actual_clean_files", -1)):
        raise ValueError("unexpected source clean file count")
    if int(config.get("augmentation_profiles_per_row", 0)) != 11:
        raise ValueError("S6 augmentation must use eleven profiles")
    expected = int(config["semantic_rows"]) * int(config["augmentation_profiles_per_row"])
    if int(config.get("expected_augmented_files", 0)) != expected:
        raise ValueError("unexpected S6 augmented file count")
    profile_ids = [profile["profile_id"] for profile in base_config.get("augmentation_profiles", [])]
    if profile_ids != EXPECTED_PROFILE_IDS:
        raise ValueError("unexpected inherited augmentation profile set")
    if int(clean_certificate.get("unexplained_duplicate_audio_hash_count", -1)) != 0:
        raise ValueError("source S6 clean view contains unexplained duplicate hashes")


def s6_aug_paths(config: dict[str, Any]) -> S6AugPaths:
    outputs = config["local_outputs"]
    return S6AugPaths(
        run_root=local_path(outputs["run_root"]),
        audio_manifest=local_path(outputs["audio_manifest"]),
        provenance_manifest=local_path(outputs["provenance_manifest"]),
        validation=local_path(outputs["validation"]),
        summary=local_path(outputs["summary"]),
    )


def load_profiles() -> list[dict[str, Any]]:
    base_config = load_json(BASE_AUGMENTATION_CONFIG)
    return list(base_config["augmentation_profiles"])


def load_clean_manifest(config: dict[str, Any], clean_certificate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    clean_certificate = clean_certificate or load_json(CLEAN_CERTIFICATE)
    clean_root = local_path("runs/data-quality/sl-corpus-v4-s6tts-vintage-clean-view-v1")
    manifest = clean_root / "audio-manifest.local.jsonl"
    provenance = clean_root / "provenance.local.jsonl"
    if not manifest.exists() or not provenance.exists():
        raise FileNotFoundError("S6 clean local manifest is unavailable")
    actual_manifest = sha256_file(manifest)
    if actual_manifest != config["source_clean_audio_manifest_sha256"]:
        raise RuntimeError(f"S6 clean audio manifest SHA256 mismatch: {actual_manifest}")
    actual_provenance = sha256_file(provenance)
    if actual_provenance != config["source_clean_provenance_manifest_sha256"]:
        raise RuntimeError(f"S6 clean provenance manifest SHA256 mismatch: {actual_provenance}")
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != int(config["semantic_rows"]):
        raise RuntimeError(f"expected {config['semantic_rows']} clean rows, found {len(rows)}")
    seen = set()
    for row in rows:
        if row.get("view_id") != config["source_clean_view_id"]:
            raise RuntimeError("unexpected source clean view id")
        if row.get("safe_key") in seen:
            raise RuntimeError("duplicate source clean safe key")
        seen.add(row.get("safe_key"))
        validate_wav(Path(row["audio_filepath"]), sample_rate=16000)
    return rows


def clean_duplicate_sets(clean_certificate: dict[str, Any]) -> list[frozenset[str]]:
    groups = clean_certificate.get("duplicate_audio_hash_groups_redacted", {}).get("groups", [])
    return [frozenset(str(row["safe_key"]) for row in group.get("rows", [])) for group in groups]


def output_path(paths: S6AugPaths, clean_row: dict[str, Any], profile_index: int, profile_id: str) -> Path:
    row_index = int(clean_row["row_index"])
    shard = f"{row_index // 1000:02d}"
    filename = f"{clean_row['safe_key']}__aug{profile_index:02d}-{profile_id}.wav"
    return paths.run_root / "wav" / f"aug{profile_index:02d}-{profile_id}" / shard / filename


def planned_tasks(clean_rows: Sequence[dict[str, Any]], profiles: Sequence[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any], int, dict[str, Any]]]:
    for clean_index, clean_row in enumerate(clean_rows):
        for profile_index, profile in enumerate(profiles, start=1):
            yield clean_index, clean_row, profile_index, profile


def render_one(args: tuple[S6AugPaths, int, dict[str, Any], int, dict[str, Any], bool]) -> dict[str, Any]:
    paths, clean_index, clean_row, profile_index, profile, overwrite = args
    profile_id = str(profile["profile_id"])
    out = output_path(paths, clean_row, profile_index, profile_id)
    rel = out.relative_to(paths.run_root)
    semantic_key = str(clean_row["safe_key"])
    parameters = parameters_for_profile(profile, semantic_key=semantic_key)
    parameter_seed = f"{semantic_key}:{profile_id}"
    started = time.perf_counter()
    reused = False
    if out.exists() and not overwrite:
        info = validate_wav(out, sample_rate=16000)
        details = {"reused_existing_audio": True}
        wall = 0.0
        reused = True
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        details = render_augmented_file(
            source_audio_path=Path(clean_row["audio_filepath"]),
            output_audio_path=out,
            profile_id=profile_id,
            parameters=parameters,
            seed_text=parameter_seed,
        )
        info = validate_wav(out, sample_rate=16000)
        wall = time.perf_counter() - started
    if info.peak_ratio <= 0.001:
        raise RuntimeError(f"{semantic_key}:{profile_id}: near-empty audio")
    return {
        "schema_version": "1.0",
        "view_id": "sl-corpus-v4-s6tts-augmented-view-v1",
        "safe_key": f"{semantic_key}__aug{profile_index:02d}",
        "row_index": int(clean_row["row_index"]),
        "source_clean_view_id": "sl-corpus-v4-s6tts-clean-view-v1",
        "source_clean_safe_key": semantic_key,
        "source_clean_audio_sha256": str(clean_row["audio_sha256"]),
        "source_id": str(clean_row.get("source_id", "")),
        "source_family_id": str(clean_row.get("source_family_id", "")),
        "utterance_family_id": str(clean_row.get("utterance_family_id", "")),
        "partition_role": str(clean_row.get("partition_role", "selected_training")),
        "text_hash": str(clean_row.get("text_hash", "")),
        "profile_id": profile_id,
        "profile_index": profile_index,
        "parameter_seed": parameter_seed,
        "parameters": parameters,
        "transform_details": details,
        "audio_filepath": str(info.path.resolve()),
        "audio_relative_path": str(rel),
        "audio_sha256": info.sha256,
        "duration_seconds": round(info.duration_seconds, 6),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "sample_width": info.sample_width,
        "frames": info.frames,
        "peak_abs": info.peak_abs,
        "peak_ratio": round(info.peak_ratio, 6),
        "target_lang": "sl-SI",
        "source_type": "synthetic_tts_augmented",
        "augmentation_wall_time_seconds": round(wall, 6),
        "reused_existing_audio": reused,
    }


def public_augmented_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "audio_filepath"}


def write_augmented_manifests(paths: S6AugPaths, rows: list[dict[str, Any]]) -> None:
    write_jsonl(paths.audio_manifest, rows)
    write_jsonl(paths.provenance_manifest, [public_augmented_row(row) for row in rows])


def summarize_augmented_view(config: dict[str, Any]) -> dict[str, Any]:
    paths = s6_aug_paths(config)
    clean_certificate = load_json(CLEAN_CERTIFICATE)
    if not paths.audio_manifest.exists() or not paths.provenance_manifest.exists():
        raise FileNotFoundError("S6 augmented local manifests are missing")
    rows: list[dict[str, Any]] = []
    with paths.audio_manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    failures_path = paths.run_root / "failures.local.jsonl"
    failures: list[dict[str, Any]] = []
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as fp:
            failures = [json.loads(line) for line in fp if line.strip()]
    duplicate_paths = len(rows) - len({row["audio_relative_path"] for row in rows})
    duplicate_hashes = len(rows) - len({row["audio_sha256"] for row in rows})
    duplicate_groups = augmented_duplicate_groups(rows, clean_duplicate_sets(clean_certificate))
    unexplained = int(duplicate_groups["unexplained_duplicate_extra_file_count"])
    durations = [float(row["duration_seconds"]) for row in rows]
    peaks = [float(row["peak_ratio"]) for row in rows]
    issues: dict[str, int] = {}
    if len(rows) != int(config["expected_augmented_files"]):
        issues["row_count_mismatch"] = int(config["expected_augmented_files"]) - len(rows)
    if failures:
        by_reason: dict[str, int] = {}
        for failure in failures:
            reason = str(failure.get("reason", "unknown"))
            by_reason[reason] = by_reason.get(reason, 0) + 1
        for reason, count in sorted(by_reason.items()):
            issues[f"augmentation_failure:{reason}"] = count
    if duplicate_paths:
        issues["duplicate_paths"] = duplicate_paths
    if unexplained:
        issues["unexplained_duplicate_audio_hashes"] = unexplained
    if any(float(row["peak_ratio"]) <= 0.001 for row in rows):
        issues["empty_or_near_empty_audio"] = sum(1 for row in rows if float(row["peak_ratio"]) <= 0.001)
    if any(int(row["sample_rate"]) != 16000 or int(row["channels"]) != 1 or int(row["sample_width"]) != 2 for row in rows):
        issues["invalid_wav_format"] = sum(
            1 for row in rows if int(row["sample_rate"]) != 16000 or int(row["channels"]) != 1 or int(row["sample_width"]) != 2
        )
    profiles = load_profiles()
    summary = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-s6tts-augmented-view-v1",
        "status": "S6TTS_AUGMENTED_VIEW_AUDIO_ACCEPTED" if not issues else "S6TTS_AUGMENTATION_REJECTED_QUALITY",
        "decision_date": "2026-07-09",
        "corpus_id": config["corpus_id"],
        "source_clean_view_id": config["source_clean_view_id"],
        "augmented_view_id": config["augmented_view_id"],
        "fixed_text_sha256": config["fixed_text_sha256"],
        "semantic_rows": int(config["semantic_rows"]),
        "source_clean_files": int(config["source_clean_files"]),
        "augmentation_profiles_per_row": int(config["augmentation_profiles_per_row"]),
        "expected_augmented_files": int(config["expected_augmented_files"]),
        "actual_augmented_files": len(rows),
        "source_clean_audio_manifest_sha256": config["source_clean_audio_manifest_sha256"],
        "source_clean_provenance_manifest_sha256": config["source_clean_provenance_manifest_sha256"],
        "augmented_audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "augmented_provenance_manifest_sha256": sha256_file(paths.provenance_manifest),
        "s6tts_runtime_data_hash": config["s6tts_runtime_data_hash"],
        "tts_engine_repository": config["tts_engine_repository"],
        "tts_engine_revision": config["tts_engine_revision"],
        "augmentation_config_sha256": sha256_file(CONFIG_PATH),
        "augmentation_algorithm_version": "s6tts-transcript-preserving-augmentation-v1",
        "profile_ids": [profile["profile_id"] for profile in profiles],
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "duration_distribution": distribution(durations),
        "peak_distribution": distribution(peaks),
        "total_duration_seconds": round(sum(durations), 6),
        "duplicate_path_count": duplicate_paths,
        "duplicate_audio_hash_count": duplicate_hashes,
        "explained_duplicate_audio_hash_count": int(duplicate_groups["explained_duplicate_extra_file_count"]),
        "unexplained_duplicate_audio_hash_count": unexplained,
        "duplicate_audio_hash_groups_redacted": duplicate_groups,
        "issues_by_reason": issues,
        "synthesis_or_augmentation_failure_count": len(failures),
        "generated_audio_committed": False,
        "local_manifest_committed": False,
        "raw_text_committed": False,
        "allowed_uses": config["allowed_uses"],
        "forbidden_uses": config["forbidden_uses"],
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
        "limitations": [
            "S6TTS-generated and S6TTS-augmented audio remain internal diagnostic synthetic material only.",
            "This certificate admits an audio bank but does not define a training sampling schedule.",
            "Public distribution of S6TTS-generated or S6TTS-augmented audio is not authorized.",
        ],
    }
    validate_public_payload(summary)
    atomic_write_json(paths.validation, {"status": summary["status"], "issues_by_reason": issues})
    atomic_write_json(paths.summary, summary)
    return summary


def augmented_duplicate_groups(rows: list[dict[str, Any]], clean_duplicate_sets_: Sequence[frozenset[str]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["audio_sha256"]), []).append(row)
    redacted = []
    explained = 0
    unexplained = 0
    for audio_hash, group in sorted((key, value) for key, value in grouped.items() if len(value) > 1):
        source_keys = frozenset(str(row["source_clean_safe_key"]) for row in group)
        profile_ids = {str(row["profile_id"]) for row in group}
        parameter_blobs = {json.dumps(row.get("parameters", {}), sort_keys=True) for row in group}
        if any(source_keys.issubset(known) for known in clean_duplicate_sets_) and len(profile_ids) == 1:
            explanation = "inherited_numeric_normalization_equivalence"
        elif len(source_keys) == 1 and len(profile_ids) > 1:
            explanation = "deterministic_augmentation_profile_equivalence"
        elif len({str(row["source_clean_audio_sha256"]) for row in group}) == 1 and len(profile_ids) == 1 and len(parameter_blobs) == 1:
            explanation = "deterministic_augmentation_equivalence"
        else:
            explanation = "unexplained"
        extra = len(group) - 1
        if explanation == "unexplained":
            unexplained += extra
        else:
            explained += extra
        redacted.append(
            {
                "audio_sha256": audio_hash,
                "count": len(group),
                "explanation": explanation,
                "profile_ids": sorted(profile_ids),
                "rows": [
                    {
                        "row_index": int(row["row_index"]),
                        "safe_key": str(row["safe_key"]),
                        "source_clean_safe_key": str(row["source_clean_safe_key"]),
                        "text_sha256": str(row.get("text_hash", "")),
                        "duration_seconds": row.get("duration_seconds"),
                        "frames": row.get("frames"),
                        "peak_ratio": row.get("peak_ratio"),
                    }
                    for row in sorted(group, key=lambda item: (int(item["row_index"]), str(item["profile_id"])))
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "duplicate_group_count": len(redacted),
        "duplicate_extra_file_count": sum(len(group) - 1 for group in grouped.values() if len(group) > 1),
        "explained_duplicate_extra_file_count": explained,
        "unexplained_duplicate_extra_file_count": unexplained,
        "groups": redacted,
    }


def write_public_evidence(summary: dict[str, Any]) -> None:
    validate_public_payload(summary)
    atomic_write_json(AUGMENTED_CERTIFICATE, summary)
    atomic_write_json(AUGMENTED_REPORT_JSON, summary)
    status_sentence = (
        "This report admits one internal diagnostic S6TTS transcript-preserving augmented synthetic bank."
        if summary["status"] == "S6TTS_AUGMENTED_VIEW_AUDIO_ACCEPTED"
        else "This report records a failed or blocked S6TTS transcript-preserving augmentation admission attempt."
    )
    duplicate_lines = [
        f"- Duplicate audio hashes: {summary['duplicate_audio_hash_count']}",
        f"- Explained duplicate audio hashes: {summary['explained_duplicate_audio_hash_count']}",
        f"- Unexplained duplicate audio hashes: {summary['unexplained_duplicate_audio_hash_count']}",
    ]
    groups = summary.get("duplicate_audio_hash_groups_redacted", {})
    if groups.get("duplicate_group_count"):
        duplicate_lines.extend(
            [
                "",
                "| Audio SHA256 | Explanation | Profile IDs | Rows | Text SHA256 values |",
                "|---|---|---|---:|---|",
            ]
        )
        for group in groups.get("groups", [])[:20]:
            rows = group["rows"]
            duplicate_lines.append(
                "| `{audio}` | `{explanation}` | `{profiles}` | {row_indexes} | `{text_hashes}` |".format(
                    audio=group["audio_sha256"],
                    explanation=group["explanation"],
                    profiles=", ".join(group["profile_ids"]),
                    row_indexes=", ".join(str(row["row_index"]) for row in rows),
                    text_hashes=", ".join(row["text_sha256"] for row in rows),
                )
            )
        if groups.get("duplicate_group_count", 0) > 20:
            duplicate_lines.append("")
            duplicate_lines.append(f"Only the first 20 of {groups['duplicate_group_count']} duplicate groups are shown.")
    md = "\n".join(
        [
            "# S6TTS Vintage Augmentation Admission",
            "",
            f"Classification: `{summary['status']}`",
            "",
            f"{status_sentence} It does not authorize model training, public audio release, checkpoint acceptance, or `TRAINING_ELIGIBLE` status.",
            "",
            "## Identity",
            "",
            f"- Corpus: `{summary['corpus_id']}`",
            f"- Source clean view: `{summary['source_clean_view_id']}`",
            f"- Augmented view: `{summary['augmented_view_id']}`",
            f"- Fixed text SHA256: `{summary['fixed_text_sha256']}`",
            f"- TTS engine revision: `{summary['tts_engine_revision']}`",
            f"- Augmentation config SHA256: `{summary['augmentation_config_sha256']}`",
            "",
            "## Counts",
            "",
            f"- Semantic rows: {summary['semantic_rows']}",
            f"- Source clean files: {summary['source_clean_files']}",
            f"- Profiles per row: {summary['augmentation_profiles_per_row']}",
            f"- Expected augmented files: {summary['expected_augmented_files']}",
            f"- Actual augmented files: {summary['actual_augmented_files']}",
            f"- Augmentation failures: {summary['synthesis_or_augmentation_failure_count']}",
            f"- Duplicate paths: {summary['duplicate_path_count']}",
            "",
            "## Duplicate Audio Hashes",
            "",
            *duplicate_lines,
            "",
            "## Audio",
            "",
            f"- Format: mono signed 16-bit PCM WAV at {summary['sample_rate']} Hz",
            f"- Total duration seconds: {summary['total_duration_seconds']}",
            f"- Duration distribution: `{json.dumps(summary['duration_distribution'], sort_keys=True)}`",
            f"- Peak distribution: `{json.dumps(summary['peak_distribution'], sort_keys=True)}`",
            "",
            "## Safety",
            "",
            "- Generated audio committed: no",
            "- Local manifests committed: no",
            "- Raw text committed: no",
            "- Local absolute paths committed: no",
            "- Model training run: no",
            "- Accepted parent: none",
            "- TRAINING_ELIGIBLE issued: no",
            "",
            "## Limitations",
            "",
            "- S6TTS-generated and S6TTS-augmented audio remain internal diagnostic synthetic material only.",
            "- This PR admits an audio bank but does not define a training sampling schedule.",
            "",
        ]
    )
    atomic_write_text(AUGMENTED_REPORT_MD, md)


def validate_public_summary_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if ABSOLUTE_PATH_PATTERN.search(text):
        raise ValueError(f"{path}: contains local absolute path")
    forbidden = ["raw text", '"audio_filepath"', '"text"', "spoken_text", "target_text"]
    for token in forbidden:
        if token in text and path.suffix == ".json":
            raise ValueError(f"{path}: contains forbidden public token {token}")


def summarize_stage(paths: S6AugPaths, *, expected: int, status: str) -> dict[str, Any]:
    rows = []
    if paths.audio_manifest.exists():
        with paths.audio_manifest.open("r", encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp if line.strip()]
    failures = []
    failures_path = paths.run_root / "failures.local.jsonl"
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as fp:
            failures = [json.loads(line) for line in fp if line.strip()]
    summary = {
        "schema_version": "1.0",
        "status": status,
        "rows": len(rows),
        "expected_rows": expected,
        "failures": len(failures),
        "duplicate_path_count": len(rows) - len({row["audio_relative_path"] for row in rows}),
        "duplicate_audio_hash_count": len(rows) - len({row["audio_sha256"] for row in rows}),
    }
    atomic_write_json(paths.summary, summary)
    return summary
