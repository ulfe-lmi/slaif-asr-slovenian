from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.tts import WavInfo, atomic_write_json, atomic_write_text, sha256_file, validate_wav, write_jsonl


CONFIG_PATH = REPO_ROOT / "configs" / "tts" / "s6tts_sl_si_vintage_v1.json"
PINNED_REPOSITORY = "ulfe-lmi/s6tts"
PINNED_REVISION = "6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c"
TTS_ID = "s6tts-sl-si-vintage-v1"
VIEW_ID = "sl-corpus-v4-s6tts-clean-view-v1"
VOICE_LABEL = "s6tts-sl-si-s6-vintage"
TEXT_SHA256 = "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14"
EXPECTED_ROWS = 16000

ABSOLUTE_PATH_PATTERN = re.compile(r"(^|[\s\"'])/(home|data|data-nvme|synology|tmp|mnt|volume\d?)\b")


@dataclass(frozen=True)
class S6Paths:
    source_dir: Path
    build_dir: Path
    cli_path: Path
    runtime_ini: Path
    run_root: Path
    audio_manifest: Path
    provenance_manifest: Path
    validation: Path
    summary: Path
    logs_dir: Path


@dataclass(frozen=True)
class S6TextRow:
    index: int
    source_id: str
    source_family_id: str
    utterance_family_id: str
    text_hash: str
    spoken_text: str
    target_text: str
    partition_role: str

    @property
    def safe_key(self) -> str:
        return f"s6tts-scale2000-{self.index:05d}"


def repo_path(path_text: str) -> Path:
    return (REPO_ROOT / path_text).resolve()


def local_runs_root() -> Path:
    override = os.environ.get("SLAIF_ASR_RUNS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs"


def local_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "runs":
        return local_runs_root().joinpath(*path.parts[1:])
    return (REPO_ROOT / path).resolve()


def load_s6_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    validate_s6_config(config)
    return config


def validate_s6_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported S6TTS config schema")
    if config.get("tts_id") != TTS_ID:
        raise ValueError("unexpected S6TTS config id")
    engine = config.get("engine", {})
    if engine.get("repository") != PINNED_REPOSITORY:
        raise ValueError("unexpected S6TTS repository")
    if engine.get("revision") != PINNED_REVISION:
        raise ValueError("unexpected S6TTS revision")
    voice = config.get("voice", {})
    if voice.get("label") != VOICE_LABEL or voice.get("name") != "sl-si-s6":
        raise ValueError("unexpected S6TTS voice identity")
    if int(voice.get("native_sample_rate", 0)) != 16000 or int(voice.get("final_asr_sample_rate", 0)) != 16000:
        raise ValueError("S6TTS governed view must be 16 kHz native/final")
    inputs = config.get("inputs", {})
    if inputs.get("corpus_id") != "sl-corpus-v4-gams-16000-training-v1":
        raise ValueError("unexpected S6TTS input corpus")
    if inputs.get("fixed_text_sha256") != TEXT_SHA256:
        raise ValueError("unexpected fixed text SHA256")
    if int(inputs.get("semantic_rows", 0)) != EXPECTED_ROWS:
        raise ValueError("unexpected S6TTS semantic row count")


def s6_paths(config: dict[str, Any]) -> S6Paths:
    engine = config["engine"]
    voice = config["voice"]
    outputs = config["local_outputs"]
    source_dir = repo_path(engine["local_source_dir"])
    return S6Paths(
        source_dir=source_dir,
        build_dir=repo_path(engine["build_dir"]),
        cli_path=source_dir / engine["cli_relative_path"],
        runtime_ini=source_dir / voice["runtime_ini"],
        run_root=local_path(outputs["run_root"]),
        audio_manifest=local_path(outputs["audio_manifest"]),
        provenance_manifest=local_path(outputs["provenance_manifest"]),
        validation=local_path(outputs["validation"]),
        summary=local_path(outputs["summary"]),
        logs_dir=local_path(outputs["logs_dir"]),
    )


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tree_hash(root: Path, *, relative_prefix: str = "") -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in sorted(files):
        rel = Path(relative_prefix) / path.relative_to(root)
        file_hash = sha256_file(path)
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        entries.append({"path": str(rel), "sha256": file_hash, "bytes": path.stat().st_size})
    return {"file_count": len(files), "sha256": digest.hexdigest(), "files": entries}


def load_scale2000_rows(config: dict[str, Any]) -> list[S6TextRow]:
    path = local_path(config["inputs"]["fixed_text_local_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != config["inputs"]["fixed_text_sha256"]:
        raise RuntimeError(f"fixed text SHA256 mismatch: {actual_hash}")
    rows: list[S6TextRow] = []
    with path.open("r", encoding="utf-8") as fp:
        for index, line in enumerate(fp):
            if not line.strip():
                continue
            payload = json.loads(line)
            spoken = str(payload.get("spoken_text", ""))
            target = str(payload.get("target_text", ""))
            if not spoken or spoken != target:
                raise RuntimeError(f"invalid scale-2000 text row at index {index}")
            rows.append(
                S6TextRow(
                    index=index,
                    source_id=str(payload.get("source_id") or payload.get("candidate_id") or f"row-{index:05d}"),
                    source_family_id=str(payload.get("source_family_id") or ""),
                    utterance_family_id=str(payload.get("utterance_family_id") or ""),
                    text_hash=text_sha256(spoken),
                    spoken_text=spoken,
                    target_text=target,
                    partition_role=str(payload.get("partition_role") or "selected_training"),
                )
            )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} scale-2000 rows, found {len(rows)}")
    return rows


def build_s6_command(*, cli_path: Path, ini_path: Path, text: str, output_file: Path) -> list[str]:
    if not text:
        raise ValueError("S6TTS text is required")
    return [str(cli_path), "--ini", str(ini_path), "--text", text, "-o", str(output_file)]


def run_s6_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    if any(part == "shell=True" for part in command):
        raise ValueError("shell=True is forbidden")
    return subprocess.run(list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def s6_output_path(paths: S6Paths, row: S6TextRow) -> Path:
    shard = f"{row.index // 1000:02d}"
    return paths.run_root / "wav" / shard / f"{row.safe_key}.wav"


def synthesize_one(paths: S6Paths, row: S6TextRow, *, overwrite: bool = False) -> dict[str, Any]:
    output = s6_output_path(paths, row)
    log_path = paths.logs_dir / f"{row.safe_key}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        info = validate_wav(output, sample_rate=16000)
        return build_local_row(paths, row, info, wall_time_seconds=0.0, reused=True)
    if output.exists() and overwrite:
        output.unlink()
    temp_output = output.with_suffix(".part.wav")
    temp_output.unlink(missing_ok=True)
    command = build_s6_command(cli_path=paths.cli_path, ini_path=paths.runtime_ini, text=row.spoken_text, output_file=temp_output)
    start = time.perf_counter()
    completed = run_s6_command(command)
    wall_time = time.perf_counter() - start
    atomic_write_text(log_path, completed.stdout)
    if completed.returncode != 0:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"{row.safe_key}: S6TTS failed with exit {completed.returncode}")
    if not temp_output.exists():
        raise FileNotFoundError(temp_output)
    os.replace(temp_output, output)
    info = validate_wav(output, sample_rate=16000)
    return build_local_row(paths, row, info, wall_time_seconds=wall_time, reused=False)


def synthesize_worker(args: tuple[S6Paths, S6TextRow, bool]) -> dict[str, Any]:
    paths, row, overwrite = args
    return synthesize_one(paths, row, overwrite=overwrite)


def synthesize_worker_status(args: tuple[S6Paths, S6TextRow, bool]) -> dict[str, Any]:
    paths, row, overwrite = args
    try:
        return {"ok": True, "row": synthesize_one(paths, row, overwrite=overwrite)}
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "safe_key": row.safe_key,
                "row_index": row.index,
                "text_hash": row.text_hash,
                "reason": exc.__class__.__name__,
                "message": str(exc).replace(str(paths.run_root), "<RUN_ROOT>").replace(str(paths.source_dir), "<S6TTS_SOURCE>"),
            },
        }


def build_local_row(paths: S6Paths, row: S6TextRow, info: WavInfo, *, wall_time_seconds: float, reused: bool) -> dict[str, Any]:
    rel_audio = info.path.relative_to(paths.run_root)
    return {
        "schema_version": "1.0",
        "view_id": VIEW_ID,
        "safe_key": row.safe_key,
        "row_index": row.index,
        "source_id": row.source_id,
        "source_family_id": row.source_family_id,
        "utterance_family_id": row.utterance_family_id,
        "partition_role": row.partition_role,
        "source_type": "synthetic_tts",
        "target_lang": "sl-SI",
        "text": row.target_text,
        "text_hash": row.text_hash,
        "audio_filepath": str(info.path.resolve()),
        "audio_relative_path": str(rel_audio),
        "audio_sha256": info.sha256,
        "duration_seconds": round(info.duration_seconds, 6),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "sample_width": info.sample_width,
        "frames": info.frames,
        "peak_abs": info.peak_abs,
        "peak_ratio": round(info.peak_ratio, 6),
        "synthesis_wall_time_seconds": round(wall_time_seconds, 6),
        "reused_existing_audio": reused,
        "tts": {
            "engine": "s6tts",
            "engine_repository": PINNED_REPOSITORY,
            "engine_revision": PINNED_REVISION,
            "voice_id": "sl-si-s6",
            "voice_label": VOICE_LABEL,
            "runtime_ini": "data/sl-si-s6/sint.ini",
            "output_format": "mono signed 16-bit PCM WAV at 16 kHz",
        },
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"text", "audio_filepath"}}


def write_manifest_pair(paths: S6Paths, rows: list[dict[str, Any]]) -> None:
    write_jsonl(paths.audio_manifest, rows)
    write_jsonl(paths.provenance_manifest, [public_row(row) for row in rows])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(percentile(values, 50), 6),
        "p95": round(percentile(values, 95), 6),
        "max": round(max(values), 6),
    }


SLOVENIAN_NUMBER_WORDS = {
    "0": "nic",
    "1": "ena",
    "2": "dva",
    "3": "tri",
    "4": "stiri",
    "5": "pet",
    "6": "sest",
    "7": "sedem",
    "8": "osem",
    "9": "devet",
    "10": "deset",
    "20": "dvajset",
    "30": "trideset",
    "40": "stirideset",
    "50": "petdeset",
    "60": "sestdeset",
    "70": "sedemdeset",
    "80": "osemdeset",
    "90": "devetdeset",
    "100": "sto",
}


def numeric_spoken_equivalence_key(text: str) -> str:
    """Return a conservative key for digit/word equivalence in local manifests."""
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def replace_number(match: re.Match[str]) -> str:
        return SLOVENIAN_NUMBER_WORDS.get(match.group(0), match.group(0))

    normalized = re.sub(r"\b\d+\b", replace_number, normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def duplicate_audio_hash_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["audio_sha256"]), []).append(row)
    duplicate_groups = {audio_hash: group for audio_hash, group in grouped.items() if len(group) > 1}
    redacted_groups = []
    explained_extra = 0
    unexplained_extra = 0
    for audio_hash, group in sorted(duplicate_groups.items()):
        equivalence_keys = {numeric_spoken_equivalence_key(str(row.get("text", ""))) for row in group}
        explanation = "numeric_normalization_equivalence" if len(equivalence_keys) == 1 else "unexplained"
        extra = len(group) - 1
        if explanation == "numeric_normalization_equivalence":
            explained_extra += extra
        else:
            unexplained_extra += extra
        redacted_groups.append(
            {
                "audio_sha256": audio_hash,
                "count": len(group),
                "explanation": explanation,
                "rows": [
                    {
                        "row_index": int(row["row_index"]),
                        "safe_key": str(row["safe_key"]),
                        "text_sha256": str(row.get("text_hash", "")),
                        "chars": len(str(row.get("text", ""))),
                        "utf8_bytes": len(str(row.get("text", "")).encode("utf-8")),
                        "duration_seconds": row.get("duration_seconds"),
                        "frames": row.get("frames"),
                        "peak_ratio": row.get("peak_ratio"),
                    }
                    for row in sorted(group, key=lambda item: int(item["row_index"]))
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "duplicate_group_count": len(redacted_groups),
        "duplicate_extra_file_count": sum(len(group) - 1 for group in duplicate_groups.values()),
        "explained_duplicate_extra_file_count": explained_extra,
        "unexplained_duplicate_extra_file_count": unexplained_extra,
        "groups": redacted_groups,
    }


def summarize_local_view(config: dict[str, Any], paths: S6Paths) -> dict[str, Any]:
    if not paths.audio_manifest.exists() or not paths.provenance_manifest.exists():
        raise FileNotFoundError("S6TTS local manifests are missing")
    rows: list[dict[str, Any]] = []
    with paths.audio_manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    failures_path = paths.run_root / "failures.local.jsonl"
    failures: list[dict[str, Any]] = []
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    failures.append(json.loads(line))
    duplicate_paths = len(rows) - len({row["audio_relative_path"] for row in rows})
    duplicate_hashes = len(rows) - len({row["audio_sha256"] for row in rows})
    duplicate_groups = duplicate_audio_hash_groups(rows)
    unexplained_duplicate_hashes = int(duplicate_groups["unexplained_duplicate_extra_file_count"])
    durations = [float(row["duration_seconds"]) for row in rows]
    peaks = [float(row["peak_ratio"]) for row in rows]
    issues: dict[str, int] = {}
    if len(rows) != EXPECTED_ROWS:
        issues["row_count_mismatch"] = EXPECTED_ROWS - len(rows)
    if failures:
        by_reason: dict[str, int] = {}
        for failure in failures:
            by_reason[str(failure.get("reason", "unknown"))] = by_reason.get(str(failure.get("reason", "unknown")), 0) + 1
        for reason, count in sorted(by_reason.items()):
            issues[f"synthesis_failure:{reason}"] = count
    if duplicate_paths:
        issues["duplicate_paths"] = duplicate_paths
    if unexplained_duplicate_hashes:
        issues["unexplained_duplicate_audio_hashes"] = unexplained_duplicate_hashes
    runtime_tree = tree_hash(paths.source_dir / "data" / "sl-si-s6", relative_prefix="data/sl-si-s6")
    summary = {
        "schema_version": "1.0",
        "certificate_id": VIEW_ID,
        "corpus_id": config["inputs"]["corpus_id"],
        "view_id": VIEW_ID,
        "status": "S6TTS_SCALE2000_CLEAN_VIEW_AUDIO_ACCEPTED" if not issues else "S6TTS_REJECTED_SYNTHESIS_QUALITY",
        "fixed_text_sha256": config["inputs"]["fixed_text_sha256"],
        "semantic_rows": EXPECTED_ROWS,
        "expected_clean_files": EXPECTED_ROWS,
        "actual_clean_files": len(rows),
        "voice_label": VOICE_LABEL,
        "tts_engine": "s6tts",
        "tts_engine_repository": PINNED_REPOSITORY,
        "tts_engine_revision": PINNED_REVISION,
        "s6tts_runtime_data_hash": runtime_tree["sha256"],
        "s6tts_runtime_data_file_count": runtime_tree["file_count"],
        "s6tts_provenance_certificate": "s6tts-lab-provenance-v1",
        "audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "provenance_manifest_sha256": sha256_file(paths.provenance_manifest),
        "duration_distribution": distribution(durations),
        "peak_distribution": distribution(peaks),
        "total_duration_seconds": round(sum(durations), 6),
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "duplicate_path_count": duplicate_paths,
        "duplicate_audio_hash_count": duplicate_hashes,
        "explained_duplicate_audio_hash_count": int(duplicate_groups["explained_duplicate_extra_file_count"]),
        "unexplained_duplicate_audio_hash_count": unexplained_duplicate_hashes,
        "duplicate_audio_hash_groups_redacted": duplicate_groups,
        "synthesis_failure_count": len(failures),
        "issues_by_reason": issues,
        "local_manifest_committed": False,
        "generated_audio_committed": False,
        "allowed_uses": ["internal synthetic diagnostic audio", "future training only by explicit work order", "aggregate reporting"],
        "forbidden_uses": [
            "public audio release",
            "model release claim",
            "TRAINING_ELIGIBLE",
            "checkpoint acceptance",
            "real-gate acceptance evidence",
        ],
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
        "limitations": [
            "S6TTS voice/runtime provenance is treated as internal diagnostic pending release review.",
            "The view is a synthetic clean voice addition only and does not authorize model training.",
        ],
    }
    validate_public_payload(summary)
    atomic_write_json(paths.validation, {"status": summary["status"], "issues_by_reason": issues})
    atomic_write_json(paths.summary, summary)
    return summary


def validate_public_payload(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden_keys = {"text", "audio_filepath", "local_path", "manifest_rows", "raw_text"}
    if any(f'"{key}"' in text for key in forbidden_keys):
        raise ValueError("public S6TTS payload contains raw text or local path fields")
    if ABSOLUTE_PATH_PATTERN.search(text):
        raise ValueError("public S6TTS payload contains local absolute paths")


def write_public_evidence(summary: dict[str, Any], *, report_json: Path, report_md: Path, certificate_json: Path) -> None:
    validate_public_payload(summary)
    atomic_write_json(certificate_json, summary)
    atomic_write_json(report_json, summary)
    if summary["status"] == "S6TTS_SCALE2000_CLEAN_VIEW_AUDIO_ACCEPTED":
        status_sentence = (
            "This report admits one internal diagnostic S6TTS clean synthetic voice view for the fixed scale-2000 text corpus."
        )
    else:
        status_sentence = (
            "This report records a failed internal diagnostic S6TTS clean synthetic voice-view admission attempt for the fixed scale-2000 text corpus."
        )
    duplicate_lines = []
    duplicate_groups = summary.get("duplicate_audio_hash_groups_redacted", {})
    if duplicate_groups.get("duplicate_group_count"):
        duplicate_lines = [
            "",
            "## Duplicate Audio Hashes",
            "",
            f"- Duplicate groups: {duplicate_groups['duplicate_group_count']}",
            f"- Extra duplicate files: {duplicate_groups['duplicate_extra_file_count']}",
            f"- Explained by numeric normalization: {duplicate_groups['explained_duplicate_extra_file_count']}",
            f"- Unexplained duplicate files: {duplicate_groups['unexplained_duplicate_extra_file_count']}",
            "",
            "| Audio SHA256 | Explanation | Rows | Text SHA256 values | Duration seconds | Frames |",
            "|---|---|---:|---|---:|---:|",
        ]
        for group in duplicate_groups.get("groups", []):
            rows = group["rows"]
            duplicate_lines.append(
                "| `{audio}` | `{explanation}` | {row_indexes} | `{text_hashes}` | {duration} | {frames} |".format(
                    audio=group["audio_sha256"],
                    explanation=group["explanation"],
                    row_indexes=", ".join(str(row["row_index"]) for row in rows),
                    text_hashes=", ".join(row["text_sha256"] for row in rows),
                    duration=rows[0]["duration_seconds"],
                    frames=rows[0]["frames"],
                )
            )
        duplicate_lines.extend(
            [
                "",
                "Raw text for these duplicate cases is retained only in ignored local debugging files and is not committed.",
                "",
            ]
        )
    md = "\n".join(
        [
            "# S6TTS Vintage Clean-View Admission",
            "",
            f"Classification: `{summary['status']}`",
            "",
            f"{status_sentence} It does not authorize model training, public audio release, checkpoint acceptance, or `TRAINING_ELIGIBLE` status.",
            "",
            "## Identity",
            "",
            f"- Corpus: `{summary['corpus_id']}`",
            f"- View: `{summary['view_id']}`",
            f"- Fixed text SHA256: `{summary['fixed_text_sha256']}`",
            f"- TTS engine: `{summary['tts_engine']}` at `{summary['tts_engine_revision']}`",
            f"- Voice label: `{summary['voice_label']}`",
            "",
            "## Counts",
            "",
            f"- Semantic rows: {summary['semantic_rows']}",
            f"- Expected clean files: {summary['expected_clean_files']}",
            f"- Actual clean files: {summary['actual_clean_files']}",
            f"- Duplicate paths: {summary['duplicate_path_count']}",
            f"- Duplicate audio hashes: {summary['duplicate_audio_hash_count']}",
            f"- Explained duplicate audio hashes: {summary.get('explained_duplicate_audio_hash_count', 0)}",
            f"- Unexplained duplicate audio hashes: {summary.get('unexplained_duplicate_audio_hash_count', 0)}",
            f"- Synthesis failures: {summary['synthesis_failure_count']}",
            "",
            *duplicate_lines,
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
            "- Public release authorized: no",
            "- Accepted parent: none",
            "- TRAINING_ELIGIBLE issued: no",
            "",
        ]
    )
    atomic_write_text(report_md, md)


def smoke_rows() -> list[S6TextRow]:
    fixtures = [
        "Čmrlj šviga čez žametno polje.",
        "Špela želi vroč čaj.",
        "Žiga vpraša, kje je ključ.",
    ]
    return [
        S6TextRow(
            index=index,
            source_id=f"smoke-{index}",
            source_family_id="s6tts-smoke",
            utterance_family_id=f"s6tts-smoke-{index}",
            text_hash=text_sha256(text),
            spoken_text=text,
            target_text=text,
            partition_role="synthetic_smoke",
        )
        for index, text in enumerate(fixtures)
    ]


def sample_rows(rows: Sequence[S6TextRow], count: int = 16) -> list[S6TextRow]:
    return list(rows[:count])
