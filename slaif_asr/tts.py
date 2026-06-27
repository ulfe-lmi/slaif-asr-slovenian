from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slaif_asr.config import REPO_ROOT


TTS_CONFIG_PATH = REPO_ROOT / "configs" / "tts" / "piper_sl_si_artur_medium.json"
SUPPORTED_SCHEMA_VERSION = "1.0"
CANDIDATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


@dataclass(frozen=True)
class Candidate:
    schema_version: str
    candidate_id: str
    spoken_text: str
    target_text: str
    language: str
    partition_role: str
    phenomena: tuple[str, ...]
    generation: dict[str, Any]


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_seconds: float
    peak_abs: int
    peak_ratio: float
    sha256: str


def load_tts_config(path: Path = TTS_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def repo_resolve(path_text: str) -> Path:
    return (REPO_ROOT / path_text).resolve()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        fp.write(text)
        temp_name = fp.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def validate_candidate_record(row: dict[str, Any]) -> Candidate:
    if row.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {row.get('schema_version')!r}")
    candidate_id = str(row.get("candidate_id", ""))
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
    language = row.get("language")
    if language != "sl-SI":
        raise ValueError(f"{candidate_id}: language must be sl-SI")
    spoken_text = normalized_text(str(row.get("spoken_text", "")))
    target_text = normalized_text(str(row.get("target_text", "")))
    if not spoken_text or not target_text:
        raise ValueError(f"{candidate_id}: spoken_text and target_text are required")
    if spoken_text != row.get("spoken_text") or target_text != row.get("target_text"):
        raise ValueError(f"{candidate_id}: text must be NFC and whitespace-normalized")
    if spoken_text != target_text:
        raise ValueError(f"{candidate_id}: spoken_text must equal target_text in this PR")
    if len(spoken_text) > 240:
        raise ValueError(f"{candidate_id}: text is too long")
    partition_role = row.get("partition_role")
    if partition_role != "synthetic_smoke":
        raise ValueError(f"{candidate_id}: partition_role must be synthetic_smoke")
    phenomena = row.get("phenomena")
    if not isinstance(phenomena, list) or not all(isinstance(item, str) and item for item in phenomena):
        raise ValueError(f"{candidate_id}: phenomena must be a non-empty string list")
    generation = row.get("generation")
    if not isinstance(generation, dict) or generation.get("system") != "manual-fixture":
        raise ValueError(f"{candidate_id}: generation.system must be manual-fixture")
    return Candidate(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        candidate_id=candidate_id,
        spoken_text=spoken_text,
        target_text=target_text,
        language=language,
        partition_role=partition_role,
        phenomena=tuple(phenomena),
        generation=dict(generation),
    )


def load_candidates(path: Path) -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate = validate_candidate_record(row)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if candidate.candidate_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate candidate_id {candidate.candidate_id}")
            seen.add(candidate.candidate_id)
            candidates.append(candidate)
    if not candidates:
        raise ValueError(f"{path}: no candidates")
    return candidates


def inspect_wav(path: Path) -> WavInfo:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        comptype = wav.getcomptype()
        raw = wav.readframes(frames)
    if comptype != "NONE":
        raise ValueError(f"{path}: expected PCM WAV, got {comptype}")
    if frames <= 0 or not raw:
        raise ValueError(f"{path}: empty audio")
    peak_abs = 0
    if sample_width == 2:
        samples = array("h")
        samples.frombytes(raw)
        if samples.itemsize != 2:
            raise ValueError(f"{path}: unsupported host sample size")
        peak_abs = max(abs(sample) for sample in samples) if samples else 0
        peak_ratio = peak_abs / 32768.0
    else:
        peak_ratio = math.nan
    return WavInfo(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration_seconds=frames / sample_rate,
        peak_abs=peak_abs,
        peak_ratio=peak_ratio,
        sha256=sha256_file(path),
    )


def validate_wav(path: Path, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> WavInfo:
    info = inspect_wav(path)
    if info.channels != channels:
        raise ValueError(f"{path}: expected {channels} channel(s), got {info.channels}")
    if info.sample_rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, got {info.sample_rate}")
    if info.sample_width != sample_width:
        raise ValueError(f"{path}: expected {sample_width}-byte samples, got {info.sample_width}")
    return info


def sox_version() -> str:
    if shutil.which("sox") is None:
        try:
            from importlib import metadata

            audioop_version = metadata.version("audioop-lts")
            return f"python-audioop-lts-ratecv {audioop_version}"
        except metadata.PackageNotFoundError:
            return f"python-audioop-ratecv {sys.version_info.major}.{sys.version_info.minor}"
    completed = subprocess.run(
        ["sox", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "sox --version failed")
    return completed.stdout.splitlines()[0]


def convert_to_16k_pcm(native_wav: Path, final_wav: Path) -> None:
    final_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_wav.with_name(final_wav.stem + ".part" + final_wav.suffix)
    if shutil.which("sox") is None:
        import audioop

        with wave.open(str(native_wav), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            comptype = source.getcomptype()
            frames = source.getnframes()
            raw = source.readframes(frames)
        if comptype != "NONE":
            raise ValueError(f"{native_wav}: expected PCM WAV, got {comptype}")
        if channels != 1 or sample_width != 2:
            raise ValueError(f"{native_wav}: fallback converter requires mono 16-bit PCM")
        converted, _state = audioop.ratecv(raw, sample_width, channels, sample_rate, 16000, None)
        with wave.open(str(temp_path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(converted)
        os.replace(temp_path, final_wav)
        return
    command = [
        "sox",
        str(native_wav),
        "-r",
        "16000",
        "-c",
        "1",
        "-b",
        "16",
        "-e",
        "signed-integer",
        str(temp_path),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(completed.stderr.strip() or "SoX conversion failed")
    os.replace(temp_path, final_wav)


def build_piper_command(
    *,
    piper_python: Path,
    model_path: Path,
    config_path: Path,
    output_file: Path,
    text: str | None = None,
    input_file: Path | None = None,
) -> list[str]:
    if (text is None) == (input_file is None):
        raise ValueError("provide exactly one of text or input_file")
    command = [
        str(piper_python),
        "-m",
        "piper",
        "--cuda",
        "--debug",
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--output-file",
        str(output_file),
    ]
    if input_file is not None:
        command.extend(["--input-file", str(input_file)])
    else:
        command.append(str(text))
    return command


def run_piper_command(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if any(part == "shell=True" for part in command):
        raise ValueError("shell=True is forbidden")
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)


def render_candidates(
    *,
    candidates: list[Candidate],
    config: dict[str, Any],
    output_root: Path | None = None,
) -> dict[str, Any]:
    voice_root = repo_resolve(config["voice"]["local_storage_dir"])
    model_path = voice_root / next(item["path"] for item in config["voice"]["files"] if item["role"] == "model")
    config_path = voice_root / next(item["path"] for item in config["voice"]["files"] if item["role"] == "config")
    piper_python = repo_resolve(config["engine"]["environment"]) / "bin" / "python"
    native_dir = repo_resolve(config["local_artifacts"]["native_audio_dir"]) if output_root is None else output_root / "native-22050"
    final_dir = repo_resolve(config["local_artifacts"]["final_audio_dir"]) if output_root is None else output_root / "final-16000"
    log_dir = repo_resolve(config["local_artifacts"]["log_dir"]) if output_root is None else output_root / "logs"
    provenance_path = repo_resolve(config["local_artifacts"]["provenance_jsonl"]) if output_root is None else output_root / "rendered-records.jsonl"
    manifest_path = repo_resolve(config["local_artifacts"]["manifest_jsonl"]) if output_root is None else output_root / "nemo-manifest.jsonl"
    sidecar_path = repo_resolve(config["local_artifacts"]["manifest_sidecar_json"]) if output_root is None else output_root / "nemo-manifest.provenance.json"
    for required in (piper_python, model_path, config_path):
        if not required.exists():
            raise FileNotFoundError(required)

    env = os.environ.copy()
    conversion = {
        "tool": "sox",
        "version": sox_version(),
        "parameters": ["-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer"],
    }
    provenance_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    per_record: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        native_path = native_dir / f"{candidate.candidate_id}.native.wav"
        final_path = final_dir / f"{candidate.candidate_id}.wav"
        log_path = log_dir / f"{candidate.candidate_id}.piper.log"
        native_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_piper_command(
            piper_python=piper_python,
            model_path=model_path,
            config_path=config_path,
            output_file=native_path,
            text=candidate.spoken_text,
        )
        start = time.perf_counter()
        completed = run_piper_command(command, env=env)
        wall_time = time.perf_counter() - start
        atomic_write_text(log_path, completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"{candidate.candidate_id}: Piper failed with exit {completed.returncode}")
        if "Using CUDA" not in completed.stdout:
            raise RuntimeError(f"{candidate.candidate_id}: Piper log did not confirm CUDA use")
        if "Failed to create CUDAExecutionProvider" in completed.stdout or "CPUExecutionProvider" in completed.stdout:
            raise RuntimeError(f"{candidate.candidate_id}: Piper did not create CUDAExecutionProvider cleanly")
        native_info = validate_wav(native_path, sample_rate=int(config["voice"]["native_sample_rate"]))
        convert_to_16k_pcm(native_path, final_path)
        final_info = validate_wav(final_path, sample_rate=int(config["voice"]["final_asr_sample_rate"]))
        row = {
            "schema_version": "1.0",
            "candidate_id": candidate.candidate_id,
            "audio_filepath": str(final_path.resolve()),
            "duration_seconds": round(final_info.duration_seconds, 6),
            "sample_rate": final_info.sample_rate,
            "channels": final_info.channels,
            "text": candidate.target_text,
            "language": candidate.language,
            "target_lang": candidate.language,
            "partition_role": candidate.partition_role,
            "source_type": "synthetic_tts",
            "audio_sha256": final_info.sha256,
            "native_audio": {
                "path": str(native_path.resolve()),
                "sample_rate": native_info.sample_rate,
                "channels": native_info.channels,
                "sample_width": native_info.sample_width,
                "sha256": native_info.sha256,
            },
            "audio_validation": {
                "final_sample_width": final_info.sample_width,
                "final_peak_abs": final_info.peak_abs,
                "final_peak_ratio": round(final_info.peak_ratio, 6),
                "native_peak_abs": native_info.peak_abs,
                "native_peak_ratio": round(native_info.peak_ratio, 6),
                "conversion": conversion,
            },
            "tts": {
                "engine": config["engine"]["repository_name"],
                "engine_revision": config["engine"]["revision"],
                "engine_license": config["engine"]["license"],
                "voice": config["voice"]["name"],
                "voice_repository": config["voice"]["repository"],
                "voice_revision": config["voice"]["revision"],
                "execution_provider": config["runtime"]["required_execution_provider"],
                "physical_gpu": int(config["runtime"]["physical_gpu"]),
            },
        }
        manifest_rows.append(
            {
                "audio_filepath": str(final_path.resolve()),
                "duration": round(final_info.duration_seconds, 6),
                "text": candidate.target_text,
                "lang": candidate.language,
                "target_lang": candidate.language,
            }
        )
        provenance_rows.append(row)
        per_record.append(
            {
                "candidate_id": candidate.candidate_id,
                "command": command,
                "wall_time_seconds": round(wall_time, 3),
                "native_wav": str(native_path),
                "final_wav": str(final_path),
                "log": str(log_path),
            }
        )

    write_jsonl(provenance_path, provenance_rows)
    write_jsonl(manifest_path, manifest_rows)
    sidecar = {
        "schema_version": "1.0",
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_ids": [candidate.candidate_id for candidate in sorted(candidates, key=lambda item: item.candidate_id)],
        "provenance_jsonl": str(provenance_path.resolve()),
        "manifest_jsonl": str(manifest_path.resolve()),
        "voice": config["voice"]["name"],
        "voice_revision": config["voice"]["revision"],
    }
    atomic_write_json(sidecar_path, sidecar)
    return {
        "candidate_count": len(candidates),
        "successful": len(candidates),
        "failed": 0,
        "total_wall_time_seconds": round(time.perf_counter() - total_start, 3),
        "provenance_jsonl": str(provenance_path),
        "manifest_jsonl": str(manifest_path),
        "manifest_sha256": sidecar["manifest_sha256"],
        "records": per_record,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)
