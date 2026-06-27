from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.acoustic_quality import distribution, read_audio_stats
from slaif_asr.batched_streaming import NvidiaSmiMonitor, file_sha256, parse_monitor_csv
from slaif_asr.config import REPO_ROOT
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.real_eval import atomic_write_json, atomic_write_jsonl
from slaif_asr.tts import convert_to_16k_pcm, sox_version


SUPERTONIC_CONFIG_PATH = REPO_ROOT / "configs/tts/supertonic3_sl_multivoice_v1.json"
SUPER_AUDIO_CERTIFICATE_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-supertonic3-multivoice-audio-v1.json"
SUPER_AUDIO_REPORT_JSON = REPO_ROOT / "docs/data-reports/0010-supertonic3-multivoice-acoustic-admission.json"
SUPER_AUDIO_REPORT_MD = REPO_ROOT / "docs/data-reports/0010-supertonic3-multivoice-acoustic-admission.md"
TRAINING_STYLES = ("M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4")
HELD_OUT_STYLES = ("M5", "F5")
ALL_STYLES = (*TRAINING_STYLES[:4], *HELD_OUT_STYLES[:1], *TRAINING_STYLES[4:], *HELD_OUT_STYLES[1:])
REQUIRED_ASSET_RELATIVE_PATHS = (
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    *(f"voice_styles/{style}.json" for style in ALL_STYLES),
)
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
SUPPORTED_TTS_IDS = {
    "supertonic3-sl-multivoice-v1",
    "supertonic3-sl-multivoice-batched-replay-v1",
    "supertonic3-sl-scale200-training-v1",
    "supertonic3-sl-scale2000-training-v1",
    "supertonic3-sl-scale8000-training-v1",
}


@dataclass(frozen=True)
class SupertonicPaths:
    run_root: Path
    native_root: Path
    final_root: Path
    native_manifest: Path
    audio_manifest: Path
    training_audio_manifest: Path
    holdout_audio_manifest: Path
    training_probe_manifest: Path
    exposure_schedule: Path
    validation: Path
    summary: Path
    progress_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class SupertonicTextItem:
    item_id: str
    source_key: str
    partition_role: str
    text: str
    text_sha256: str
    source_id: str
    source_family_id: str
    utterance_family_id: str
    domain: str
    phenomena: tuple[str, ...]
    selection_reason: str | None = None
    selection_rank: int | None = None
    source_audio_sha256: str | None = None
    piper_duration: float | None = None


@dataclass(frozen=True)
class SupertonicBatchPlanItem:
    item: SupertonicTextItem
    voice_style: str
    partition_stage: str
    preprocessed_text_length: int
    source_key_hash: str
    identity: str


@dataclass(frozen=True)
class _BatchedStyle:
    ttl: Any
    dp: Any


@dataclass(frozen=True)
class _ListArray:
    rows: tuple[tuple[Any, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        if not self.rows:
            return (0, 0)
        return (len(self.rows), len(self.rows[0]))

    def tolist(self) -> list[list[Any]]:
        return [list(row) for row in self.rows]


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_supertonic_config(path: Path = SUPERTONIC_CONFIG_PATH) -> dict[str, Any]:
    config = read_json(repo_path(path))
    if config.get("tts_id") not in SUPPORTED_TTS_IDS:
        raise ValueError("unexpected Supertonic config id")
    package = config.get("package", {})
    model = config.get("model", {})
    synthesis = config.get("synthesis", {})
    voices = config.get("voice_styles", {})
    if package.get("name") != "supertonic" or package.get("version") != "1.3.1":
        raise ValueError("Supertonic package pin must be supertonic==1.3.1")
    if model.get("repository") != "Supertone/supertonic-3" or model.get("revision") != "724fb5abbf5502583fb520898d45929e62f02c0b":
        raise ValueError("Supertonic model identity mismatch")
    if model.get("auto_download") is not False:
        raise ValueError("governed Supertonic execution must use auto_download=false")
    if config.get("language", {}).get("code") != "sl" or config.get("language", {}).get("fallback_na_allowed") is not False:
        raise ValueError("Supertonic synthesis must use explicit sl language and forbid na fallback")
    if tuple(voices.get("training", [])) != TRAINING_STYLES:
        raise ValueError("Supertonic training voice allocation mismatch")
    if tuple(voices.get("held_out", [])) != HELD_OUT_STYLES:
        raise ValueError("Supertonic held-out voice allocation mismatch")
    expected = {"total_steps": 8, "speed": 1.05, "max_chunk_length": 300, "silence_duration": 0.3}
    for key, value in expected.items():
        if synthesis.get(key) != value:
            raise ValueError(f"Supertonic synthesis.{key} must be {value}")
    if synthesis.get("expression_tags_allowed") is not False or synthesis.get("custom_voice_builder_allowed") is not False:
        raise ValueError("expression tags and custom voice styles are forbidden")
    runtime = config.get("runtime", {})
    if runtime.get("execution_device") not in {"cpu", "cuda"}:
        raise ValueError("Supertonic runtime.execution_device must be cpu or cuda")
    if runtime.get("execution_device") == "cuda":
        if runtime.get("required_provider") != "CUDAExecutionProvider":
            raise ValueError("GPU Supertonic execution requires CUDAExecutionProvider")
        if runtime.get("cpu_provider_fallback_allowed") is not False:
            raise ValueError("GPU Supertonic execution must reject CPU provider fallback")
    return config


def supertonic_paths(config: dict[str, Any]) -> SupertonicPaths:
    outputs = config["local_outputs"]
    audio_manifest = repo_path(outputs["audio_manifest"])
    return SupertonicPaths(
        run_root=repo_path(outputs["run_root"]),
        native_root=repo_path(outputs["run_root"]) / "native-44100",
        final_root=repo_path(outputs["run_root"]) / "final-16000",
        native_manifest=repo_path(outputs["native_manifest"]),
        audio_manifest=audio_manifest,
        training_audio_manifest=repo_path(outputs.get("training_audio_manifest", audio_manifest.with_name("training-audio-manifest.local.jsonl"))),
        holdout_audio_manifest=repo_path(outputs.get("holdout_audio_manifest", audio_manifest.with_name("holdout-audio-manifest.local.jsonl"))),
        training_probe_manifest=repo_path(outputs["training_probe_manifest"]),
        exposure_schedule=repo_path(outputs["exposure_schedule"]),
        validation=repo_path(outputs["validation"]),
        summary=repo_path(outputs["summary"]),
        progress_dir=repo_path(outputs["progress_dir"]),
        logs_dir=repo_path(outputs["logs_dir"]),
    )


def model_dir(config: dict[str, Any]) -> Path:
    return repo_path(config["model"]["local_dir"])


def venv_python(config: dict[str, Any]) -> Path:
    return repo_path(config["package"]["environment"]) / "bin" / "python"


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def optional_file_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.exists() else None


def supertonic_execution_device(config: dict[str, Any]) -> str:
    return str(config.get("runtime", {}).get("execution_device", "cpu"))


def assert_supertonic_runtime_environment(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    device = supertonic_execution_device(config)
    if device == "cuda":
        expected = str(runtime.get("cuda_visible_devices", "1"))
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
            raise RuntimeError(f"Supertonic GPU synthesis must run with CUDA_VISIBLE_DEVICES={expected!r}")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None}:
        raise RuntimeError('Supertonic CPU synthesis must run with CUDA_VISIBLE_DEVICES=""')


def maybe_reexec_with_supertonic_cuda_libraries(config: dict[str, Any]) -> None:
    if supertonic_execution_device(config) != "cuda":
        return
    if os.environ.get("SUPERTONIC_CUDA_LIB_PATH_READY") == "1":
        return
    site_packages_roots = sorted((Path(sys.prefix) / "lib").glob("python*/site-packages"))
    lib_dirs: list[str] = []
    for root in site_packages_roots:
        for lib_dir in sorted((root / "nvidia").glob("*/*")):
            if lib_dir.name == "lib" and lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))
    if not lib_dirs:
        return
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, existing] if existing else lib_dirs)
    env["SUPERTONIC_CUDA_LIB_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def assert_supertonic_cpu_environment(config: dict[str, Any] | None = None) -> None:
    assert_supertonic_runtime_environment(config or {"runtime": {"execution_device": "cpu"}})


def configure_supertonic_onnx_providers(config: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort

    runtime = config.get("runtime", {})
    requested = str(runtime.get("required_provider", "CPUExecutionProvider"))
    available = list(ort.get_available_providers())
    if requested not in available:
        raise RuntimeError(f"required Supertonic ONNX provider {requested} not available; available={available}")
    providers = [requested]
    if runtime.get("cpu_provider_fallback_allowed") is True and requested != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    import supertonic.config as supertonic_config
    import supertonic.loader as supertonic_loader

    supertonic_config.DEFAULT_ONNX_PROVIDERS = providers
    supertonic_loader.DEFAULT_ONNX_PROVIDERS = providers
    return {"available_providers": available, "requested_providers": providers, "primary_provider": requested}


def supertonic_session_provider_summary(tts: Any, config: dict[str, Any]) -> dict[str, Any]:
    required = str(config.get("runtime", {}).get("required_provider", "CPUExecutionProvider"))
    sessions = {
        "duration_predictor": getattr(tts.model, "dp_ort"),
        "text_encoder": getattr(tts.model, "text_enc_ort"),
        "vector_estimator": getattr(tts.model, "vector_est_ort"),
        "vocoder": getattr(tts.model, "vocoder_ort"),
    }
    providers: dict[str, list[str]] = {}
    for name, session in sessions.items():
        if hasattr(session, "disable_fallback") and config.get("runtime", {}).get("cpu_provider_fallback_allowed") is False:
            session.disable_fallback()
        session_providers = list(session.get_providers())
        providers[name] = session_providers
        if not session_providers or session_providers[0] != required:
            raise RuntimeError(f"Supertonic session {name} is not using {required} as primary provider: {session_providers}")
    return {"required_provider": required, "sessions": providers}


def assert_public_payload_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public Supertonic payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public Supertonic payload contains row IDs or local paths")


def verify_input_identities(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    if "scale200_fixed_text" in inputs:
        fixed_text = repo_path(inputs["scale200_fixed_text"])
        if file_sha256(fixed_text) != inputs["scale200_fixed_text_sha256"]:
            raise RuntimeError("scale-200 fixed text SHA mismatch")
        selected_rows = read_jsonl(fixed_text)
        if len(selected_rows) != int(inputs["selected_rows"]):
            raise RuntimeError("scale-200 selected row count mismatch")
        return {
            "scale200_fixed_text_sha256": file_sha256(fixed_text),
            "selected_rows": len(selected_rows),
            "include_holdout": bool(inputs.get("include_holdout", False)),
        }
    selected = repo_path(inputs["selected_training_manifest"])
    selected_audio = repo_path(inputs["selected_training_audio_manifest"])
    holdout = repo_path(inputs["synthetic_holdout_text"])
    if file_sha256(selected) != inputs["selected_training_manifest_sha256"]:
        raise RuntimeError("selected-training manifest SHA mismatch")
    if file_sha256(selected_audio) != inputs["selected_training_audio_manifest_sha256"]:
        raise RuntimeError("selected-training audio manifest SHA mismatch")
    if file_sha256(holdout) != inputs["synthetic_holdout_text_sha256"]:
        raise RuntimeError("synthetic holdout text SHA mismatch")
    selected_rows = read_jsonl(selected)
    holdout_rows = read_jsonl(holdout)
    if len(selected_rows) != int(inputs["selected_rows"]):
        raise RuntimeError("selected-training row count mismatch")
    if len(holdout_rows) != int(inputs["synthetic_holdout_rows"]):
        raise RuntimeError("holdout row count mismatch")
    return {
        "selected_training_manifest_sha256": file_sha256(selected),
        "selected_training_audio_manifest_sha256": file_sha256(selected_audio),
        "selected_rows": len(selected_rows),
        "holdout_text_sha256": file_sha256(holdout),
        "holdout_rows": len(holdout_rows),
    }


def load_selected_items(config: dict[str, Any]) -> list[SupertonicTextItem]:
    inputs = config["inputs"]
    if "scale200_fixed_text" in inputs:
        rows = read_jsonl(repo_path(inputs["scale200_fixed_text"]))
        output = []
        for row in rows:
            cid = str(row["candidate_id"])
            text = str(row["target_text"])
            output.append(
                SupertonicTextItem(
                    item_id=cid,
                    source_key=cid,
                    partition_role="selected_training",
                    text=text,
                    text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    source_id=str(row["source_id"]),
                    source_family_id=str(row["source_family_id"]),
                    utterance_family_id=str(row["utterance_family_id"]),
                    domain=str(row.get("domain", "")),
                    phenomena=tuple(str(item) for item in row.get("phenomena", [])),
                    selection_reason="scale200",
                    selection_rank=0,
                )
            )
        return sorted(output, key=lambda item: item.item_id)
    selected_rows = read_jsonl(repo_path(inputs["selected_training_manifest"]))
    selected_audio_rows = read_jsonl(repo_path(inputs["selected_training_audio_manifest"]))
    audio_by_id = {str(row["selected_training_id"]): row for row in selected_audio_rows}
    output: list[SupertonicTextItem] = []
    for row in selected_rows:
        selected_id = str(row["selected_training_id"])
        audio_row = audio_by_id[selected_id]
        output.append(
            SupertonicTextItem(
                item_id=selected_id,
                source_key=selected_id,
                partition_role="selected_training",
                text=str(row["text"]),
                text_sha256=str(row["text_sha256"]),
                source_id=str(audio_row["source_id"]),
                source_family_id=str(audio_row["source_family_id"]),
                utterance_family_id=str(audio_row["utterance_family_id"]),
                domain=str(audio_row.get("domain", "")),
                phenomena=tuple(str(item) for item in audio_row.get("phenomena", [])),
                selection_reason=str(row.get("selection_reason", "")),
                selection_rank=int(row.get("selection_rank", 0)),
                source_audio_sha256=str(row.get("audio_sha256", "")),
                piper_duration=float(row["duration"]),
            )
        )
    return sorted(output, key=lambda item: item.item_id)


def load_holdout_items(config: dict[str, Any]) -> list[SupertonicTextItem]:
    inputs = config["inputs"]
    rows = read_jsonl(repo_path(inputs["synthetic_holdout_text"]))
    output: list[SupertonicTextItem] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        text = str(row["target_text"])
        output.append(
            SupertonicTextItem(
                item_id=candidate_id,
                source_key=candidate_id,
                partition_role="synthetic_holdout",
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source_id=str(row["source_id"]),
                source_family_id=str(row["source_family_id"]),
                utterance_family_id=str(row["utterance_family_id"]),
                domain=str(row.get("domain", "")),
                phenomena=tuple(str(item) for item in row.get("phenomena", [])),
            )
        )
    return sorted(output, key=lambda item: item.item_id)


def asset_file_hashes(asset_root: Path) -> dict[str, str]:
    if not asset_root.exists():
        raise FileNotFoundError(asset_root)
    hashes: dict[str, str] = {}
    for relative in REQUIRED_ASSET_RELATIVE_PATHS:
        path = asset_root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[relative] = file_sha256(path)
    return hashes


def asset_tree_sha256(asset_hashes: dict[str, str]) -> str:
    lines = [f"{relative}\t{digest}" for relative, digest in sorted(asset_hashes.items())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def verify_assets(config: dict[str, Any]) -> dict[str, Any]:
    root = model_dir(config)
    hashes = asset_file_hashes(root)
    style_hashes = {style: hashes[f"voice_styles/{style}.json"] for style in ALL_STYLES}
    return {
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "model_dir_exists": root.exists(),
        "required_asset_count": len(REQUIRED_ASSET_RELATIVE_PATHS),
        "onnx_files": {key: hashes[key] for key in REQUIRED_ASSET_RELATIVE_PATHS if key.endswith(".onnx")},
        "config_files": {key: hashes[key] for key in ("onnx/tts.json", "onnx/unicode_indexer.json")},
        "voice_style_hashes": style_hashes,
        "asset_tree_sha256": asset_tree_sha256(hashes),
    }


def runtime_versions(config: dict[str, Any]) -> dict[str, Any]:
    python = venv_python(config)
    if not python.exists():
        return {"python": str(python), "status": "missing"}
    code = (
        "import importlib.metadata as m, json, pathlib\n"
        "pkgs=['supertonic','onnxruntime','onnxruntime-gpu','numpy','soundfile','huggingface-hub','audioop-lts']\n"
        "print(json.dumps({p:m.version(p) for p in pkgs}, sort_keys=True))\n"
    )
    completed = subprocess.run([str(python), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return {"python": str(python), "status": "failed", "stderr": completed.stderr.strip()}
    versions = json.loads(completed.stdout)
    return {"python": str(python), "status": "present", "packages": versions}


def package_wheel_hashes(config: dict[str, Any]) -> dict[str, str]:
    python = venv_python(config)
    packages = ["supertonic", "onnxruntime", "onnxruntime-gpu", "numpy", "soundfile", "huggingface-hub", "audioop-lts"]
    if not python.exists():
        return {}
    script = (
        "import importlib.metadata as m, json, pathlib, hashlib\n"
        f"packages={packages!r}\n"
        "out={}\n"
        "for pkg in packages:\n"
        "  dist=m.distribution(pkg)\n"
        "  record=None\n"
        "  for f in dist.files or []:\n"
        "    if str(f).endswith('RECORD'):\n"
        "      record=dist.locate_file(f); break\n"
        "  if record and pathlib.Path(record).exists():\n"
        "    out[pkg]=hashlib.sha256(pathlib.Path(record).read_bytes()).hexdigest()\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    completed = subprocess.run([str(python), "-c", script], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return {}
    return json.loads(completed.stdout)


def download_assets(config: dict[str, Any], revision: str) -> dict[str, Any]:
    if revision != config["model"]["revision"]:
        raise ValueError("download revision must match config")
    target = model_dir(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp_text:
        tmp = Path(tmp_text) / "download"
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=config["model"]["repository"], revision=revision, local_dir=str(tmp))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(tmp), str(target))
    assets = verify_assets(config)
    atomic_write_json(supertonic_paths(config).run_root / "asset-tree.local.json", assets)
    return assets


def _safe_temp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.stem}.part.{os.getpid()}{path.suffix}")


def _write_native_wav(path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = np.asarray(waveform).squeeze()
    temp = _safe_temp(path)
    sf.write(str(temp), arr, int(sample_rate), subtype="PCM_16")
    os.replace(temp, path)


def _duration_value(value: Any) -> float:
    try:
        return float(value[0])
    except Exception:
        return float(value)


def build_variant_plan(config: dict[str, Any], partition: str) -> list[tuple[SupertonicTextItem, str]]:
    if partition == "training":
        return [(item, style) for item in load_selected_items(config) for style in TRAINING_STYLES]
    if partition == "holdout":
        if config.get("inputs", {}).get("include_holdout") is False:
            return []
        return [(item, style) for item in load_holdout_items(config) for style in HELD_OUT_STYLES]
    raise ValueError(f"unsupported partition: {partition}")


def _preprocessed_text_length(text: str) -> int:
    return len(" ".join(text.split()))


def build_batched_variant_plan(config: dict[str, Any]) -> list[SupertonicBatchPlanItem]:
    items: list[SupertonicBatchPlanItem] = []
    for partition in ("training", "holdout"):
        for item, voice_style in build_variant_plan(config, partition):
            source_key_hash = stable_sha256(f"{item.partition_role}:{item.source_key}")
            identity = f"{item.partition_role}\t{item.source_key}\t{voice_style}"
            items.append(
                SupertonicBatchPlanItem(
                    item=item,
                    voice_style=voice_style,
                    partition_stage=partition,
                    preprocessed_text_length=_preprocessed_text_length(item.text),
                    source_key_hash=source_key_hash,
                    identity=identity,
                )
            )
    sorted_items = sorted(
        items,
        key=lambda row: (
            row.preprocessed_text_length,
            row.item.partition_role,
            row.source_key_hash,
            row.voice_style,
        ),
    )
    source_shard = config.get("source_shard")
    if source_shard:
        worker_index = int(source_shard["worker_index"])
        worker_count = int(source_shard["worker_count"])
        if worker_count <= 0 or not (0 <= worker_index < worker_count):
            raise ValueError("invalid Supertonic source shard")
        source_keys = sorted({row.item.source_key for row in sorted_items}, key=stable_sha256)
        allowed = {key for index, key in enumerate(source_keys) if index % worker_count == worker_index}
        sorted_items = [row for row in sorted_items if row.item.source_key in allowed]
    return sorted_items


def partition_batched_plan(plan: Sequence[SupertonicBatchPlanItem], batch_size: int) -> list[list[SupertonicBatchPlanItem]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [list(plan[index : index + batch_size]) for index in range(0, len(plan), batch_size)]


def batch_identity_sha256(batch: Sequence[SupertonicBatchPlanItem]) -> str:
    return hashlib.sha256(("\n".join(row.identity for row in batch) + "\n").encode("utf-8")).hexdigest()


def deterministic_batch_seed(*, experiment_seed: int, batch_index: int, identity_sha256: str) -> int:
    digest = hashlib.sha256(f"{experiment_seed}:{batch_index}:{identity_sha256}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _style_batch(styles: dict[str, Any], voice_styles: Sequence[str]) -> _BatchedStyle:
    values_ttl = [styles[style].ttl for style in voice_styles]
    values_dp = [styles[style].dp for style in voice_styles]

    try:
        import numpy as np
    except ModuleNotFoundError:
        def concat(values: Sequence[Any]) -> _ListArray:
            rows: list[tuple[Any, ...]] = []
            for value in values:
                raw_rows = value.tolist() if hasattr(value, "tolist") else value
                rows.extend(tuple(row) for row in raw_rows)
            return _ListArray(tuple(rows))

        return _BatchedStyle(ttl=concat(values_ttl), dp=concat(values_dp))

    return _BatchedStyle(
        ttl=np.concatenate(values_ttl, axis=0),
        dp=np.concatenate(values_dp, axis=0),
    )


def _waveform_for_row(waveforms: Any, durations: Any, row_index: int, sample_rate: int) -> Any:
    import numpy as np

    wav = np.asarray(waveforms)
    if wav.ndim == 1:
        wav = wav.reshape(1, -1)
    duration = _duration_value(durations[row_index])
    frames = max(1, min(wav.shape[1], int(math.ceil(duration * sample_rate))))
    return wav[row_index, :frames]


def _is_oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda_oom" in text or "cuda oom" in text


def _make_native_row(
    *,
    config: dict[str, Any],
    assets: dict[str, Any],
    provider_summary: dict[str, Any],
    paths: SupertonicPaths,
    plan_item: SupertonicBatchPlanItem,
    waveform: Any,
    duration: Any,
    wall_seconds: float,
) -> dict[str, Any]:
    item = plan_item.item
    voice_style = plan_item.voice_style
    relative = _variant_relative_name(item, voice_style)
    native_path = paths.native_root / relative
    if native_path.exists():
        raise FileExistsError(f"native Supertonic output already exists: {native_path}")
    _write_native_wav(native_path, waveform, int(config["synthesis"]["native_sample_rate"]))
    native_stats = read_audio_stats(native_path)
    return {
        "schema_version": "1.0",
        "source_key": item.source_key,
        "partition_role": item.partition_role,
        "voice_style_id": voice_style,
        "voice_style_json_sha256": assets["voice_style_hashes"][voice_style],
        "source_text_sha256": item.text_sha256,
        "source_audio_sha256": item.source_audio_sha256,
        "utterance_family_id": item.utterance_family_id,
        "source_family_id": item.source_family_id,
        "source_id": item.source_id,
        "domain": item.domain,
        "phenomena": list(item.phenomena),
        "selection_reason": item.selection_reason,
        "selection_rank": item.selection_rank,
        "native_audio_filepath": str(native_path.resolve()),
        "native_audio_sha256": native_stats.sha256,
        "native_sample_rate": native_stats.sample_rate,
        "native_channels": native_stats.channels,
        "native_sample_width": native_stats.sample_width,
        "native_frames": native_stats.frames,
        "native_duration_seconds": round(native_stats.duration_seconds, 6),
        "supertonic_duration_seconds": round(_duration_value(duration), 6),
        "tts": supertonic_provenance(config, assets, voice_style),
        "onnx_runtime": provider_summary,
        "runtime": {"synthesis_wall_time_seconds": round(wall_seconds, 6)},
        "partition_stage": plan_item.partition_stage,
    }


def _convert_native_row(paths: SupertonicPaths, row: dict[str, Any]) -> dict[str, Any]:
    native_path = Path(row["native_audio_filepath"])
    relative = native_path.relative_to(paths.native_root)
    final_path = paths.final_root / relative
    if final_path.exists():
        raise FileExistsError(f"final Supertonic output already exists: {final_path}")
    convert_started = time.perf_counter()
    convert_to_16k_pcm(native_path, final_path)
    conversion_wall = time.perf_counter() - convert_started
    stats_started = time.perf_counter()
    native_stats = read_audio_stats(native_path)
    final_stats = read_audio_stats(final_path)
    stats_wall = time.perf_counter() - stats_started
    converted = dict(row)
    converted.update(
        {
            "audio_filepath": str(final_path.resolve()),
            "audio_sha256": final_stats.sha256,
            "sample_rate": final_stats.sample_rate,
            "channels": final_stats.channels,
            "sample_width": final_stats.sample_width,
            "frames": final_stats.frames,
            "duration_seconds": round(final_stats.duration_seconds, 6),
            "target_text_sha256": row["source_text_sha256"],
            "language": "sl-SI",
            "target_lang": "sl-SI",
            "source_type": "synthetic_tts",
            "audio_validation": {
                "native_peak_ratio": round(native_stats.peak_ratio, 6),
                "final_peak_ratio": round(final_stats.peak_ratio, 6),
                "conversion": {
                    "tool": "sox",
                    "version": sox_version(),
                    "parameters": ["-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer"],
                },
            },
            "parallel_stage": {
                "conversion_wall_seconds": round(conversion_wall, 6),
                "stat_wall_seconds": round(stats_wall, 6),
            },
        }
    )
    return converted


def _variant_relative_name(item: SupertonicTextItem, voice_style: str) -> str:
    safe = item.item_id.replace("/", "_")
    return f"{item.partition_role}/{voice_style}/{safe}.{voice_style}.wav"


def synthesize_partition(
    config: dict[str, Any],
    *,
    partition: str,
    progress_interval_seconds: float,
    tts_factory: Any | None = None,
) -> dict[str, Any]:
    assert_supertonic_runtime_environment(config)
    assets = verify_assets(config)
    paths = supertonic_paths(config)
    plan = build_variant_plan(config, partition)
    reporter = LiveProgressReporter(
        stage=f"synthesize-{partition}",
        ndjson_path=paths.progress_dir / f"synthesize-{partition}.local.ndjson",
    )
    reporter.start(f"synthesizing {partition}")
    real_supertonic_runtime = tts_factory is None
    if tts_factory is None:
        from supertonic import TTS

        provider_setup = configure_supertonic_onnx_providers(config)
        tts_factory = TTS
    else:
        provider_setup = {"available_providers": [], "requested_providers": [], "primary_provider": "fixture"}
    with heartbeat_thread(reporter, interval_seconds=10.0, message="Supertonic model load in progress"):
        tts = tts_factory(model=config["model"]["name"], model_dir=str(model_dir(config)), auto_download=False)
    provider_summary = supertonic_session_provider_summary(tts, config) if real_supertonic_runtime else provider_setup
    if int(getattr(tts, "sample_rate", 0)) != int(config["synthesis"]["native_sample_rate"]):
        raise RuntimeError(f"Supertonic native sample rate mismatch: {getattr(tts, 'sample_rate', None)}")
    styles = {style: tts.get_voice_style(style) for style in ALL_STYLES}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_emit = started
    for index, (item, voice_style) in enumerate(plan, start=1):
        relative = _variant_relative_name(item, voice_style)
        native_path = paths.native_root / relative
        start = time.perf_counter()
        waveform, duration = tts.synthesize(
            item.text,
            voice_style=styles[voice_style],
            total_steps=int(config["synthesis"]["total_steps"]),
            speed=float(config["synthesis"]["speed"]),
            max_chunk_length=int(config["synthesis"]["max_chunk_length"]),
            silence_duration=float(config["synthesis"]["silence_duration"]),
            lang=config["language"]["code"],
            verbose=False,
        )
        wall = time.perf_counter() - start
        _write_native_wav(native_path, waveform, int(config["synthesis"]["native_sample_rate"]))
        native_stats = read_audio_stats(native_path)
        rows.append(
            {
                "schema_version": "1.0",
                "source_key": item.source_key,
                "partition_role": item.partition_role,
                "voice_style_id": voice_style,
                "voice_style_json_sha256": assets["voice_style_hashes"][voice_style],
                "source_text_sha256": item.text_sha256,
                "source_audio_sha256": item.source_audio_sha256,
                "utterance_family_id": item.utterance_family_id,
                "source_family_id": item.source_family_id,
                "source_id": item.source_id,
                "domain": item.domain,
                "phenomena": list(item.phenomena),
                "selection_reason": item.selection_reason,
                "selection_rank": item.selection_rank,
                "native_audio_filepath": str(native_path.resolve()),
                "native_audio_sha256": native_stats.sha256,
                "native_sample_rate": native_stats.sample_rate,
                "native_channels": native_stats.channels,
                "native_sample_width": native_stats.sample_width,
                "native_frames": native_stats.frames,
                "native_duration_seconds": round(native_stats.duration_seconds, 6),
                "supertonic_duration_seconds": round(_duration_value(duration), 6),
                "tts": supertonic_provenance(config, assets, voice_style),
                "onnx_runtime": provider_summary,
                "runtime": {"synthesis_wall_time_seconds": round(wall, 6)},
            }
        )
        now = time.perf_counter()
        if index % 25 == 0 or now - last_emit >= progress_interval_seconds:
            elapsed = now - started
            reporter.progress(
                processed_rows=index,
                total_rows=len(plan),
                examples_per_second=round(index / elapsed, 6) if elapsed else None,
            )
            last_emit = now
    existing = []
    if paths.native_manifest.exists():
        existing = [row for row in read_jsonl(paths.native_manifest) if row.get("partition_stage") != partition]
    for row in rows:
        row["partition_stage"] = partition
    atomic_write_jsonl(paths.native_manifest, [*existing, *rows])
    wall = time.perf_counter() - started
    reporter.complete(processed_rows=len(plan), total_rows=len(plan), message="synthesis complete")
    return {
        "partition": partition,
        "requested": len(plan),
        "generated": len(rows),
        "wall_time_seconds": round(wall, 6),
        "rows_per_minute": round(len(rows) / wall * 60.0, 6) if wall else None,
        "native_audio_seconds_per_wall_second": round(sum(row["native_duration_seconds"] for row in rows) / wall, 6) if wall else None,
    }


def _synthesize_batched_core(
    *,
    config: dict[str, Any],
    assets: dict[str, Any],
    provider_summary: dict[str, Any],
    paths: SupertonicPaths,
    tts: Any,
    styles: dict[str, Any],
    batch: Sequence[SupertonicBatchPlanItem],
    batch_index: int,
    experiment_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    identity = batch_identity_sha256(batch)
    seed = deterministic_batch_seed(
        experiment_seed=experiment_seed,
        batch_index=batch_index,
        identity_sha256=identity,
    )
    np.random.seed(seed)
    style = _style_batch(styles, [row.voice_style for row in batch])
    texts = [row.item.text for row in batch]
    started = time.perf_counter()
    waveforms, durations = tts.model(
        texts,
        style,
        total_step=int(config["synthesis"]["total_steps"]),
        speed=float(config["synthesis"]["speed"]),
        lang=config["language"]["code"],
    )
    wall = time.perf_counter() - started
    rows = []
    for row_index, plan_item in enumerate(batch):
        rows.append(
            _make_native_row(
                config=config,
                assets=assets,
                provider_summary=provider_summary,
                paths=paths,
                plan_item=plan_item,
                waveform=_waveform_for_row(waveforms, durations, row_index, int(config["synthesis"]["native_sample_rate"])),
                duration=durations[row_index],
                wall_seconds=wall / max(1, len(batch)),
            )
        )
    return rows, {
        "batch_index": batch_index,
        "requested_size": len(batch),
        "actual_size": len(batch),
        "identity_sha256": identity,
        "seed": seed,
        "wall_time_seconds": round(wall, 6),
        "native_audio_seconds": round(sum(float(row["native_duration_seconds"]) for row in rows), 6),
        "oom_fallback": False,
    }


def _synthesize_batched_with_fallback(
    *,
    config: dict[str, Any],
    assets: dict[str, Any],
    provider_summary: dict[str, Any],
    paths: SupertonicPaths,
    tts: Any,
    styles: dict[str, Any],
    batch: Sequence[SupertonicBatchPlanItem],
    batch_index: int,
    experiment_seed: int,
    fallback_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        rows, summary = _synthesize_batched_core(
            config=config,
            assets=assets,
            provider_summary=provider_summary,
            paths=paths,
            tts=tts,
            styles=styles,
            batch=batch,
            batch_index=batch_index,
            experiment_seed=experiment_seed,
        )
        return rows, [summary]
    except Exception as exc:
        if not _is_oom_error(exc) or len(batch) <= fallback_size:
            raise
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for fallback_index, start in enumerate(range(0, len(batch), fallback_size), start=1):
        subbatch = list(batch[start : start + fallback_size])
        if len(subbatch) > fallback_size:
            raise RuntimeError("invalid Supertonic fallback batch")
        sub_rows, sub_summary = _synthesize_batched_core(
            config=config,
            assets=assets,
            provider_summary=provider_summary,
            paths=paths,
            tts=tts,
            styles=styles,
            batch=subbatch,
            batch_index=(batch_index * 1000) + fallback_index,
            experiment_seed=experiment_seed,
        )
        sub_summary["oom_fallback"] = True
        sub_summary["original_batch_index"] = batch_index
        rows.extend(sub_rows)
        summaries.append(sub_summary)
    return rows, summaries


def synthesize_batched_supertonic_audio(
    config: dict[str, Any],
    *,
    progress_interval_seconds: float,
    tts_factory: Any | None = None,
) -> dict[str, Any]:
    """Synthesize all Supertonic variants through native model batches.

    The function writes native WAVs, converts them to 16 kHz in a bounded CPU
    worker pool while later GPU batches synthesize, and emits the same manifests
    consumed by the existing Supertonic training implementation.
    """

    assert_supertonic_runtime_environment(config)
    verify_input_identities(config)
    assets = verify_assets(config)
    paths = supertonic_paths(config)
    batch_config = config.get("batch_synthesis", {})
    batch_size = int(batch_config.get("batch_size", 32))
    fallback_size = int(batch_config.get("oom_fallback_batch_size", 16))
    if batch_size != 32 or fallback_size != 16:
        raise ValueError("the batched replay requires synthesis batch size 32 with OOM fallback 16")
    plan = build_batched_variant_plan(config)
    expected_total = int(config.get("expected_counts", {}).get("total_variants", 1472))
    if len(plan) != expected_total:
        raise RuntimeError(f"expected {expected_total} Supertonic variants, found {len(plan)}")
    batches = partition_batched_plan(plan, batch_size)
    workers = min(int(config.get("conversion", {}).get("max_workers", 16)), os.cpu_count() or 1)
    max_pending = max(workers * int(config.get("conversion", {}).get("bounded_queue_multiplier", 2)), workers)
    reporter = LiveProgressReporter(stage="synthesize-batched", ndjson_path=paths.progress_dir / "synthesize-batched.local.ndjson")
    existing_manifests = [
        paths.native_manifest,
        paths.audio_manifest,
        paths.training_audio_manifest,
        paths.holdout_audio_manifest,
        paths.training_probe_manifest,
    ]
    if all(path.exists() for path in existing_manifests):
        existing_native_rows = read_jsonl(paths.native_manifest)
        existing_audio_rows = read_jsonl(paths.audio_manifest)
        existing_training_rows = read_jsonl(paths.training_audio_manifest)
        existing_holdout_rows = read_jsonl(paths.holdout_audio_manifest)
        expected_training = int(config.get("expected_counts", {}).get("selected_training", 1280))
        expected_holdout = int(config.get("expected_counts", {}).get("synthetic_holdout", 192))
        missing_paths = [
            row.get("audio_filepath")
            for row in existing_audio_rows
            if not Path(str(row.get("audio_filepath", ""))).exists()
        ]
        missing_native_paths = [
            row.get("native_audio_filepath")
            for row in existing_audio_rows
            if not Path(str(row.get("native_audio_filepath", ""))).exists()
        ]
        if (
            len(existing_native_rows) == expected_total
            and len(existing_audio_rows) == expected_total
            and len(existing_training_rows) == expected_training
            and len(existing_holdout_rows) == expected_holdout
            and not missing_paths
            and not missing_native_paths
        ):
            reporter.start("using existing complete Supertonic audio bank")
            schedule = (
                validate_exposure_schedule(config, read_jsonl(paths.exposure_schedule))
                if config.get("training_schedule", {}).get("write_legacy_schedule", True)
                else {"status": "SKIPPED", "reason": "scale-200 schedule is owned by scale200_corpus"}
            )
            summary = {
                "schema_version": "1.0",
                "status": "PASSED",
                "synthesis_mode": "native-batched-supertonic-core",
                "resumed_from_complete_manifests": True,
                "batch_size": batch_size,
                "batch_count": len(batches),
                "actual_batch_sizes": [len(batch) for batch in batches],
                "oom_fallback_count": None,
                "native_rows": len(existing_native_rows),
                "converted_rows": len(existing_audio_rows),
                "workers": workers,
                "max_pending_futures": max_pending,
                "manifests": {
                    "native_manifest_sha256": file_sha256(paths.native_manifest),
                    "audio_manifest_sha256": file_sha256(paths.audio_manifest),
                    "training_audio_manifest_sha256": file_sha256(paths.training_audio_manifest),
                    "holdout_audio_manifest_sha256": file_sha256(paths.holdout_audio_manifest),
                    "training_probe_manifest_sha256": file_sha256(paths.training_probe_manifest),
                    "exposure_schedule_sha256": optional_file_sha256(paths.exposure_schedule),
                },
                "exposure_schedule": schedule,
            }
            atomic_write_json(paths.run_root / "batched-synthesis-summary.local.json", summary)
            reporter.complete(processed_rows=len(existing_audio_rows), total_rows=len(plan), message="existing batch synthesis complete")
            return summary
        if missing_paths or missing_native_paths:
            raise RuntimeError("existing Supertonic manifests reference missing audio files")
    reporter.start("batch-synthesizing Supertonic audio")
    real_supertonic_runtime = tts_factory is None
    if tts_factory is None:
        maybe_reexec_with_supertonic_cuda_libraries(config)
        from supertonic import TTS

        provider_setup = configure_supertonic_onnx_providers(config)
        tts_factory = TTS
    else:
        provider_setup = {"available_providers": [], "requested_providers": [], "primary_provider": "fixture"}
    with heartbeat_thread(reporter, interval_seconds=10.0, message="Supertonic model load in progress"):
        tts = tts_factory(model=config["model"]["name"], model_dir=str(model_dir(config)), auto_download=False)
    provider_summary = supertonic_session_provider_summary(tts, config) if real_supertonic_runtime else provider_setup
    if int(getattr(tts, "sample_rate", 0)) != int(config["synthesis"]["native_sample_rate"]):
        raise RuntimeError(f"Supertonic native sample rate mismatch: {getattr(tts, 'sample_rate', None)}")
    styles = {style: tts.get_voice_style(style) for style in ALL_STYLES}
    monitor_path = paths.run_root / "gpu-monitor.local.csv"
    monitor_selector = str(config.get("runtime", {}).get("cuda_visible_devices", "1")).split(",")[0]
    monitor = NvidiaSmiMonitor(physical_gpu_index=monitor_selector, output_csv=monitor_path, interval_seconds=0.2)
    native_rows: list[dict[str, Any]] = []
    converted_rows: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
    conversion_wall_sum = 0.0
    validation_stat_wall_sum = 0.0
    started = time.perf_counter()
    synthesis_wall_sum = 0.0

    def collect(done: set[concurrent.futures.Future[dict[str, Any]]]) -> None:
        nonlocal conversion_wall_sum, validation_stat_wall_sum
        for future in done:
            row = future.result()
            converted_rows.append(row)
            conversion_wall_sum += float(row.get("parallel_stage", {}).get("conversion_wall_seconds", 0.0))
            validation_stat_wall_sum += float(row.get("parallel_stage", {}).get("stat_wall_seconds", 0.0))

    monitor.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for batch_index, batch in enumerate(batches, start=1):
                rows, summaries = _synthesize_batched_with_fallback(
                    config=config,
                    assets=assets,
                    provider_summary=provider_summary,
                    paths=paths,
                    tts=tts,
                    styles=styles,
                    batch=batch,
                    batch_index=batch_index,
                    experiment_seed=int(batch_config.get("seed", 1234)),
                    fallback_size=fallback_size,
                )
                native_rows.extend(rows)
                batch_summaries.extend(summaries)
                synthesis_wall_sum += sum(float(summary["wall_time_seconds"]) for summary in summaries)
                for row in rows:
                    pending.add(pool.submit(_convert_native_row, paths, row))
                while len(pending) >= max_pending:
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    collect(done)
                elapsed = time.perf_counter() - started
                reporter.progress(
                    processed_rows=len(native_rows),
                    total_rows=len(plan),
                    step=batch_index,
                    total_steps=len(batches),
                    examples_per_second=round(len(native_rows) / elapsed, 6) if elapsed else None,
                )
            if pending:
                done, pending = concurrent.futures.wait(pending)
                collect(done)
    except Exception as exc:
        reporter.failed(message="batch synthesis failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    if len(native_rows) != expected_total or len(converted_rows) != expected_total:
        raise RuntimeError(f"Supertonic output count mismatch: native={len(native_rows)} converted={len(converted_rows)}")
    native_rows.sort(key=lambda value: (str(value["partition_role"]), str(value["source_key"]), str(value["voice_style_id"])))
    converted_rows.sort(key=lambda value: (str(value["partition_role"]), str(value["source_key"]), str(value["voice_style_id"])))
    atomic_write_jsonl(paths.native_manifest, native_rows)
    atomic_write_jsonl(paths.audio_manifest, converted_rows)
    training_rows = [row for row in converted_rows if row["partition_role"] == "selected_training"]
    holdout_rows = [row for row in converted_rows if row["partition_role"] == "synthetic_holdout"]
    atomic_write_jsonl(paths.training_audio_manifest, training_rows)
    atomic_write_jsonl(paths.holdout_audio_manifest, holdout_rows)
    if config.get("training_schedule", {}).get("write_training_probe_manifest", True):
        write_training_probe_manifest(config, converted_rows)
    elif paths.training_probe_manifest.exists():
        paths.training_probe_manifest.unlink()
    schedule = (
        build_exposure_schedule(config, converted_rows)
        if config.get("training_schedule", {}).get("write_legacy_schedule", True)
        else {"status": "SKIPPED", "reason": "scale-200 schedule is owned by scale200_corpus"}
    )
    wall = time.perf_counter() - started
    monitor_summary = parse_monitor_csv(monitor_path)
    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "synthesis_mode": "native-batched-supertonic-core",
        "batch_size": batch_size,
        "batch_count": len(batches),
        "actual_batch_sizes": [len(batch) for batch in batches],
        "oom_fallback_count": sum(1 for summary in batch_summaries if summary.get("oom_fallback")),
        "native_rows": len(native_rows),
        "converted_rows": len(converted_rows),
        "workers": workers,
        "max_pending_futures": max_pending,
        "timings": {
            "gpu_synthesis_wall_seconds_sum": round(synthesis_wall_sum, 6),
            "conversion_wall_seconds_sum": round(conversion_wall_sum, 6),
            "validation_stat_wall_seconds_sum": round(validation_stat_wall_sum, 6),
            "complete_data_stage_wall_seconds": round(wall, 6),
        },
        "throughput": {
            "generated_audio_seconds_per_wall_second": round(sum(float(row["native_duration_seconds"]) for row in native_rows) / wall, 6) if wall else None,
            "items_per_second": round(len(native_rows) / wall, 6) if wall else None,
        },
        "gpu_monitor": monitor_summary,
        "batch_summaries": batch_summaries,
        "manifests": {
            "native_manifest_sha256": file_sha256(paths.native_manifest),
            "audio_manifest_sha256": file_sha256(paths.audio_manifest),
            "training_audio_manifest_sha256": file_sha256(paths.training_audio_manifest),
            "holdout_audio_manifest_sha256": file_sha256(paths.holdout_audio_manifest),
            "training_probe_manifest_sha256": optional_file_sha256(paths.training_probe_manifest),
            "exposure_schedule_sha256": optional_file_sha256(paths.exposure_schedule),
        },
        "exposure_schedule": schedule,
    }
    atomic_write_json(paths.run_root / "batched-synthesis-summary.local.json", summary)
    reporter.complete(processed_rows=len(native_rows), total_rows=len(plan), message="batch synthesis complete")
    return summary


def supertonic_provenance(config: dict[str, Any], assets: dict[str, Any], voice_style: str) -> dict[str, Any]:
    return {
        "engine": "supertonic-3",
        "package": config["package"]["name"],
        "package_version": config["package"]["version"],
        "package_license": config["package"]["license"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "model_license": config["model"]["license"],
        "asset_tree_sha256": assets["asset_tree_sha256"],
        "voice_style_id": voice_style,
        "language": config["language"]["code"],
        "total_steps": config["synthesis"]["total_steps"],
        "speed": config["synthesis"]["speed"],
        "max_chunk_length": config["synthesis"]["max_chunk_length"],
        "silence_duration": config["synthesis"]["silence_duration"],
        "execution_device": supertonic_execution_device(config),
    }


def convert_native_audio(config: dict[str, Any], *, progress_interval_seconds: float) -> dict[str, Any]:
    paths = supertonic_paths(config)
    native_rows = read_jsonl(paths.native_manifest)
    reporter = LiveProgressReporter(stage="convert", ndjson_path=paths.progress_dir / "convert.local.ndjson")
    reporter.start("converting Supertonic audio")
    rows = []
    started = time.perf_counter()
    last_emit = started
    for index, row in enumerate(native_rows, start=1):
        native_path = Path(row["native_audio_filepath"])
        relative = native_path.relative_to(paths.native_root)
        final_path = paths.final_root / relative
        convert_to_16k_pcm(native_path, final_path)
        final_stats = read_audio_stats(final_path)
        converted = dict(row)
        converted.update(
            {
                "audio_filepath": str(final_path.resolve()),
                "audio_sha256": final_stats.sha256,
                "sample_rate": final_stats.sample_rate,
                "channels": final_stats.channels,
                "sample_width": final_stats.sample_width,
                "frames": final_stats.frames,
                "duration_seconds": round(final_stats.duration_seconds, 6),
                "target_text_sha256": row["source_text_sha256"],
                "language": "sl-SI",
                "target_lang": "sl-SI",
                "source_type": "synthetic_tts",
                "audio_validation": {
                    "native_peak_ratio": round(read_audio_stats(native_path).peak_ratio, 6),
                    "final_peak_ratio": round(final_stats.peak_ratio, 6),
                    "conversion": {
                        "tool": "sox",
                        "version": sox_version(),
                        "parameters": ["-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer"],
                    },
                },
            }
        )
        rows.append(converted)
        now = time.perf_counter()
        if index % 50 == 0 or now - last_emit >= progress_interval_seconds:
            elapsed = now - started
            reporter.progress(
                processed_rows=index,
                total_rows=len(native_rows),
                examples_per_second=round(index / elapsed, 6) if elapsed else None,
            )
            last_emit = now
    rows.sort(key=lambda value: (str(value["partition_role"]), str(value["source_key"]), str(value["voice_style_id"])))
    atomic_write_jsonl(paths.audio_manifest, rows)
    training_rows = [row for row in rows if row["partition_role"] == "selected_training"]
    holdout_rows = [row for row in rows if row["partition_role"] == "synthetic_holdout"]
    atomic_write_jsonl(paths.training_audio_manifest, training_rows)
    atomic_write_jsonl(paths.holdout_audio_manifest, holdout_rows)
    write_training_probe_manifest(config, rows)
    build_exposure_schedule(config, rows)
    wall = time.perf_counter() - started
    reporter.complete(processed_rows=len(rows), total_rows=len(rows), message="conversion complete")
    return {
        "converted": len(rows),
        "audio_manifest_sha256": file_sha256(paths.audio_manifest),
        "training_audio_manifest_sha256": file_sha256(paths.training_audio_manifest),
        "holdout_audio_manifest_sha256": file_sha256(paths.holdout_audio_manifest),
        "wall_time_seconds": round(wall, 6),
    }


def stable_selected_order(rows: Sequence[Any]) -> list[str]:
    ids = [getattr(row, "selected_training_id", getattr(row, "item_id", "")) for row in rows]
    return sorted(ids, key=stable_sha256)


def write_training_probe_manifest(config: dict[str, Any], audio_rows: Sequence[dict[str, Any]]) -> None:
    paths = supertonic_paths(config)
    training_rows = [row for row in audio_rows if row["partition_role"] == "selected_training" and row["voice_style_id"] in TRAINING_STYLES]
    by_id_voice = {(str(row["source_key"]), str(row["voice_style_id"])): row for row in training_rows}
    ordered = sorted({str(row["source_key"]) for row in training_rows}, key=stable_sha256)
    probe = []
    for index, source_key in enumerate(ordered):
        voice = TRAINING_STYLES[index % len(TRAINING_STYLES)]
        row = dict(by_id_voice[(source_key, voice)])
        row["evaluation_split"] = "supertonic_training_voice_probe"
        probe.append(row)
    atomic_write_jsonl(paths.training_probe_manifest, probe)


def build_exposure_schedule(config: dict[str, Any], audio_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = supertonic_paths(config)
    training_rows = [row for row in audio_rows if row["partition_role"] == "selected_training" and row["voice_style_id"] in TRAINING_STYLES]
    by_id_voice = {(str(row["source_key"]), str(row["voice_style_id"])): row for row in training_rows}
    ordered = sorted({str(row["source_key"]) for row in training_rows}, key=stable_sha256)
    expected_rows = int(config.get("inputs", {}).get("selected_rows", 160))
    if len(ordered) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} selected-training IDs for exposure schedule")
    if expected_rows != 160:
        raise RuntimeError("legacy Supertonic exposure schedule supports only 160 selected-training IDs")
    schedule = []
    for epoch in range(1, 13):
        for position, source_key in enumerate(ordered):
            group_index = position // 20
            voice = TRAINING_STYLES[(group_index + (epoch - 1)) % len(TRAINING_STYLES)]
            row = by_id_voice[(source_key, voice)]
            schedule.append(
                {
                    "epoch": epoch,
                    "source_key": source_key,
                    "voice_style_id": voice,
                    "audio_filepath": row["audio_filepath"],
                    "audio_sha256": row["audio_sha256"],
                    "duration": row["duration_seconds"],
                    "target_text_sha256": row["target_text_sha256"],
                    "utterance_family_id": row["utterance_family_id"],
                    "source_family_id": row["source_family_id"],
                }
            )
    atomic_write_jsonl(paths.exposure_schedule, schedule)
    return validate_exposure_schedule(config, schedule)


def validate_exposure_schedule(config: dict[str, Any], schedule: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["voice_style_id"]) for row in schedule)
    by_epoch: dict[int, Counter[str]] = defaultdict(Counter)
    by_id: dict[str, set[str]] = defaultdict(set)
    for row in schedule:
        by_epoch[int(row["epoch"])][str(row["voice_style_id"])] += 1
        by_id[str(row["source_key"])].add(str(row["voice_style_id"]))
    issues = []
    if len(schedule) != 1920:
        issues.append("expected_1920_exposures")
    for style in TRAINING_STYLES:
        if counts[style] != 240:
            issues.append(f"style_{style}_count")
    for style in HELD_OUT_STYLES:
        if counts[style] != 0:
            issues.append(f"heldout_{style}_leakage")
    for epoch in range(1, 13):
        if sum(by_epoch[epoch].values()) != 160:
            issues.append(f"epoch_{epoch}_row_count")
        for style in TRAINING_STYLES:
            if by_epoch[epoch][style] != 20:
                issues.append(f"epoch_{epoch}_{style}_count")
    if any(len(styles) != 8 for styles in by_id.values()):
        issues.append("not_every_row_sees_all_voices")
    summary = {
        "status": "PASSED" if not issues else "FAILED",
        "scheduled_exposures": len(schedule),
        "exposures_by_training_style": dict(sorted(counts.items())),
        "rows_per_epoch": {str(epoch): sum(by_epoch[epoch].values()) for epoch in range(1, 13)},
        "issues": issues,
        "schedule_sha256": file_sha256(supertonic_paths(config).exposure_schedule) if supertonic_paths(config).exposure_schedule.exists() else None,
    }
    if issues:
        raise RuntimeError(f"invalid Supertonic exposure schedule: {issues}")
    return summary


def validate_supertonic_audio(config: dict[str, Any], *, progress_interval_seconds: float = 5.0) -> dict[str, Any]:
    paths = supertonic_paths(config)
    audio_rows = read_jsonl(paths.audio_manifest)
    asset_identity = verify_assets(config)
    reporter = LiveProgressReporter(stage="validate-supertonic-audio", ndjson_path=paths.progress_dir / "validate.local.ndjson")
    reporter.start("validating Supertonic audio")
    thresholds = read_json(REPO_ROOT / "configs/data_quality/synthetic_audio_v1.json")["waveform_thresholds"]
    issues: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    path_seen: set[str] = set()
    hash_seen: set[str] = set()
    rows_by_partition = Counter(str(row["partition_role"]) for row in audio_rows)
    voice_by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    text_voice_counts: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()
    last_emit = started
    for index, row in enumerate(audio_rows, start=1):
        partition = str(row["partition_role"])
        voice = str(row["voice_style_id"])
        source_key_hash = hashlib.sha256(str(row["source_key"]).encode("utf-8")).hexdigest()
        voice_by_partition[partition][voice] += 1
        text_voice_counts[str(row["source_key"])][voice] += 1
        path = str(row["audio_filepath"])
        if path in path_seen:
            issues.append({"reason": "duplicate_audio_path", "source_key_hash": source_key_hash})
        path_seen.add(path)
        stats = read_audio_stats(Path(path))
        if stats.sha256 in hash_seen:
            issues.append({"reason": "duplicate_audio_sha256", "source_key_hash": source_key_hash})
        hash_seen.add(stats.sha256)
        native = read_audio_stats(Path(row["native_audio_filepath"]))
        if native.sample_rate != 44100 or native.channels != 1 or native.sample_width != 2:
            issues.append({"reason": "native_format", "source_key_hash": source_key_hash})
        if stats.sample_rate != 16000 or stats.channels != 1 or stats.sample_width != 2:
            issues.append({"reason": "final_format", "source_key_hash": source_key_hash})
        if stats.duration_seconds < 0.2 or stats.duration_seconds > 30.0:
            issues.append({"reason": "duration_bounds", "source_key_hash": source_key_hash})
        if stats.peak_ratio < float(thresholds["minimum_peak_ratio"]):
            issues.append({"reason": "low_peak", "source_key_hash": source_key_hash})
        if stats.rms_ratio < float(thresholds["minimum_rms_ratio"]):
            issues.append({"reason": "low_rms", "source_key_hash": source_key_hash})
        if stats.active_frame_fraction < float(thresholds["minimum_active_frame_fraction"]):
            issues.append({"reason": "low_active_fraction", "source_key_hash": source_key_hash})
        if stats.clipping_fraction > float(thresholds["maximum_clipping_fraction"]):
            issues.append({"reason": "clipping", "source_key_hash": source_key_hash})
        stats_rows.append({"partition_role": partition, "voice_style_id": voice, **stats.__dict__})
        now = time.perf_counter()
        if index % 50 == 0 or now - last_emit >= progress_interval_seconds:
            reporter.progress(processed_rows=index, total_rows=len(audio_rows))
            last_emit = now
    expected_counts = config.get("expected_counts", {})
    expected_training = int(expected_counts.get("selected_training", 1280))
    expected_holdout = int(expected_counts.get("synthetic_holdout", 192))
    if rows_by_partition["selected_training"] != expected_training:
        issues.append({"reason": "training_count"})
    if rows_by_partition["synthetic_holdout"] != expected_holdout:
        issues.append({"reason": "holdout_count"})
    if voice_by_partition["selected_training"].get("M5", 0) or voice_by_partition["selected_training"].get("F5", 0):
        issues.append({"reason": "heldout_voice_training_leakage"})
    if any(voice_by_partition["synthetic_holdout"].get(style, 0) for style in TRAINING_STYLES):
        issues.append({"reason": "training_voice_holdout_leakage"})
    for source_key, counts in text_voice_counts.items():
        if any(counts[style] for style in TRAINING_STYLES) and any(counts[style] != 1 for style in TRAINING_STYLES):
            issues.append({"reason": "selected_text_missing_voice", "source_key_hash": hashlib.sha256(source_key.encode()).hexdigest()})
        if any(counts[style] for style in HELD_OUT_STYLES) and any(counts[style] != 1 for style in HELD_OUT_STYLES):
            issues.append({"reason": "holdout_text_missing_voice", "source_key_hash": hashlib.sha256(source_key.encode()).hexdigest()})
    schedule = (
        validate_exposure_schedule(config, read_jsonl(paths.exposure_schedule))
        if config.get("training_schedule", {}).get("write_legacy_schedule", True)
        else {"status": "SKIPPED", "reason": "scale-200 schedule is owned by scale200_corpus"}
    )
    durations = [float(row["duration_seconds"]) for row in stats_rows]
    by_voice: dict[str, dict[str, Any]] = {}
    for voice in ALL_STYLES:
        voice_durations = [float(row["duration_seconds"]) for row in stats_rows if row["voice_style_id"] == voice]
        by_voice[voice] = {"count": len(voice_durations), "duration_seconds": distribution(voice_durations)}
    status = "AUDIO_ACCEPTED" if not issues else "AUDIO_REJECTED"
    payload = {
        "schema_version": "1.0",
        "validator_algorithm_version": "supertonic3-synthetic-audio-validator-v1",
        "status": status,
        "row_count": len(audio_rows),
        "native_file_count": len(audio_rows),
        "final_file_count": len(audio_rows),
        "training_final_files": rows_by_partition["selected_training"],
        "holdout_final_files": rows_by_partition["synthetic_holdout"],
        "audio_manifest_sha256": file_sha256(paths.audio_manifest),
        "training_audio_manifest_sha256": file_sha256(paths.training_audio_manifest) if paths.training_audio_manifest.exists() else None,
        "holdout_audio_manifest_sha256": file_sha256(paths.holdout_audio_manifest) if paths.holdout_audio_manifest.exists() else None,
        "asset_identity": asset_identity,
        "voice_counts": {partition: dict(sorted(counts.items())) for partition, counts in voice_by_partition.items()},
        "duration_distribution": distribution(durations),
        "duration_by_voice_style": by_voice,
        "duplicate_paths": len(audio_rows) - len(path_seen),
        "duplicate_hashes": len(audio_rows) - len(hash_seen),
        "failures_by_reason": dict(sorted(Counter(issue["reason"] for issue in issues).items())),
        "issues": issues,
        "exposure_schedule": schedule,
        "limitations": [
            "All audio is synthetic.",
            "Preset labels are Supertonic voice styles, not verified speakers or demographic evidence.",
            "Waveform checks do not prove transcript correctness or natural prosody.",
            "This is AUDIO_ACCEPTED only and never TRAINING_ELIGIBLE.",
        ],
    }
    atomic_write_json(paths.validation, payload)
    reporter.complete(processed_rows=len(audio_rows), total_rows=len(audio_rows), message="validation complete")
    return payload


def cpu_identity() -> dict[str, Any]:
    model_name = ""
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.lower().startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        model_name = platform.processor()
    return {
        "model": model_name or platform.processor(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def summarize_supertonic_audio(config: dict[str, Any]) -> dict[str, Any]:
    paths = supertonic_paths(config)
    validation = read_json(paths.validation)
    native_rows = read_jsonl(paths.native_manifest)
    audio_rows = read_jsonl(paths.audio_manifest)
    synthesis_wall = sum(float(row.get("runtime", {}).get("synthesis_wall_time_seconds", 0.0)) for row in native_rows)
    total_native_duration = sum(float(row["native_duration_seconds"]) for row in native_rows)
    versions = runtime_versions(config)
    public_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-supertonic3-multivoice-audio-v1",
        "status": validation["status"],
        "decision_date": "2026-06-25",
        "tts": {
            "package": config["package"]["name"],
            "package_version": config["package"]["version"],
            "package_license": config["package"]["license"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "model_license": config["model"]["license"],
            "asset_tree_sha256": validation["asset_identity"]["asset_tree_sha256"],
        },
        "voice_styles": {
            "available": list(ALL_STYLES),
            "training": list(TRAINING_STYLES),
            "held_out": list(HELD_OUT_STYLES),
            "voice_style_hashes": validation["asset_identity"]["voice_style_hashes"],
            "counts": validation["voice_counts"],
            "duration_by_style": validation["duration_by_voice_style"],
            "labels_are_presets_only": True,
        },
        "counts": {
            "selected_texts": 160,
            "holdout_texts": 96,
            "native_training_files": validation["training_final_files"],
            "final_training_files": validation["training_final_files"],
            "native_holdout_files": validation["holdout_final_files"],
            "final_holdout_files": validation["holdout_final_files"],
            "audio_manifest_rows": validation["row_count"],
        },
        "hashes": {
            "audio_manifest_sha256": validation["audio_manifest_sha256"],
            "training_audio_manifest_sha256": validation["training_audio_manifest_sha256"],
            "holdout_audio_manifest_sha256": validation["holdout_audio_manifest_sha256"],
            "native_manifest_sha256": file_sha256(paths.native_manifest),
            "training_probe_manifest_sha256": file_sha256(paths.training_probe_manifest),
            "exposure_schedule_sha256": optional_file_sha256(paths.exposure_schedule),
            "tts_config_sha256": file_sha256(SUPERTONIC_CONFIG_PATH),
        },
        "audio_format": {
            "native_sample_rate": 44100,
            "final_sample_rate": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "encoding": "signed-16-bit-pcm-wav",
        },
        "validation": {
            "duplicate_paths": validation["duplicate_paths"],
            "duplicate_hashes": validation["duplicate_hashes"],
            "failures_by_reason": validation["failures_by_reason"],
            "duration_distribution": validation["duration_distribution"],
        },
        "synthesis": {
            "execution_device": supertonic_execution_device(config),
            "required_provider": config.get("runtime", {}).get("required_provider"),
            "human_override": config.get("runtime", {}).get("human_override"),
            "runtime_packages": versions.get("packages", {}),
            "wheel_record_hashes": package_wheel_hashes(config),
            "cpu": cpu_identity(),
            "thread_settings": {
                "SUPERTONIC_INTRA_OP_THREADS": os.environ.get("SUPERTONIC_INTRA_OP_THREADS"),
                "SUPERTONIC_INTER_OP_THREADS": os.environ.get("SUPERTONIC_INTER_OP_THREADS"),
            },
            "aggregate_synthesis_wall_seconds": round(synthesis_wall, 6),
            "generated_audio_seconds_per_wall_second": round(total_native_duration / synthesis_wall, 6) if synthesis_wall else None,
            "rows_per_minute": round(len(native_rows) / synthesis_wall * 60.0, 6) if synthesis_wall else None,
            "peak_process_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "limitations": validation["limitations"],
    }
    assert_public_payload_safe(public_certificate)
    public_report = {
        "schema_version": "1.0",
        "report_id": "0010-supertonic3-multivoice-acoustic-admission",
        "certificate": public_certificate,
    }
    assert_public_payload_safe(public_report)
    atomic_write_json(SUPER_AUDIO_CERTIFICATE_PATH, public_certificate)
    atomic_write_json(SUPER_AUDIO_REPORT_JSON, public_report)
    write_supertonic_audio_markdown(SUPER_AUDIO_REPORT_MD, public_report)
    local_summary = {
        "certificate_sha256": file_sha256(SUPER_AUDIO_CERTIFICATE_PATH),
        "report_json_sha256": file_sha256(SUPER_AUDIO_REPORT_JSON),
        "report_markdown_sha256": file_sha256(SUPER_AUDIO_REPORT_MD),
        "status": public_certificate["status"],
        "audio_manifest_sha256": validation["audio_manifest_sha256"],
    }
    atomic_write_json(paths.summary, local_summary)
    return local_summary


def write_supertonic_audio_markdown(path: Path, payload: dict[str, Any]) -> None:
    cert = payload["certificate"]
    lines = [
        "# Supertonic 3 Multi-voice Acoustic Admission",
        "",
        f"Status: `{cert['status']}`",
        "",
        "This privacy-safe report contains aggregate synthetic-audio evidence only. It contains no generated text, IDs, audio paths, local paths, hypotheses, or references.",
        "",
        "## Identity",
        "",
        f"- Package: `{cert['tts']['package']}=={cert['tts']['package_version']}`",
        f"- Model: `{cert['tts']['model_repository']}@{cert['tts']['model_revision']}`",
        f"- Asset-tree SHA256: `{cert['tts']['asset_tree_sha256']}`",
        "",
        "## Counts",
        "",
        f"- Selected texts: {cert['counts']['selected_texts']}",
        f"- Holdout texts: {cert['counts']['holdout_texts']}",
        f"- Final training WAVs: {cert['counts']['final_training_files']}",
        f"- Final held-out WAVs: {cert['counts']['final_holdout_files']}",
        f"- Audio manifest SHA256: `{cert['hashes']['audio_manifest_sha256']}`",
        f"- Training audio manifest SHA256: `{cert['hashes']['training_audio_manifest_sha256']}`",
        f"- Held-out audio manifest SHA256: `{cert['hashes']['holdout_audio_manifest_sha256']}`",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in cert["limitations"]],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_supertonic_training_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(supertonic_paths(load_supertonic_config(config["tts_config"])).exposure_schedule)


def training_records_for_supertonic_epoch(clean_records: Sequence[Any], schedule: Sequence[dict[str, Any]], epoch: int) -> dict[str, Any]:
    from slaif_asr.corpus_v2_training import TrainingRecord

    clean_by_id = {row.selected_training_id: row for row in clean_records}
    output: dict[str, TrainingRecord] = {}
    for row in schedule:
        if int(row["epoch"]) != epoch:
            continue
        selected_id = str(row["source_key"])
        clean = clean_by_id[selected_id]
        output[selected_id] = TrainingRecord(
            selected_training_id=clean.selected_training_id,
            audio_filepath=str(row["audio_filepath"]),
            duration=float(row["duration"]),
            text=clean.text,
            text_sha256=clean.text_sha256,
            audio_sha256=str(row["audio_sha256"]),
            selection_reason=clean.selection_reason,
            selection_rank=clean.selection_rank,
        )
    if set(output) != set(clean_by_id):
        raise RuntimeError("Supertonic schedule does not cover every selected-training row exactly once")
    return output
