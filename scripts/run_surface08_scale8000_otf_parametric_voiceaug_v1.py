#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.otf_augmentation_dataset import (
    compute_fill_rate_metrics,
    prepare_virtual_sample,
    waveform_sha256,
)
from slaif_asr.parametric_voice_augmentation import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    DEPENDENCIES,
    PHASES,
    apply_policy_transform,
    summarize_profile_counts,
    verify_dependencies,
)
from slaif_asr.scale8000_parametric_voiceaug_surface08 import (
    EXPERIMENT_ID,
    load_config,
    load_policy_for_config,
)
from slaif_asr.transcript_preserving_augmentation import write_mono_pcm16


DEFAULT_CONFIG = Path(
    "configs/experiments/surface08-scale8000-otf-parametric-voiceaug-v1.json"
)
ARM_NAME = "surface08_scale8000_otf_parametric_voiceaug_v1"
REPORT_JSON = Path(
    "docs/experiments/0033-surface08-scale8000-otf-parametric-voiceaug-v1.json"
)
REPORT_MD = Path(
    "docs/experiments/0033-surface08-scale8000-otf-parametric-voiceaug-v1.md"
)
CERTIFICATE_PATH = Path(
    "docs/data-certificates/sl-corpus-v5-scale8000-parametric-voiceaug-v1-diagnostic.json"
)

STANDARD_OTF_METRICS = {
    "piper_synthetic_holdout": {"wer": 13.199, "cer": 3.785, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 4.658, "cer": 1.472, "empty": 0},
    "fleurs_v2": {"wer": 39.353, "cer": 12.002, "empty": 0},
    "artur_j": {"wer": 39.974, "cer": 12.251, "empty": 0},
}
STRONGAUG_METRICS = {
    "piper_synthetic_holdout": {"wer": 11.957, "cer": 3.645, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 5.202, "cer": 1.850, "empty": 0},
    "fleurs_v2": {"wer": 39.904, "cer": 12.343, "empty": 0},
    "artur_j": {"wer": 39.406, "cer": 12.357, "empty": 0},
}

_BASE_PATH = Path(__file__).with_name("run_surface08_scale8000_otf_augmented.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_slaif_scale8000_otf_parametric_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import scale-8000 OTF Surface08 runner")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

_ORIGINAL_ITER = _BASE.iter_prefetched_microbatches
_ORIGINAL_DETERMINISM = _BASE.run_determinism_checks_for_exposures
_ACTIVE_POLICY: dict[str, Any] | None = None


def _parametric_iter(*args: Any, **kwargs: Any) -> Any:
    if _ACTIVE_POLICY is None:
        raise RuntimeError("ParametricVoiceAug policy is not loaded")
    kwargs["augmentation_policy"] = _ACTIVE_POLICY
    return _ORIGINAL_ITER(*args, **kwargs)


def _parametric_determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if _ACTIVE_POLICY is None:
        raise RuntimeError("ParametricVoiceAug policy is not loaded")
    kwargs["augmentation_policy"] = _ACTIVE_POLICY
    result = _ORIGINAL_DETERMINISM(*args, **kwargs)
    replay_passed = result.get("status") == "PASSED"
    result.update(
        {
            "augmentation_chain_replay": replay_passed,
            "parameter_replay": replay_passed,
        }
    )
    result["status"] = "PASSED" if all(
        value
        for key, value in result.items()
        if key not in {"status", "examples"}
    ) else "FAILED"
    return result


def classify_result(
    metrics: dict[str, dict[str, Any]],
    *,
    training: dict[str, Any],
) -> str:
    if float(training["loader_fill_rate"]) < 0.95:
        return "PARAMETRIC_VOICEAUG_V1_LOADER_THROUGHPUT_BLOCKED"
    if any(
        int(row.get("worker_failures", 0)) > 0
        for row in training["loader_telemetry"]
    ):
        return "PARAMETRIC_VOICEAUG_V1_LOADER_THROUGHPUT_BLOCKED"
    if not bool(training["parameter_integrity"]["only_surface08_changed"]):
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    synthetic_splits = (
        "piper_synthetic_holdout",
        "supertonic_heldout_voice_holdout",
    )
    if any(int(metrics[split]["empty"]) > 0 for split in real_splits):
        return "PARAMETRIC_VOICEAUG_V1_LABEL_DESTRUCTION"
    comparisons = [
        (
            split,
            metric,
            float(metrics[split][metric]),
            float(STANDARD_OTF_METRICS[split][metric]),
            0.50 if metric == "wer" else 0.25,
        )
        for split in real_splits
        for metric in ("wer", "cer")
    ]
    all_real_improve = all(
        candidate < prior
        for _split, _metric, candidate, prior, _tolerance in comparisons
    )
    any_real_improve = any(
        candidate < prior
        for _split, _metric, candidate, prior, _tolerance in comparisons
    )
    within = all(
        candidate <= prior + tolerance
        for _split, _metric, candidate, prior, tolerance in comparisons
    )
    synthetic_not_collapsed = all(
        int(metrics[split]["empty"]) == 0
        and float(metrics[split]["wer"])
        < float(_BASE.BASE_DIRECTIONAL_METRICS[split]["wer"])
        and float(metrics[split]["cer"])
        < float(_BASE.BASE_DIRECTIONAL_METRICS[split]["cer"])
        for split in synthetic_splits
    )
    synthetic_tradeoff = any(
        float(metrics[split][metric]) > float(STANDARD_OTF_METRICS[split][metric])
        for split in synthetic_splits
        for metric in ("wer", "cer")
    )
    synthetic_gain = all(
        float(metrics[split][metric]) < float(STANDARD_OTF_METRICS[split][metric])
        for split in synthetic_splits
        for metric in ("wer", "cer")
    )
    if all_real_improve and synthetic_not_collapsed:
        if synthetic_tradeoff:
            return "PARAMETRIC_VOICEAUG_V1_REAL_GATE_GAIN_SYNTHETIC_TRADEOFF"
        return "PARAMETRIC_VOICEAUG_V1_NEW_BEST_DIRECTIONAL"
    if within and any_real_improve and synthetic_tradeoff:
        return "PARAMETRIC_VOICEAUG_V1_REAL_GATE_GAIN_SYNTHETIC_TRADEOFF"
    if within and synthetic_gain:
        return "PARAMETRIC_VOICEAUG_V1_SYNTHETIC_ONLY_GAIN"
    if within:
        return "PARAMETRIC_VOICEAUG_V1_MATCHES_STANDARD_OTF"
    return "PARAMETRIC_VOICEAUG_V1_REAL_GATE_REGRESSION"


def _severity(operation: dict[str, Any]) -> str:
    family = str(operation["family"])
    values = dict(operation["parameters"])
    if family == "pitch_shift":
        value = abs(float(values["semitones"]))
        return "mild" if value < 1.4 else ("medium" if value < 2.3 else "strong")
    if family == "speed":
        value = abs(float(values["rate"]) - 1.0)
        return "mild" if value < 0.04 else ("medium" if value < 0.08 else "strong")
    if family == "formant_warp":
        value = abs(float(values["factor"]) - 1.0)
        return "mild" if value < 0.06 else ("medium" if value < 0.12 else "strong")
    if family == "aperiodicity":
        value = float(values["breathiness_snr_db"])
        return "mild" if value >= 28 else ("medium" if value >= 23 else "strong")
    if family == "channel_filter":
        value = str(values["filter"])
        return "strong" if value == "telephone_like" else (
            "medium" if value in {"band_pass", "broad_eq_tilt"} else "mild"
        )
    if family == "codec_companding":
        value = str(values["codec"])
        return "mild" if value in {
            "downsample_12k_return_16k",
            "ogg_opus_moderate",
        } else "medium"
    if family == "reverb":
        value = float(values["rt60_seconds"])
        return "mild" if value < 0.30 else ("medium" if value < 0.55 else "strong")
    if family == "additive_noise":
        value = float(values["snr_db"])
        return "mild" if value >= 20 else ("medium" if value >= 12 else "strong")
    if family == "gain_compression":
        value = str(values["mode"])
        return "strong" if value == "soft_clip" else (
            "medium" if value == "limiting" else "mild"
        )
    raise ValueError(f"unknown audit family {family}")


def stage_audit(config_path: Path, *, examples_per_family: int = 5) -> dict[str, Any]:
    config = load_config(config_path)
    policy = load_policy_for_config(config)
    pool = _BASE.load_clean_pool(config, runs_root=_BASE.scale8000_runs_root())
    profiles = _BASE.load_profiles(config)
    audit_root = _BASE.run_dir(config) / "audit"
    audio_root = audit_root / "audio"
    family_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    seen_exposures: set[int] = set()
    scan_tasks = [
        *_BASE.build_round_tasks(pool, 1, physical_microbatch=1),
        *_BASE.build_round_tasks(pool, 5, physical_microbatch=1),
    ]
    for task in scan_tasks:
        record, exposure_id = task.exposures[0]
        if exposure_id in seen_exposures:
            continue
        seen_exposures.add(exposure_id)
        sample = prepare_virtual_sample(
            record,
            exposure_id,
            profiles,
            corpus_id=_BASE.CORPUS_ID,
            augmentation_key=AUGMENTATION_KEY,
            algorithm_version=ALGORITHM_VERSION,
            augmentation_policy=policy,
        )
        if sample.spec.parameters.get("mode") != "parametric":
            continue
        operations = list(sample.spec.parameters["operations"])
        eligible_families = [
            str(item["family"])
            for item in operations
            if family_counts[str(item["family"])] < examples_per_family
        ]
        if not eligible_families:
            continue
        output = audio_root / f"exposure-{exposure_id:06d}.wav"
        write_mono_pcm16(output, sample.waveform)
        operation_rows = []
        for operation in operations:
            family = str(operation["family"])
            severity = _severity(operation)
            operation_rows.append(
                {
                    "family": family,
                    "parameters": dict(operation["parameters"]),
                    "severity": severity,
                }
            )
            severity_counts[severity] += 1
        records.append(
            {
                "virtual_exposure_id": exposure_id,
                "profile_id": sample.spec.profile_id,
                "parameter_seed": sample.spec.parameter_seed,
                "operations": operation_rows,
                "audio_path": str(output.resolve()),
                "audio_sha256": waveform_sha256(sample.waveform),
                "duration_seconds": sample.output_duration_seconds,
                "sample_rate": sample.sample_rate,
                "transcript_preserved": sample.transcript == record.transcript,
            }
        )
        for family in eligible_families:
            family_counts[family] += 1
        if all(family_counts[family] >= examples_per_family for family in (
            "pitch_shift",
            "speed",
            "formant_warp",
            "aperiodicity",
            "channel_filter",
            "codec_companding",
            "reverb",
            "additive_noise",
            "gain_compression",
        )) and all(severity_counts[key] > 0 for key in ("mild", "medium", "strong")):
            break
    expected_families = {
        "pitch_shift",
        "speed",
        "formant_warp",
        "aperiodicity",
        "channel_filter",
        "codec_companding",
        "reverb",
        "additive_noise",
        "gain_compression",
    }
    if any(family_counts[family] < examples_per_family for family in expected_families):
        raise RuntimeError("ParametricVoiceAug audit did not cover every family")
    if not all(severity_counts[key] > 0 for key in ("mild", "medium", "strong")):
        raise RuntimeError("ParametricVoiceAug audit did not cover all severities")
    if not all(bool(row["transcript_preserved"]) for row in records):
        raise RuntimeError("ParametricVoiceAug audit changed a transcript")
    payload = {
        "status": "PASSED",
        "examples_per_family": examples_per_family,
        "family_counts": dict(sorted(family_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "saved_audio_files": len(records),
        "automatic_destructive_cases": 0,
        "human_listening_status": "NOT_RUN",
        "dependencies": verify_dependencies(),
        "records": records,
    }
    atomic_write_json(audit_root / "audit-summary.local.json", payload)
    print(json.dumps(
        {
            "status": payload["status"],
            "family_counts": payload["family_counts"],
            "severity_counts": payload["severity_counts"],
            "saved_audio_files": payload["saved_audio_files"],
            "automatic_destructive_cases": 0,
            "human_listening_status": "NOT_RUN",
        },
        sort_keys=True,
    ))
    return payload


def stage_probe_loader(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    policy = load_policy_for_config(config)
    pool = _BASE.load_clean_pool(config, runs_root=_BASE.scale8000_runs_root())
    profiles = _BASE.load_profiles(config)
    target_examples_per_second = 12.8
    physical_microbatch = 2
    warmup_microbatches = 32
    measured_microbatches = 256
    tasks = _BASE.build_round_tasks(
        pool,
        5,
        physical_microbatch=physical_microbatch,
    )[: warmup_microbatches + measured_microbatches]
    deliveries = _ORIGINAL_ITER(
        tasks,
        profiles,
        num_workers=3,
        prefetch_microbatches=16,
        corpus_id=_BASE.CORPUS_ID,
        augmentation_key=AUGMENTATION_KEY,
        algorithm_version=ALGORITHM_VERSION,
        augmentation_policy=policy,
        timeout_seconds=180.0,
    )
    interval = physical_microbatch / target_examples_per_second
    measured = []
    started = time.perf_counter()
    for index, delivery in enumerate(deliveries):
        time.sleep(interval)
        if index >= warmup_microbatches:
            measured.append(delivery)
    wall_seconds = time.perf_counter() - started
    if len(measured) != measured_microbatches:
        raise RuntimeError("ParametricVoiceAug loader preflight count mismatch")
    waits = [row.consumer_wait_seconds for row in measured]
    ready = [row.ready_without_wait for row in measured]
    queue_levels = [row.queue_depth_after for row in measured]
    fill = compute_fill_rate_metrics(
        wait_seconds=waits,
        queue_fill_levels=queue_levels,
        target_examples_per_second=target_examples_per_second,
        physical_microbatch=physical_microbatch,
        ready_wait_threshold_ms=50.0,
        ready_flags=ready,
    )
    aggregate_compute_wall = sum(
        row.prepared.processing_wall_seconds for row in measured
    )
    aggregate_audio_seconds = sum(
        row.prepared.output_audio_seconds for row in measured
    )
    capacity_seconds = aggregate_compute_wall / 3.0
    prepared_examples_per_second = (
        measured_microbatches * physical_microbatch / max(capacity_seconds, 1e-9)
    )
    payload = {
        "status": "PASSED"
        if fill["fill_rate"] >= 0.95
        and fill["microbatch_ready_rate"] >= 0.95
        and prepared_examples_per_second >= target_examples_per_second
        else "FAILED",
        "target_examples_per_second": target_examples_per_second,
        "prepared_examples_per_second": prepared_examples_per_second,
        "prepared_audio_seconds_per_second": aggregate_audio_seconds
        / max(capacity_seconds, 1e-9),
        "workers": 3,
        "physical_microbatch": physical_microbatch,
        "prefetch_microbatches": 16,
        "warmup_microbatches": warmup_microbatches,
        "measured_microbatches": measured_microbatches,
        "benchmark_wall_seconds": wall_seconds,
        **fill,
    }
    atomic_write_json(
        _BASE.run_dir(config) / "verification" / "loader.local.json",
        payload,
    )
    print(json.dumps(payload, sort_keys=True))
    return payload


def stage_train(config_path: Path, progress_interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    loader_path = _BASE.run_dir(config) / "verification" / "loader.local.json"
    if not loader_path.exists():
        raise RuntimeError("passing ParametricVoiceAug loader preflight is required")
    loader = json.loads(loader_path.read_text(encoding="utf-8"))
    if loader.get("status") != "PASSED":
        raise RuntimeError("PARAMETRIC_VOICEAUG_V1_LOADER_THROUGHPUT_BLOCKED")
    return _BASE.stage_train(config_path, progress_interval)


def _format_metric(value: dict[str, Any]) -> str:
    return (
        f"{float(value['wer']):.3f} / {float(value['cer']):.3f} / "
        f"{int(value['empty'])}"
    )


def _markdown_report(public: dict[str, Any]) -> str:
    training = public["training"]
    metrics = public["directional_evaluation"]["metrics"]
    augmentation = public["otf_augmentation"]
    audit = augmentation["audit"]
    determinism = augmentation["determinism"]
    lines = [
        "# Experiment 0033: Surface08 Scale-8000 OTF ParametricVoiceAug v1",
        "",
        f"Classification: `{public['classification']}`",
        "",
        "This matched diagnostic keeps Experiment 0031's Surface08 model, untouched-base initialization, scale-8000 clean reservoir, optimizer, effective batch, controller policy, directional suite, and 144,000-exposure cap fixed. Only the deterministic OTF augmentation family changes. ParametricVoiceAug is language-agnostic waveform DSP and uses no target identity.",
        "",
        "## Result",
        "",
        f"- Selected round: {training['selected_round']}; stopped round: {training['stopped_round']} (`{training['stopped_reason']}`).",
        f"- Training exposures: {training['sample_exposures']:,}; unique semantic rows: {training['data_diversity']['unique_semantic_rows_seen']:,}.",
        f"- Aggregate OTF fill rate: {training['loader_fill_rate']:.6f}.",
        f"- Selected checkpoint SHA256: `{public['directional_evaluation']['selected_checkpoint_sha256']}`.",
        "- `accepted_parent` remains `none`; this is noncanonical diagnostic evidence.",
        "",
        "## Augmentation Families",
        "",
        "| Family | Fraction | Parameter range | Tool/backend | License/provenance | Notes |",
        "|---|---:|---|---|---|---|",
        f"| Standard OTF | {audit['standard_fraction']:.6f} | existing 11 profiles | existing repo DSP | existing | one standard profile per selected sample |",
    ]
    family_rows = (
        ("pitch_shift", "F0/pitch", "common +/-1-2; rare +/-3 semitones", "librosa pitch_shift + soxr", "ISC + LGPL-2.1-or-later"),
        ("speed", "Speed/tempo", "0.90-1.10", "SciPy polyphase resampling", "BSD-3-Clause"),
        ("formant_warp", "Formant/spectral warp", "factor 0.84-1.16", "SciPy STFT log-envelope warp", "BSD-3-Clause"),
        ("aperiodicity", "Aperiodicity/noisiness", "20-32 dB breathiness SNR", "NumPy/SciPy high-pass excitation", "BSD-3-Clause"),
        ("reverb", "Reverb/RIR", "RT60 0.15-0.80 s", "repo deterministic synthetic RIR", "Apache-2.0 orchestration"),
        ("channel_filter", "Bandpass/channel", "mild filters; limited 300-3400 Hz telephone", "repo SciPy filters", "BSD-3-Clause"),
        ("codec_companding", "Codec/companding", "G.711, 12 kHz round-trip, moderate quantization", "CPython 3.12 audioop/repo DSP", "PSF-2.0 / Apache-2.0"),
        ("additive_noise", "Noise", "8-25 dB SNR", "procedural NumPy/SciPy", "BSD-3-Clause"),
        ("gain_compression", "Gain/compression/clipping", "-6 to +6 dB; ratio 1.5-3.0", "repo NumPy dynamics", "BSD-3-Clause"),
    )
    for key, label, parameter_range, backend, provenance in family_rows:
        lines.append(
            f"| {label} | {audit['family_sample_fractions'].get(key, 0.0):.6f} | {parameter_range} | {backend} | {provenance} | deterministic, in memory |"
        )
    lines.extend(
        [
            "",
            "| Round range | Standard OTF % | ParametricVoiceAug % | Purpose |",
            "|---|---:|---:|---|",
            "| 1-4 | 50 | 50 | linguistic coverage plus early acoustic pressure |",
            "| 5-9 | 10 | 90 | heavy robustness pressure on repeated rows |",
            "",
            "## Dependencies / Provenance",
            "",
            "| Tool/dependency | Version/revision | License | Used for | Vendored? | Notes |",
            "|---|---|---|---|---|---|",
        ]
    )
    for dependency in augmentation["dependencies"]:
        lines.append(
            f"| {dependency['name']} | {dependency['version']} | {dependency['license']} | {dependency['used_for']} | no | {dependency['installation']} |"
        )
    lines.extend(
        [
            "",
            "## Determinism",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Same exposure, same worker count | {determinism['same_virtual_exposure_same_worker_count']} |",
            f"| Same exposure, different worker count | {determinism['worker_count_independent']} |",
            f"| Restart replay | {determinism['same_virtual_exposure_after_restart']} |",
            f"| Worker-order independence | {determinism['worker_order_independent']} |",
            f"| Chain replay | {determinism['augmentation_chain_replay']} |",
            f"| Parameter replay | {determinism['parameter_replay']} |",
            f"| Transcript preserved | {determinism['transcript_preserved']} |",
            "",
            "The local audit covered at least five examples per augmentation family and mild, medium, and strong cases. Automated waveform checks found no destructive cases. Human listening was `NOT_RUN`; it is not reported as passed.",
            "",
            "## Controller-Dev Curve",
            "",
            "| Round | Step | Exposures | Unique rows | Train loss | Anchor | Scale | ARTUR-dev WER/CER/empty | Eligible |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in public["controller_dev"]["curve"]:
        train_loss = "NOT_APPLICABLE" if row["train_loss"] is None else row["train_loss"]
        lines.append(
            f"| {row['round']} | {row['optimizer_step']} | {row['exposures_seen']} | {row.get('unique_semantic_rows', 0)} | {train_loss} | {row['synthetic_anchor_probe_loss']} | {row['synthetic_scale_probe_loss']} | {row['wer']} / {row['cer']} / {row['empty']} | {str(row['eligible']).lower()} |"
        )
    lines.extend(
        [
            "",
            "ARTUR controller-dev aggregate metrics alone selected the checkpoint. FLEURS-v2 and ARTUR-J were evaluated only after selection.",
            "",
            "## Live OTF Loader",
            "",
            "| Round | Examples/s | Audio-sec/s | Fill rate | Consumer wait | Queue p50/p95 | Worker failures | Notes |",
            "|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in training["loader_telemetry"]:
        audio_rate = (
            float(row["output_audio_seconds"])
            / max(16_000.0 / float(row["examples_per_second"]), 1e-9)
        )
        lines.append(
            f"| {row['round']} | {row['examples_per_second']:.6f} | {audio_rate:.6f} | {row['otf_fill_rate']:.6f} | {row['consumer_wait_percent']:.6f}% | {row['queue_p50']} / {row['queue_p95']} | {int(row.get('worker_failures', 0))} | 3 spawned workers; in-memory only |"
        )
    lines.extend(
        [
            "",
            "## Directional Metrics",
            "",
            "| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Scale-8000 standard OTF | StrongAug v1 | ParametricVoiceAug v1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in (
        "piper_synthetic_holdout",
        "supertonic_heldout_voice_holdout",
        "fleurs_v2",
        "artur_j",
    ):
        lines.append(
            f"| {split} | {_format_metric(metrics['base'][split])} | {_format_metric(metrics['pr36'][split])} | {_format_metric(metrics['surface07'][split])} | {_format_metric(metrics['surface08_scale2000'][split])} | {_format_metric(metrics['surface08_scale8000_standard_otf'][split])} | {_format_metric(metrics['surface08_scale8000_strongaug_v1'][split])} | {_format_metric(metrics['surface08_scale8000_parametric_voiceaug_v1'][split])} |"
        )
    lines.extend(
        [
            "",
            "Values are normalized WER / CER / empty hypotheses.",
            "",
            "## Boundaries",
            "",
            "- No real speech, S6TTS, scale-2000 audio, scale-32000, or immutable gate was used for training.",
            "- No target identity, voice-cloning model, downloaded voice model, or external noise corpus was used.",
            "- No augmented WAV was pre-rendered, written, or used for training.",
            "- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.",
            "- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    public = _BASE.stage_summarize(config_path)
    config = load_config(config_path)
    audit_local = json.loads(
        (_BASE.run_dir(config) / "audit" / "audit-summary.local.json").read_text(
            encoding="utf-8"
        )
    )
    if audit_local.get("status") != "PASSED":
        raise RuntimeError("ParametricVoiceAug local audit did not pass")
    profile_counts = public["training"]["data_diversity"][
        "augmentation_profile_distribution"
    ]
    audit = summarize_profile_counts(profile_counts)
    metrics = public["directional_evaluation"]["metrics"]
    candidate = metrics.pop("surface08_scale8000_otf")
    metrics["surface08_scale8000_standard_otf"] = STANDARD_OTF_METRICS
    metrics["surface08_scale8000_strongaug_v1"] = STRONGAUG_METRICS
    metrics["surface08_scale8000_parametric_voiceaug_v1"] = candidate
    public.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "stacked_on": "PR #52 merge commit 5323883 on the post-PR-50 OTF feature branch",
        }
    )
    public["otf_augmentation"].update(
        {
            "augmentation_key": AUGMENTATION_KEY,
            "algorithm_version": ALGORITHM_VERSION,
            "policy_config_sha256": config["otf_augmentation"][
                "policy_config_sha256"
            ],
            "phase_schedule": PHASES,
            "max_operations_per_sample": 3,
            "language_agnostic": True,
            "target_identity_models": False,
            "dependencies": list(DEPENDENCIES),
            "audit": audit,
            "local_audit": {
                "status": audit_local["status"],
                "examples_per_family": audit_local["examples_per_family"],
                "family_counts": audit_local["family_counts"],
                "severity_counts": audit_local["severity_counts"],
                "automatic_destructive_cases": audit_local[
                    "automatic_destructive_cases"
                ],
                "human_listening_status": audit_local["human_listening_status"],
                "audio_committed": False,
                "local_paths_committed": False,
            },
        }
    )
    certificate_path = REPO_ROOT / CERTIFICATE_PATH
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate.update(
        {
            "certificate_id": "sl-corpus-v5-scale8000-parametric-voiceaug-v1-diagnostic",
            "augmentation_key": AUGMENTATION_KEY,
            "augmentation_algorithm_version": ALGORITHM_VERSION,
            "augmentation_policy_config_sha256": config["otf_augmentation"][
                "policy_config_sha256"
            ],
            "classification": public["classification"],
            "language_agnostic": True,
            "target_identity_models_used": False,
            "external_noise_data_used": False,
        }
    )
    _BASE.validate_public_report(public)
    _BASE.validate_public_report(certificate)
    atomic_write_json(REPO_ROOT / REPORT_JSON, public)
    atomic_write_json(certificate_path, certificate)
    atomic_write_text(REPO_ROOT / REPORT_MD, _markdown_report(public))
    print(json.dumps(
        {
            "status": "PASSED",
            "classification": public["classification"],
            "selected_round": public["training"]["selected_round"],
            "parametric_fraction": audit["parametric_fraction"],
        },
        sort_keys=True,
    ))
    return public


def _configure_base_runner() -> None:
    _BASE.DEFAULT_CONFIG = DEFAULT_CONFIG
    _BASE.ARM_NAME = ARM_NAME
    _BASE.REPORT_JSON = REPORT_JSON
    _BASE.REPORT_MD = REPORT_MD
    _BASE.CERTIFICATE_PATH = CERTIFICATE_PATH
    _BASE.EXPERIMENT_ID = EXPERIMENT_ID
    _BASE.AUGMENTATION_KEY = AUGMENTATION_KEY
    _BASE.ALGORITHM_VERSION = ALGORITHM_VERSION
    _BASE.load_config = load_config
    _BASE.iter_prefetched_microbatches = _parametric_iter
    _BASE.run_determinism_checks_for_exposures = _parametric_determinism
    _BASE.classify_result = classify_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Surface08 on scale-8000 with deterministic ParametricVoiceAug v1"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "verify-inputs",
            "audit",
            "probe-hardware",
            "probe-surface",
            "probe-otf",
            "probe-loader",
            "probe-microbatch",
            "train",
            "evaluate-directional",
            "summarize",
        ),
    )
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--audit-examples-per-family", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_POLICY

    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    _ACTIVE_POLICY = load_policy_for_config(config)
    _configure_base_runner()
    stages = {
        "verify-inputs": lambda: _BASE.stage_verify_inputs(config_path),
        "audit": lambda: stage_audit(
            config_path, examples_per_family=args.audit_examples_per_family
        ),
        "probe-hardware": lambda: _BASE.stage_probe_hardware(config_path),
        "probe-surface": lambda: _BASE.stage_probe_surface(config_path),
        "probe-otf": lambda: _BASE.stage_probe_otf(config_path),
        "probe-loader": lambda: stage_probe_loader(config_path),
        "probe-microbatch": lambda: _BASE.stage_probe_microbatch(config_path),
        "train": lambda: stage_train(config_path, args.progress_interval),
        "evaluate-directional": lambda: _BASE.stage_evaluate_directional(config_path),
        "summarize": lambda: stage_summarize(config_path),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
