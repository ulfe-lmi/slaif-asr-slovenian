#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.scale8000_strongaug_surface08 import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    EXPERIMENT_ID,
    load_config,
    load_policy_for_config,
)
from slaif_asr.strongaug_v1 import PARAMETER_RANGES, PHASES, summarize_profile_counts


DEFAULT_CONFIG = Path("configs/experiments/surface08-scale8000-otf-strongaug-v1.json")
ARM_NAME = "surface08_scale8000_otf_strongaug_v1"
REPORT_JSON = Path("docs/experiments/0032-surface08-scale8000-otf-strongaug-v1.json")
REPORT_MD = Path("docs/experiments/0032-surface08-scale8000-otf-strongaug-v1.md")
CERTIFICATE_PATH = Path(
    "docs/data-certificates/sl-corpus-v5-scale8000-otf-strongaug-surface08-diagnostic-v1.json"
)

STANDARD_OTF_METRICS = {
    "piper_synthetic_holdout": {"wer": 13.199, "cer": 3.785, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 4.658, "cer": 1.472, "empty": 0},
    "fleurs_v2": {"wer": 39.353, "cer": 12.002, "empty": 0},
    "artur_j": {"wer": 39.974, "cer": 12.251, "empty": 0},
}

_BASE_PATH = Path(__file__).with_name("run_surface08_scale8000_otf_augmented.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_slaif_scale8000_otf_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import scale-8000 OTF Surface08 runner")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

_ORIGINAL_ITER = _BASE.iter_prefetched_microbatches
_ORIGINAL_DETERMINISM = _BASE.run_determinism_checks_for_exposures
_ACTIVE_POLICY: dict[str, Any] | None = None


def _strong_iter(*args: Any, **kwargs: Any) -> Any:
    if _ACTIVE_POLICY is None:
        raise RuntimeError("StrongAug policy is not loaded")
    kwargs["augmentation_policy"] = _ACTIVE_POLICY
    return _ORIGINAL_ITER(*args, **kwargs)


def _strong_determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if _ACTIVE_POLICY is None:
        raise RuntimeError("StrongAug policy is not loaded")
    kwargs["augmentation_policy"] = _ACTIVE_POLICY
    return _ORIGINAL_DETERMINISM(*args, **kwargs)


def classify_result(
    metrics: dict[str, dict[str, Any]],
    *,
    training: dict[str, Any],
) -> str:
    if float(training["loader_fill_rate"]) < 0.95:
        return "STRONGAUG_V1_LOADER_THROUGHPUT_BLOCKED"
    if not bool(training["parameter_integrity"]["only_surface08_changed"]):
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    synthetic_splits = (
        "piper_synthetic_holdout",
        "supertonic_heldout_voice_holdout",
    )
    if any(int(metrics[split]["empty"]) > 0 for split in real_splits):
        return "STRONGAUG_V1_LABEL_DESTRUCTION"
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
    all_real_improve = all(candidate < prior for _split, _metric, candidate, prior, _tol in comparisons)
    within = all(candidate <= prior + tolerance for _split, _metric, candidate, prior, tolerance in comparisons)
    synthetic_not_collapsed = all(
        int(metrics[split]["empty"]) == 0
        and float(metrics[split]["wer"]) < float(_BASE.BASE_DIRECTIONAL_METRICS[split]["wer"])
        and float(metrics[split]["cer"]) < float(_BASE.BASE_DIRECTIONAL_METRICS[split]["cer"])
        for split in synthetic_splits
    )
    synthetic_tradeoff = any(
        float(metrics[split][metric]) > float(STANDARD_OTF_METRICS[split][metric])
        for split in synthetic_splits
        for metric in ("wer", "cer")
    )
    if all_real_improve and synthetic_not_collapsed:
        if synthetic_tradeoff:
            return "STRONGAUG_V1_REAL_GATE_GAIN_SYNTHETIC_TRADEOFF"
        return "STRONGAUG_V1_NEW_BEST_DIRECTIONAL"
    if within:
        return "STRONGAUG_V1_MATCHES_STANDARD_OTF"
    synthetic_gain = all(
        float(metrics[split][metric]) < float(STANDARD_OTF_METRICS[split][metric])
        for split in synthetic_splits
        for metric in ("wer", "cer")
    )
    if synthetic_gain:
        return "STRONGAUG_V1_SYNTHETIC_ONLY_GAIN"
    return "STRONGAUG_V1_REAL_GATE_REGRESSION"


def _format_metric(value: dict[str, Any]) -> str:
    return f"{float(value['wer']):.3f} / {float(value['cer']):.3f} / {int(value['empty'])}"


def _markdown_report(public: dict[str, Any]) -> str:
    training = public["training"]
    metrics = public["directional_evaluation"]["metrics"]
    audit = public["otf_augmentation"]["audit"]
    lines = [
        "# Experiment 0032: Surface08 Scale-8000 OTF StrongAug v1",
        "",
        f"Classification: `{public['classification']}`",
        "",
        "This diagnostic keeps PR #51's model surface, untouched-base initialization, scale-8000 clean reservoir, optimizer, effective batch, controller policy, and 144,000-exposure cap fixed. The only scientific change is deterministic StrongAug v1 intensity and curriculum. All transformed waveforms were produced in memory.",
        "",
        "## Result",
        "",
        f"- Selected round: {training['selected_round']}; stopped round: {training['stopped_round']} (`{training['stopped_reason']}`).",
        f"- Training exposures: {training['sample_exposures']:,}; unique semantic rows: {training['data_diversity']['unique_semantic_rows_seen']:,}.",
        f"- Aggregate OTF fill rate: {training['loader_fill_rate']:.6f}.",
        f"- Selected checkpoint SHA256: `{public['directional_evaluation']['selected_checkpoint_sha256']}`.",
        "- `accepted_parent` remains `none`; this is noncanonical diagnostic evidence.",
        "",
        "## Augmentation Audit",
        "",
        "| Augmentation family | Fraction | Parameter range | Notes |",
        "|---|---:|---|---|",
        f"| Standard OTF | {audit['standard_fraction']:.6f} | existing 11 profiles | one standard profile per selected sample |",
        f"| Additive noise | {audit['family_sample_fractions'].get('additive_noise', 0.0):.6f} | SNR 8-25 dB; usually 10-25 dB | white, pink, brown, hum, fan/HVAC procedural noise |",
        f"| Reverb | {audit['family_sample_fractions'].get('reverb', 0.0):.6f} | RT60 0.12-0.45 s | deterministic short-to-moderate synthetic room response |",
        f"| Bandpass/filter | {audit['family_sample_fractions'].get('channel_filter', 0.0):.6f} | mild low/high/bandpass/tilt; 8% telephone candidate | lower-SNR plus telephone is prohibited |",
        f"| Codec/companding | {audit['family_sample_fractions'].get('codec_companding', 0.0):.6f} | G.711 mu-law/A-law, 12 kHz round-trip, moderate quantization | deterministic in memory |",
        f"| Gain/compression | {audit['family_sample_fractions'].get('gain_compression', 0.0):.6f} | gain -6 to +6 dB; ratio 1.5-3.0 | mild compression/limiting; rare soft clipping |",
        f"| Speed/pitch | {audit['family_sample_fractions'].get('speed', 0.0):.6f} | speed 0.94-1.06 | no large pitch shift |",
        "",
        "| Round range | Standard OTF % | StrongAug % | Purpose |",
        "|---|---:|---:|---|",
        "| 1-4 | 70 | 30 | broad linguistic coverage with mild robustness |",
        "| 5-9 | 25 | 75 | robustness pressure on repeated rows |",
        "",
        "Observed deterministic mixture:",
        "",
        f"- Coverage phase: `{json.dumps(audit['phases']['coverage'], sort_keys=True)}`",
        f"- Robustness phase: `{json.dumps(audit['phases']['robustness'], sort_keys=True)}`",
        "",
        "## Controller-Dev Curve",
        "",
        "| Round | Step | Exposures | Unique semantic rows | Train loss | Anchor | Scale | ARTUR-dev WER | CER | Empty | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in public["controller_dev"]["curve"]:
        train_loss = "NOT_APPLICABLE" if row["train_loss"] is None else row["train_loss"]
        lines.append(
            f"| {row['round']} | {row['optimizer_step']} | {row['exposures_seen']} | {row.get('unique_semantic_rows', 0)} | {train_loss} | {row['synthetic_anchor_probe_loss']} | {row['synthetic_scale_probe_loss']} | {row['wer']} | {row['cer']} | {row['empty']} | {str(row['eligible']).lower()} |"
        )
    lines.extend(
        [
            "",
            "ARTUR controller-dev aggregate metrics alone selected the checkpoint. FLEURS-v2 and ARTUR-J were evaluated only after selection.",
            "",
            "## Live OTF Loader Telemetry",
            "",
            "| Round | Examples/s | OTF fill rate | Consumer wait % | Queue p50/p95 | Notes |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in training["loader_telemetry"]:
        lines.append(
            f"| {row['round']} | {row['examples_per_second']} | {row['otf_fill_rate']} | {row['consumer_wait_percent']} | {row['queue_p50']} / {row['queue_p95']} | 3 spawned workers; in-memory only |"
        )
    lines.extend(
        [
            "",
            "## Directional Metrics",
            "",
            "| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Surface08 scale-8000 standard OTF | Surface08 scale-8000 StrongAug v1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in (
        "piper_synthetic_holdout",
        "supertonic_heldout_voice_holdout",
        "fleurs_v2",
        "artur_j",
    ):
        lines.append(
            f"| {split} | {_format_metric(metrics['base'][split])} | {_format_metric(metrics['pr36'][split])} | {_format_metric(metrics['surface07'][split])} | {_format_metric(metrics['surface08_scale2000'][split])} | {_format_metric(metrics['surface08_scale8000_standard_otf'][split])} | {_format_metric(metrics['surface08_scale8000_strongaug_v1'][split])} |"
        )
    lines.extend(
        [
            "",
            "Values are normalized WER / CER / empty hypotheses.",
            "",
            "## Boundaries",
            "",
            "- No real speech, S6TTS, scale-2000 audio, or immutable gate was used for training.",
            "- No augmented WAV was pre-rendered, written, or used.",
            "- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.",
            "- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    public = _BASE.stage_summarize(config_path)
    config = load_config(config_path)
    profile_counts = public["training"]["data_diversity"][
        "augmentation_profile_distribution"
    ]
    audit = summarize_profile_counts(profile_counts)
    metrics = public["directional_evaluation"]["metrics"]
    candidate = metrics.pop("surface08_scale8000_otf")
    metrics["surface08_scale8000_standard_otf"] = STANDARD_OTF_METRICS
    metrics["surface08_scale8000_strongaug_v1"] = candidate
    public.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "stacked_on": "PR #51 merge commit d69f70f; Experiment 0031 is the direct comparator",
        }
    )
    public["otf_augmentation"].update(
        {
            "augmentation_key": AUGMENTATION_KEY,
            "algorithm_version": ALGORITHM_VERSION,
            "policy_config_sha256": config["otf_augmentation"][
                "policy_config_sha256"
            ],
            "parameter_ranges": PARAMETER_RANGES,
            "phase_schedule": PHASES,
            "max_operations_per_sample": 3,
            "audit": audit,
        }
    )
    certificate_path = REPO_ROOT / CERTIFICATE_PATH
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate.update(
        {
            "certificate_id": "sl-corpus-v5-scale8000-otf-strongaug-surface08-diagnostic-v1",
            "augmentation_key": AUGMENTATION_KEY,
            "augmentation_algorithm_version": ALGORITHM_VERSION,
            "augmentation_policy_config_sha256": config["otf_augmentation"][
                "policy_config_sha256"
            ],
            "classification": public["classification"],
        }
    )
    _BASE.validate_public_report(public)
    _BASE.validate_public_report(certificate)
    atomic_write_json(REPO_ROOT / REPORT_JSON, public)
    atomic_write_json(certificate_path, certificate)
    atomic_write_text(REPO_ROOT / REPORT_MD, _markdown_report(public))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "classification": public["classification"],
                "selected_round": public["training"]["selected_round"],
                "strong_fraction": audit["strong_fraction"],
            },
            sort_keys=True,
        )
    )
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
    _BASE.iter_prefetched_microbatches = _strong_iter
    _BASE.run_determinism_checks_for_exposures = _strong_determinism
    _BASE.classify_result = classify_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Surface08 on scale-8000 with deterministic StrongAug v1"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "verify-inputs",
            "probe-hardware",
            "probe-surface",
            "probe-otf",
            "probe-microbatch",
            "train",
            "evaluate-directional",
            "summarize",
        ),
    )
    parser.add_argument("--progress-interval", type=float, default=10.0)
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
        "probe-hardware": lambda: _BASE.stage_probe_hardware(config_path),
        "probe-surface": lambda: _BASE.stage_probe_surface(config_path),
        "probe-otf": lambda: _BASE.stage_probe_otf(config_path),
        "probe-microbatch": lambda: _BASE.stage_probe_microbatch(config_path),
        "train": lambda: _BASE.stage_train(config_path, args.progress_interval),
        "evaluate-directional": lambda: _BASE.stage_evaluate_directional(config_path),
        "summarize": lambda: stage_summarize(config_path),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
