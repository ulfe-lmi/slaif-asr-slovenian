from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.transcript_preserving_augmentation import (
    add_hum,
    add_noise,
    apply_filter,
    apply_profile_transform,
    apply_rir,
    codec_simulation,
    db_to_linear,
    match_source_rms,
    parameters_for_profile,
    resample_ratio,
)


AUGMENTATION_KEY = "scale8000-otf-strongaug-v1"
ALGORITHM_VERSION = "otf-strongaug-v1"
ROUND_SIZE = 16_000
MAX_ROUNDS = 9
MAX_OPERATIONS = 3

PHASES = {
    "coverage": {
        "rounds": [1, 2, 3, 4],
        "standard_fraction": 0.70,
        "strong_fraction": 0.30,
    },
    "robustness": {
        "rounds": [5, 6, 7, 8, 9],
        "standard_fraction": 0.25,
        "strong_fraction": 0.75,
    },
}

CHAIN_TEMPLATES = (
    ("additive_noise", "channel_filter"),
    ("additive_noise", "reverb"),
    ("reverb", "gain_compression"),
    ("codec_companding", "channel_filter"),
    ("speed", "additive_noise"),
    ("additive_noise", "channel_filter", "gain_compression"),
    ("reverb", "channel_filter", "gain_compression"),
)

PARAMETER_RANGES = {
    "standard_otf": {
        "description": "Existing eleven transcript-preserving profile families",
    },
    "additive_noise": {
        "noise_types": ["white", "pink", "brown", "hum", "fan_hvac"],
        "snr_db": [8.0, 25.0],
        "usual_snr_db": [10.0, 25.0],
    },
    "channel_filter": {
        "filters": [
            "mild_low_pass",
            "mild_high_pass",
            "band_pass",
            "telephone_like",
            "broad_eq_tilt",
        ],
        "telephone_probability": 0.08,
    },
    "reverb": {"rt60_seconds": [0.12, 0.45]},
    "gain_compression": {
        "gain_db": [-6.0, 6.0],
        "compression_ratio": [1.5, 3.0],
        "soft_clip_probability": 0.08,
    },
    "codec_companding": {
        "codecs": [
            "g711_mulaw",
            "g711_alaw",
            "downsample_12k_return_16k",
            "ogg_opus_moderate",
        ]
    },
    "speed": {"rate": [0.94, 1.06]},
}


def _digest(key: str, *parts: str) -> str:
    message = "\x1f".join(parts).encode("utf-8")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _unit(key: str, *parts: str) -> float:
    return int(_digest(key, *parts)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _choice(options: Sequence[Any], key: str, *parts: str) -> Any:
    if not options:
        raise ValueError("deterministic choice requires options")
    return options[int(_digest(key, *parts)[:16], 16) % len(options)]


def _between(low: float, high: float, key: str, *parts: str) -> float:
    return round(low + (high - low) * _unit(key, *parts), 6)


def validate_policy(policy: dict[str, Any]) -> None:
    expected = {
        "schema_version": "1.0",
        "policy_id": "scale8000-otf-strongaug-v1",
        "augmentation_key": AUGMENTATION_KEY,
        "algorithm_version": ALGORITHM_VERSION,
        "round_size_exposures": ROUND_SIZE,
        "max_rounds": MAX_ROUNDS,
        "max_operations_per_sample": MAX_OPERATIONS,
        "disk_output": False,
        "transcript_preserved": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"StrongAug policy {key} must be {value!r}")
    if policy.get("phases") != PHASES:
        raise ValueError("StrongAug curriculum phases drifted")
    if policy.get("chain_templates") != [list(item) for item in CHAIN_TEMPLATES]:
        raise ValueError("StrongAug chain templates drifted")
    ranges = policy.get("parameter_ranges", {})
    if ranges != PARAMETER_RANGES:
        raise ValueError("StrongAug parameter ranges drifted")
    if float(ranges["additive_noise"]["snr_db"][0]) < 8.0:
        raise ValueError("StrongAug v1 noise floor cannot be below 8 dB SNR")


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("StrongAug policy must be a JSON object")
    validate_policy(payload)
    return payload


def phase_for_exposure(virtual_exposure_id: int) -> tuple[int, str, float]:
    if virtual_exposure_id < 0:
        raise ValueError("virtual exposure ID must be non-negative")
    round_index = virtual_exposure_id // ROUND_SIZE + 1
    if round_index > MAX_ROUNDS:
        raise ValueError("StrongAug v1 exposure exceeds the nine-round budget")
    phase = "coverage" if round_index <= 4 else "robustness"
    return round_index, phase, float(PHASES[phase]["strong_fraction"])


def _operation(
    family: str,
    *,
    identity: str,
    operation_index: int,
) -> dict[str, Any]:
    seed = _digest(AUGMENTATION_KEY, identity, family, str(operation_index))
    if family == "additive_noise":
        noise_type = _choice(
            ("white", "pink", "brown", "hum", "fan_hvac"),
            AUGMENTATION_KEY,
            seed,
            "noise-type",
        )
        low_snr_case = _unit(AUGMENTATION_KEY, seed, "low-snr-case") < 0.10
        snr_db = (
            _between(8.0, 10.0, AUGMENTATION_KEY, seed, "snr")
            if low_snr_case
            else _between(10.0, 25.0, AUGMENTATION_KEY, seed, "snr")
        )
        parameters = {"noise_type": noise_type, "snr_db": snr_db}
    elif family == "channel_filter":
        selector = _unit(AUGMENTATION_KEY, seed, "filter-selector")
        if selector < 0.08:
            filter_name = "telephone_like"
        else:
            filter_name = _choice(
                ("mild_low_pass", "mild_high_pass", "band_pass", "broad_eq_tilt"),
                AUGMENTATION_KEY,
                seed,
                "filter",
            )
        parameters = {"filter": filter_name}
    elif family == "reverb":
        parameters = {
            "rt60_seconds": _between(0.12, 0.45, AUGMENTATION_KEY, seed, "rt60")
        }
    elif family == "gain_compression":
        selector = _unit(AUGMENTATION_KEY, seed, "dynamics-selector")
        if selector < 0.08:
            mode = "soft_clip"
        elif selector < 0.30:
            mode = "limiting"
        else:
            mode = "compression"
        parameters = {
            "mode": mode,
            "gain_db": _between(-6.0, 6.0, AUGMENTATION_KEY, seed, "gain"),
            "compression_ratio": _between(
                1.5, 3.0, AUGMENTATION_KEY, seed, "compression-ratio"
            ),
            "threshold": _between(
                0.32, 0.58, AUGMENTATION_KEY, seed, "compression-threshold"
            ),
            "soft_clip_drive": _between(
                1.1, 1.6, AUGMENTATION_KEY, seed, "soft-clip-drive"
            ),
        }
    elif family == "codec_companding":
        parameters = {
            "codec": _choice(
                (
                    "g711_mulaw",
                    "g711_alaw",
                    "downsample_12k_return_16k",
                    "ogg_opus_moderate",
                ),
                AUGMENTATION_KEY,
                seed,
                "codec",
            )
        }
    elif family == "speed":
        parameters = {
            "rate": _between(0.94, 1.06, AUGMENTATION_KEY, seed, "speed")
        }
    else:
        raise ValueError(f"unsupported StrongAug operation family {family}")
    return {"family": family, "parameters": parameters, "seed": seed}


def spec_payload(
    record: Any,
    virtual_exposure_id: int,
    profiles: Sequence[dict[str, Any]],
    *,
    corpus_id: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    validate_policy(policy)
    if len(profiles) != 11:
        raise ValueError("StrongAug requires the existing eleven-profile space")
    round_index, phase, strong_fraction = phase_for_exposure(virtual_exposure_id)
    profile_space = hashlib.sha256(
        json.dumps(list(profiles), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = _digest(
        AUGMENTATION_KEY,
        corpus_id,
        str(record.semantic_key),
        str(record.voice),
        str(record.source_audio_sha256),
        str(virtual_exposure_id),
        str(round_index),
        phase,
        profile_space,
        AUGMENTATION_KEY,
        ALGORITHM_VERSION,
    )
    parameter_seed = _digest(AUGMENTATION_KEY, identity, "parameters")
    strong = _unit(AUGMENTATION_KEY, identity, "mixture") < strong_fraction
    if not strong:
        profile_index = int(_digest(AUGMENTATION_KEY, identity, "standard-profile")[:16], 16) % len(
            profiles
        )
        profile = profiles[profile_index]
        parameters = {
            "mode": "standard",
            "phase": phase,
            "round": round_index,
            "source_profile_id": str(profile["profile_id"]),
            "source_parameters": parameters_for_profile(
                profile, semantic_key=parameter_seed
            ),
        }
        profile_id = f"standard_otf/{phase}/{profile['profile_id']}"
    else:
        template = _choice(
            CHAIN_TEMPLATES, AUGMENTATION_KEY, identity, "chain-template"
        )
        operations = [
            _operation(family, identity=identity, operation_index=index)
            for index, family in enumerate(template)
        ]
        noise = next(
            (
                item
                for item in operations
                if item["family"] == "additive_noise"
            ),
            None,
        )
        channel = next(
            (
                item
                for item in operations
                if item["family"] == "channel_filter"
            ),
            None,
        )
        if (
            noise is not None
            and channel is not None
            and float(noise["parameters"]["snr_db"]) < 10.0
            and channel["parameters"]["filter"] == "telephone_like"
        ):
            channel["parameters"]["filter"] = "band_pass"
        parameters = {
            "mode": "strong",
            "phase": phase,
            "round": round_index,
            "operations": operations,
        }
        chain_name = "+".join(str(item) for item in template)
        profile_id = f"strongaug_v1/{phase}/{chain_name}"
        profile_index = -1
    return {
        "profile_id": profile_id,
        "profile_index": profile_index,
        "parameters": parameters,
        "parameter_seed": parameter_seed,
        "augmentation_identity_sha256": identity,
        "profile_space_sha256": profile_space,
    }


def _apply_dynamics(samples: Any, parameters: dict[str, Any]) -> Any:
    import numpy as np

    transformed = samples * db_to_linear(float(parameters["gain_db"]))
    mode = str(parameters["mode"])
    if mode == "compression":
        threshold = float(parameters["threshold"])
        ratio = float(parameters["compression_ratio"])
        magnitude = np.abs(transformed)
        compressed = np.where(
            magnitude <= threshold,
            magnitude,
            threshold + (magnitude - threshold) / ratio,
        )
        return np.sign(transformed) * compressed
    if mode == "limiting":
        return np.clip(transformed, -0.85, 0.85)
    if mode == "soft_clip":
        drive = float(parameters["soft_clip_drive"])
        return np.tanh(transformed * drive) / math.tanh(drive)
    raise ValueError(f"unsupported dynamics mode {mode}")


def apply_policy_transform(
    samples: Any,
    *,
    parameters: dict[str, Any],
    seed_text: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    mode = str(parameters["mode"])
    if mode == "standard":
        return apply_profile_transform(
            samples,
            str(parameters["source_profile_id"]),
            dict(parameters["source_parameters"]),
            seed_text=seed_text,
        )
    if mode != "strong":
        raise ValueError("unknown StrongAug sample mode")
    operations = list(parameters["operations"])
    if not 2 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError("StrongAug chains must contain two or three operations")
    transformed = np.asarray(samples, dtype=np.float64)
    for operation in operations:
        family = str(operation["family"])
        values = dict(operation["parameters"])
        operation_seed = str(operation["seed"])
        if family == "additive_noise":
            noise_type = str(values["noise_type"])
            snr_db = float(values["snr_db"])
            if snr_db < 8.0:
                raise ValueError("StrongAug v1 SNR cannot be below 8 dB")
            if noise_type == "hum":
                transformed = add_hum(transformed, snr_db=snr_db)
            elif noise_type == "fan_hvac":
                transformed = add_noise(
                    transformed,
                    snr_db=snr_db,
                    seed_text=operation_seed,
                    colour="low",
                )
                transformed = add_hum(transformed, snr_db=snr_db + 6.0)
            else:
                transformed = add_noise(
                    transformed,
                    snr_db=snr_db,
                    seed_text=operation_seed,
                    colour=noise_type,
                )
        elif family == "channel_filter":
            transformed = apply_filter(transformed, str(values["filter"]))
        elif family == "reverb":
            transformed = apply_rir(
                transformed,
                rt60_seconds=float(values["rt60_seconds"]),
                seed_text=operation_seed,
            )
        elif family == "gain_compression":
            transformed = _apply_dynamics(transformed, values)
        elif family == "codec_companding":
            transformed = codec_simulation(transformed, str(values["codec"]))
        elif family == "speed":
            transformed = resample_ratio(transformed, float(values["rate"]))
        else:
            raise ValueError(f"unsupported StrongAug operation {family}")
    transformed, safety = match_source_rms(samples, transformed)
    if transformed.ndim != 1 or transformed.size == 0 or not np.isfinite(transformed).all():
        raise RuntimeError("StrongAug produced an invalid waveform")
    return transformed, {
        "mode": "strong",
        "operation_count": len(operations),
        "operation_families": [str(item["family"]) for item in operations],
        **safety,
    }


def summarize_profile_counts(profile_counts: dict[str, int]) -> dict[str, Any]:
    total_samples = sum(int(value) for value in profile_counts.values())
    standard = 0
    strong = 0
    phase_counts: dict[str, Counter[str]] = {
        "coverage": Counter(),
        "robustness": Counter(),
    }
    family_counts: Counter[str] = Counter()
    for profile_id, raw_count in profile_counts.items():
        count = int(raw_count)
        parts = profile_id.split("/")
        if len(parts) < 3:
            continue
        mode, phase = parts[0], parts[1]
        if phase not in phase_counts:
            continue
        if mode == "standard_otf":
            standard += count
            phase_counts[phase]["standard"] += count
        elif mode == "strongaug_v1":
            strong += count
            phase_counts[phase]["strong"] += count
            for family in parts[2].split("+"):
                family_counts[family] += count
    phase_summary: dict[str, Any] = {}
    for phase, counts in phase_counts.items():
        phase_total = counts["standard"] + counts["strong"]
        phase_summary[phase] = {
            "samples": phase_total,
            "standard_samples": counts["standard"],
            "strong_samples": counts["strong"],
            "standard_fraction": round(counts["standard"] / phase_total, 6)
            if phase_total
            else 0.0,
            "strong_fraction": round(counts["strong"] / phase_total, 6)
            if phase_total
            else 0.0,
        }
    return {
        "total_samples": total_samples,
        "standard_samples": standard,
        "strong_samples": strong,
        "standard_fraction": round(standard / total_samples, 6)
        if total_samples
        else 0.0,
        "strong_fraction": round(strong / total_samples, 6)
        if total_samples
        else 0.0,
        "family_sample_counts": dict(sorted(family_counts.items())),
        "family_sample_fractions": {
            key: round(value / total_samples, 6)
            for key, value in sorted(family_counts.items())
        }
        if total_samples
        else {},
        "phases": phase_summary,
    }
