from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
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
    deterministic_rng,
    match_source_rms,
    peak,
    resample_ratio,
    rms,
)


AUGMENTATION_KEY = "scale8000-otf-parametric-voiceaug-v1"
ALGORITHM_VERSION = "otf-parametric-voiceaug-v1"
AUGMENTATION_FAMILY = "parametric_voiceaug_v1"
ROUND_SIZE_EXPOSURES = 16_000
MAX_ROUNDS = 9
MAX_OPERATIONS = 3
PHASES = {
    "coverage": {
        "rounds": [1, 2, 3, 4],
        "standard_fraction": 0.5,
        "parametric_fraction": 0.5,
    },
    "robustness": {
        "rounds": [5, 6, 7, 8, 9],
        "standard_fraction": 0.1,
        "parametric_fraction": 0.9,
    },
}
CHAIN_TEMPLATES = (
    ("formant_warp", "pitch_shift"),
    ("formant_warp", "pitch_shift"),
    ("formant_warp", "aperiodicity"),
    ("formant_warp", "aperiodicity"),
    ("formant_warp", "channel_filter"),
    ("formant_warp", "pitch_shift", "aperiodicity"),
    ("formant_warp", "pitch_shift", "aperiodicity"),
    ("pitch_shift", "aperiodicity"),
    ("speed", "additive_noise"),
    ("channel_filter", "codec_companding"),
    ("reverb", "gain_compression"),
    ("pitch_shift", "reverb", "additive_noise"),
    ("formant_warp", "channel_filter", "codec_companding"),
)
DEPENDENCIES = (
    {
        "name": "numpy",
        "version": "2.2.6",
        "license": "BSD-3-Clause",
        "used_for": "array, FFT, and deterministic waveform operations",
        "installation": "pinned project training environment",
        "vendored": False,
    },
    {
        "name": "scipy",
        "version": "1.18.0",
        "license": "BSD-3-Clause",
        "used_for": "filtering, convolution, resampling, STFT, and spectral-envelope warping",
        "installation": "Python wheel in the pinned project training environment",
        "vendored": False,
    },
    {
        "name": "librosa",
        "version": "0.11.0",
        "license": "ISC",
        "used_for": "duration-preserving pitch perturbation",
        "installation": "Python wheel in the pinned project training environment",
        "vendored": False,
    },
    {
        "name": "soxr",
        "version": "1.1.0",
        "license": "LGPL-2.1-or-later",
        "used_for": "librosa pitch-shift resampling backend",
        "installation": "Python wheel dependency of librosa",
        "vendored": False,
    },
    {
        "name": "CPython audioop",
        "version": "3.12 stdlib",
        "license": "PSF-2.0",
        "used_for": "existing repository G.711 mu-law and A-law companding",
        "installation": "CPython standard library",
        "vendored": False,
    },
)
SUPPORTED_OPERATIONS = {
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


def _digest(key: str, *parts: str) -> str:
    return hashlib.sha256(
        (key + "\x1f" + "\x1f".join(parts)).encode("utf-8")
    ).hexdigest()


def _unit(key: str, *parts: str) -> float:
    return int(_digest(key, *parts)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _choice(options: Sequence[Any], key: str, *parts: str) -> Any:
    if not options:
        raise ValueError("deterministic choice requires options")
    return options[int(_digest(key, *parts)[:16], 16) % len(options)]


def _between(low: float, high: float, key: str, *parts: str) -> float:
    return low + (high - low) * _unit(key, *parts)


def verify_dependencies() -> list[dict[str, Any]]:
    resolved = []
    for expected in DEPENDENCIES:
        name = str(expected["name"])
        if name == "CPython audioop":
            import audioop  # noqa: F401

            installed = f"{sys.version_info.major}.{sys.version_info.minor} stdlib"
        else:
            installed = importlib.metadata.version(name)
        if installed != expected["version"]:
            raise RuntimeError(
                f"ParametricVoiceAug dependency {name} must be "
                f"{expected['version']}, found {installed}"
            )
        resolved.append({**expected, "installed_version": installed, "status": "PASSED"})
    return resolved


def validate_policy(policy: dict[str, Any]) -> None:
    expected = {
        "schema_version": "1.0",
        "policy_id": AUGMENTATION_KEY,
        "augmentation_family": AUGMENTATION_FAMILY,
        "augmentation_key": AUGMENTATION_KEY,
        "algorithm_version": ALGORITHM_VERSION,
        "round_size_exposures": ROUND_SIZE_EXPOSURES,
        "max_rounds": MAX_ROUNDS,
        "max_operations_per_sample": MAX_OPERATIONS,
        "disk_output": False,
        "transcript_preserved": True,
        "language_agnostic": True,
        "target_identity_models": False,
        "phases": PHASES,
        "chain_templates": [list(item) for item in CHAIN_TEMPLATES],
        "dependencies": list(DEPENDENCIES),
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"ParametricVoiceAug policy {key} must be {value!r}")
    for chain in CHAIN_TEMPLATES:
        validate_chain([{"family": family, "parameters": {}} for family in chain])
    safety = policy.get("audio_safety", {})
    if safety.get("sample_rate") != 16_000:
        raise ValueError("ParametricVoiceAug output sample rate must be 16 kHz")
    if safety.get("duration_ratio") != [0.88, 1.12]:
        raise ValueError("ParametricVoiceAug duration-ratio bounds drifted")


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ParametricVoiceAug policy must be a JSON object")
    validate_policy(payload)
    return payload


def phase_for_exposure(virtual_exposure_id: int) -> tuple[int, str, float]:
    if virtual_exposure_id < 0:
        raise ValueError("virtual exposure ID must be non-negative")
    round_index = virtual_exposure_id // ROUND_SIZE_EXPOSURES + 1
    if not 1 <= round_index <= MAX_ROUNDS:
        raise ValueError("ParametricVoiceAug exposure exceeds the nine-round budget")
    phase = "coverage" if round_index <= 4 else "robustness"
    return round_index, phase, float(PHASES[phase]["parametric_fraction"])


def _signed_magnitude(
    low: float,
    high: float,
    *,
    seed: str,
    label: str,
) -> float:
    magnitude = _between(low, high, AUGMENTATION_KEY, seed, label, "magnitude")
    sign = -1.0 if _unit(AUGMENTATION_KEY, seed, label, "sign") < 0.5 else 1.0
    return sign * magnitude


def _side_range(
    low_negative: float,
    high_negative: float,
    low_positive: float,
    high_positive: float,
    *,
    seed: str,
    label: str,
) -> float:
    if _unit(AUGMENTATION_KEY, seed, label, "side") < 0.5:
        return _between(
            low_negative,
            high_negative,
            AUGMENTATION_KEY,
            seed,
            label,
            "value",
        )
    return _between(
        low_positive,
        high_positive,
        AUGMENTATION_KEY,
        seed,
        label,
        "value",
    )


def _operation(family: str, *, identity: str, operation_index: int) -> dict[str, Any]:
    seed = _digest(AUGMENTATION_KEY, identity, family, str(operation_index))
    if family == "pitch_shift":
        rare = _unit(AUGMENTATION_KEY, seed, "rare") < 0.15
        semitones = _signed_magnitude(
            2.0 if rare else 1.0,
            3.0 if rare else 2.0,
            seed=seed,
            label="semitones",
        )
        parameters = {"semitones": semitones}
    elif family == "speed":
        rare = _unit(AUGMENTATION_KEY, seed, "rare") < 0.15
        if rare:
            rate = _side_range(
                0.90,
                0.92,
                1.08,
                1.10,
                seed=seed,
                label="rate",
            )
        else:
            rate = _side_range(
                0.92,
                0.98,
                1.02,
                1.08,
                seed=seed,
                label="rate",
            )
        parameters = {"rate": rate}
    elif family == "formant_warp":
        rare = _unit(AUGMENTATION_KEY, seed, "rare") < 0.12
        if rare:
            factor = _side_range(
                0.84,
                0.88,
                1.12,
                1.16,
                seed=seed,
                label="factor",
            )
        else:
            factor = _side_range(
                0.88,
                0.97,
                1.03,
                1.12,
                seed=seed,
                label="factor",
            )
        parameters = {"factor": factor, "method": "stft_log_spectral_envelope_warp"}
    elif family == "aperiodicity":
        parameters = {
            "breathiness_snr_db": _between(
                20.0, 32.0, AUGMENTATION_KEY, seed, "breathiness-snr"
            ),
            "highpass_hz": _between(
                2500.0, 5000.0, AUGMENTATION_KEY, seed, "highpass"
            ),
        }
    elif family == "channel_filter":
        selector = _unit(AUGMENTATION_KEY, seed, "filter-selector")
        filter_name = (
            "telephone_like"
            if selector < 0.12
            else _choice(
                ("mild_low_pass", "mild_high_pass", "band_pass", "broad_eq_tilt"),
                AUGMENTATION_KEY,
                seed,
                "filter",
            )
        )
        parameters = {"filter": filter_name}
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
    elif family == "reverb":
        parameters = {
            "rt60_seconds": _between(
                0.15, 0.80, AUGMENTATION_KEY, seed, "rt60"
            )
        }
    elif family == "additive_noise":
        strong = _unit(AUGMENTATION_KEY, seed, "strong") < 0.15
        parameters = {
            "noise_type": _choice(
                ("white", "pink", "brown", "hum", "fan_hvac"),
                AUGMENTATION_KEY,
                seed,
                "noise-type",
            ),
            "snr_db": _between(
                8.0 if strong else 12.0,
                12.0 if strong else 25.0,
                AUGMENTATION_KEY,
                seed,
                "snr",
            ),
        }
    elif family == "gain_compression":
        selector = _unit(AUGMENTATION_KEY, seed, "dynamics-selector")
        mode = "soft_clip" if selector < 0.06 else (
            "limiting" if selector < 0.25 else "compression"
        )
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
    else:
        raise ValueError(f"unsupported ParametricVoiceAug operation {family}")
    return {"family": family, "parameters": parameters, "seed": seed}


def validate_chain(operations: Sequence[dict[str, Any]]) -> None:
    if not 2 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError("ParametricVoiceAug chains must contain two or three operations")
    families = [str(item["family"]) for item in operations]
    if len(families) != len(set(families)):
        raise ValueError("ParametricVoiceAug chains cannot repeat an operation")
    unknown = set(families) - SUPPORTED_OPERATIONS
    if unknown:
        raise ValueError(f"unsupported ParametricVoiceAug operations: {sorted(unknown)}")
    if {"additive_noise", "reverb", "codec_companding"}.issubset(families):
        raise ValueError("extreme noise, reverb, and codec cannot be combined")
    if "additive_noise" in families and "channel_filter" in families:
        noise = next(item for item in operations if item["family"] == "additive_noise")
        channel = next(item for item in operations if item["family"] == "channel_filter")
        if (
            channel.get("parameters", {}).get("filter") == "telephone_like"
            and float(noise.get("parameters", {}).get("snr_db", 25.0)) < 10.0
        ):
            raise ValueError("telephone filtering cannot combine with sub-10 dB noise")


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
        raise ValueError("ParametricVoiceAug requires the existing eleven-profile space")
    round_index, phase, parametric_fraction = phase_for_exposure(virtual_exposure_id)
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
    if _unit(AUGMENTATION_KEY, identity, "mode") >= parametric_fraction:
        profile_index = int(_digest(AUGMENTATION_KEY, identity, "profile")[:16], 16) % 11
        profile = profiles[profile_index]
        from slaif_asr.transcript_preserving_augmentation import parameters_for_profile

        source_parameters = parameters_for_profile(
            profile, semantic_key=parameter_seed
        )
        profile_id = f"standard_otf/{phase}/{profile['profile_id']}"
        parameters = {
            "mode": "standard",
            "phase": phase,
            "round": round_index,
            "source_profile_id": str(profile["profile_id"]),
            "source_parameters": source_parameters,
            "transcript_preserved": True,
        }
    else:
        template_index = (
            int(_digest(AUGMENTATION_KEY, identity, "chain")[:16], 16)
            % len(CHAIN_TEMPLATES)
        )
        families = CHAIN_TEMPLATES[template_index]
        operations = [
            _operation(family, identity=identity, operation_index=index)
            for index, family in enumerate(families)
        ]
        validate_chain(operations)
        profile_index = 11 + template_index
        profile_id = f"parametric_voiceaug_v1/{phase}/{'+'.join(families)}"
        parameters = {
            "mode": "parametric",
            "phase": phase,
            "round": round_index,
            "operations": operations,
            "transcript_preserved": True,
            "language_agnostic": True,
            "target_identity": None,
        }
    return {
        "profile_id": profile_id,
        "profile_index": profile_index,
        "parameters": parameters,
        "parameter_seed": parameter_seed,
        "augmentation_identity_sha256": identity,
        "profile_space_sha256": profile_space,
    }


def _fix_length(samples: Any, length: int) -> Any:
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if values.size >= length:
        return values[:length]
    return np.pad(values, (0, length - values.size))


def _pitch_shift(samples: Any, semitones: float) -> Any:
    import librosa
    import numpy as np

    if not -3.0 <= semitones <= 3.0 or abs(semitones) < 0.5:
        raise ValueError("pitch shift must remain between 0.5 and 3 semitones")
    transformed = librosa.effects.pitch_shift(
        np.asarray(samples, dtype=np.float32),
        sr=16_000,
        n_steps=float(semitones),
        bins_per_octave=12,
        res_type="soxr_hq",
        scale=False,
    )
    return _fix_length(transformed, len(samples))


def _formant_warp(samples: Any, factor: float) -> Any:
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import istft, stft

    if not 0.84 <= factor <= 1.16 or 0.98 < factor < 1.02:
        raise ValueError("formant warp factor is outside the declared range")
    _frequencies, _times, spectrum = stft(
        np.asarray(samples, dtype=np.float64),
        fs=16_000,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=512,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.maximum(np.abs(spectrum), 1e-8)
    envelope = gaussian_filter1d(
        np.log(magnitude), sigma=5.0, axis=0, mode="nearest"
    )
    positions = np.arange(envelope.shape[0], dtype=np.float64) / factor
    positions = np.clip(positions, 0.0, envelope.shape[0] - 1.0)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, envelope.shape[0] - 1)
    weight = (positions - lower)[:, None]
    warped = envelope[lower] * (1.0 - weight) + envelope[upper] * weight
    gain = np.exp(np.clip(warped - envelope, -1.0, 1.0))
    _time, transformed = istft(
        spectrum * gain,
        fs=16_000,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=512,
        input_onesided=True,
        boundary=True,
    )
    return _fix_length(transformed, len(samples))


def _add_aperiodicity(
    samples: Any,
    *,
    breathiness_snr_db: float,
    highpass_hz: float,
    seed_text: str,
) -> Any:
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, sosfilt

    if not 20.0 <= breathiness_snr_db <= 32.0:
        raise ValueError("breathiness SNR is outside the declared range")
    if not 2500.0 <= highpass_hz <= 5000.0:
        raise ValueError("aperiodicity high-pass frequency is outside range")
    rng = deterministic_rng(seed_text)
    noise = rng.normal(0.0, 1.0, size=len(samples))
    sos = butter(3, highpass_hz, btype="highpass", fs=16_000, output="sos")
    noise = sosfilt(sos, noise)
    envelope = uniform_filter1d(
        np.abs(np.asarray(samples, dtype=np.float64)), size=401, mode="nearest"
    )
    shaped = noise * (envelope + 0.05 * max(float(envelope.max()), 1e-8))
    if rms(shaped) == 0.0 or rms(samples) == 0.0:
        return samples
    target = rms(samples) / db_to_linear(breathiness_snr_db)
    return samples + shaped * (target / rms(shaped))


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


def validate_audio(
    source: Any,
    transformed: Any,
    *,
    sample_rate: int = 16_000,
    duration_ratio_bounds: tuple[float, float] | None = (0.88, 1.12),
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(transformed)
    if sample_rate != 16_000:
        raise ValueError("ParametricVoiceAug output must be 16 kHz")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ParametricVoiceAug output must be non-empty mono audio")
    if not np.isfinite(values).all():
        raise ValueError("ParametricVoiceAug output contains NaN or Inf")
    output_rms = rms(values)
    if output_rms < 1e-5:
        raise ValueError("ParametricVoiceAug output is silent or near-silent")
    output_peak = peak(values)
    if output_peak > 0.980001:
        raise ValueError("ParametricVoiceAug output exceeds the peak limit")
    clipped_fraction = float(np.mean(np.abs(values) >= 0.979999))
    if clipped_fraction > 0.01:
        raise ValueError("ParametricVoiceAug output has severe clipping")
    if getattr(source, "size", 0) <= 0:
        raise ValueError("ParametricVoiceAug source is empty")
    duration_ratio = float(values.size / len(source))
    if duration_ratio_bounds is not None:
        minimum_ratio, maximum_ratio = duration_ratio_bounds
        if not minimum_ratio <= duration_ratio <= maximum_ratio:
            raise ValueError("ParametricVoiceAug duration ratio is outside bounds")
    return {
        "sample_rate": sample_rate,
        "duration_ratio": round(duration_ratio, 8),
        "rms": round(output_rms, 8),
        "peak": round(output_peak, 8),
        "clipped_fraction": round(clipped_fraction, 8),
        "finite": True,
        "non_silent": True,
    }


def apply_policy_transform(
    samples: Any,
    *,
    parameters: dict[str, Any],
    seed_text: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    mode = str(parameters["mode"])
    if mode == "standard":
        transformed, details = apply_profile_transform(
            samples,
            str(parameters["source_profile_id"]),
            dict(parameters["source_parameters"]),
            seed_text=seed_text,
        )
        # The fixed Experiment 0031 profiles retain their own timing-silence
        # contract; the tighter ratio bound applies only to new parametric chains.
        validation = validate_audio(
            samples,
            transformed,
            duration_ratio_bounds=None,
        )
        return transformed, {**details, "mode": "standard", "validation": validation}
    if mode != "parametric":
        raise ValueError("unknown ParametricVoiceAug sample mode")
    operations = list(parameters["operations"])
    validate_chain(operations)
    transformed = np.asarray(samples, dtype=np.float64)
    for operation in operations:
        family = str(operation["family"])
        values = dict(operation["parameters"])
        operation_seed = str(operation["seed"])
        if family == "pitch_shift":
            transformed = _pitch_shift(transformed, float(values["semitones"]))
        elif family == "speed":
            rate = float(values["rate"])
            if not 0.90 <= rate <= 1.10:
                raise ValueError("speed perturbation is outside range")
            transformed = resample_ratio(transformed, rate)
        elif family == "formant_warp":
            transformed = _formant_warp(transformed, float(values["factor"]))
        elif family == "aperiodicity":
            transformed = _add_aperiodicity(
                transformed,
                breathiness_snr_db=float(values["breathiness_snr_db"]),
                highpass_hz=float(values["highpass_hz"]),
                seed_text=operation_seed,
            )
        elif family == "channel_filter":
            transformed = apply_filter(transformed, str(values["filter"]))
        elif family == "codec_companding":
            transformed = codec_simulation(transformed, str(values["codec"]))
        elif family == "reverb":
            rt60 = float(values["rt60_seconds"])
            if not 0.15 <= rt60 <= 0.80:
                raise ValueError("reverb RT60 is outside range")
            transformed = apply_rir(
                transformed, rt60_seconds=rt60, seed_text=operation_seed
            )
        elif family == "additive_noise":
            noise_type = str(values["noise_type"])
            snr_db = float(values["snr_db"])
            if not 8.0 <= snr_db <= 25.0:
                raise ValueError("additive-noise SNR is outside range")
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
        elif family == "gain_compression":
            transformed = _apply_dynamics(transformed, values)
        else:
            raise ValueError(f"unsupported ParametricVoiceAug operation {family}")
    transformed, safety = match_source_rms(samples, transformed, peak_limit=0.98)
    validation = validate_audio(samples, transformed)
    return transformed, {
        "mode": "parametric",
        "operation_count": len(operations),
        "operation_families": [str(item["family"]) for item in operations],
        "validation": validation,
        **safety,
    }


def summarize_profile_counts(profile_counts: dict[str, int]) -> dict[str, Any]:
    total_samples = sum(int(value) for value in profile_counts.values())
    standard = 0
    parametric = 0
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
        elif mode == "parametric_voiceaug_v1":
            parametric += count
            phase_counts[phase]["parametric"] += count
            for family in parts[2].split("+"):
                family_counts[family] += count
    phase_summary: dict[str, Any] = {}
    for phase, counts in phase_counts.items():
        phase_total = counts["standard"] + counts["parametric"]
        phase_summary[phase] = {
            "samples": phase_total,
            "standard_samples": counts["standard"],
            "parametric_samples": counts["parametric"],
            "standard_fraction": round(
                counts["standard"] / phase_total, 6
            ) if phase_total else 0.0,
            "parametric_fraction": round(
                counts["parametric"] / phase_total, 6
            ) if phase_total else 0.0,
        }
    return {
        "total_samples": total_samples,
        "standard_samples": standard,
        "parametric_samples": parametric,
        "standard_fraction": round(standard / total_samples, 6)
        if total_samples
        else 0.0,
        "parametric_fraction": round(parametric / total_samples, 6)
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
