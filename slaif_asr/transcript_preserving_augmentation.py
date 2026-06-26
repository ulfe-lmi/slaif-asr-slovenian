from __future__ import annotations

import audioop
import math
import os
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.acoustic_quality import read_audio_stats
from slaif_asr.scale200_corpus import stable_sha256


@dataclass(frozen=True)
class AugmentationAssignment:
    semantic_key: str
    profile_id: str
    source_voice: str
    parameter_seed: str


def deterministic_unit_interval(*parts: str) -> float:
    digest = stable_sha256(":".join(parts))
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def interpolate(minimum: float, maximum: float, value: float) -> float:
    return minimum + (maximum - minimum) * value


def choose_from(options: Sequence[Any], *, semantic_key: str, profile_id: str) -> Any:
    if not options:
        raise ValueError("options must not be empty")
    digest = stable_sha256(f"{semantic_key}:{profile_id}")
    return options[int(digest[:8], 16) % len(options)]


def assignment_for(
    *,
    semantic_key: str,
    semantic_position: int,
    profile_id: str,
    profile_index: int,
    clean_voices: Sequence[str],
) -> AugmentationAssignment:
    if not clean_voices:
        raise ValueError("clean voices are required")
    voice = clean_voices[(semantic_position + profile_index) % len(clean_voices)]
    return AugmentationAssignment(
        semantic_key=semantic_key,
        profile_id=profile_id,
        source_voice=voice,
        parameter_seed=stable_sha256(f"{semantic_key}:{profile_id}:{voice}"),
    )


def parameters_for_profile(profile: dict[str, Any], *, semantic_key: str) -> dict[str, Any]:
    profile_id = str(profile["profile_id"])
    params = dict(profile.get("parameters", {}))
    unit = deterministic_unit_interval(semantic_key, profile_id)
    if profile_id == "coupled_speed_pitch_resampling":
        return {"rate": choose_from(params["rates"], semantic_key=semantic_key, profile_id=profile_id)}
    if profile_id == "tempo_preserving_pitch":
        return {"tempo_factor": round(interpolate(params["minimum_factor"], params["maximum_factor"], unit), 6)}
    if profile_id == "mild_pitch_formant_vtlp_proxy":
        return {
            "pitch_semitones": round(interpolate(params["minimum_semitones"], params["maximum_semitones"], unit), 6),
            "formant_warp": round(interpolate(params["minimum_warp"], params["maximum_warp"], 1.0 - unit), 6),
        }
    if profile_id == "procedural_room_impulse_response":
        return {"rt60_seconds": round(interpolate(params["minimum_rt60_seconds"], params["maximum_rt60_seconds"], unit), 6)}
    if profile_id in {"environmental_background_noise", "coloured_electrical_noise", "compound_realistic_condition"}:
        return {"snr_db": round(interpolate(params["minimum_snr_db"], params["maximum_snr_db"], unit), 6)}
    if profile_id == "microphone_channel_filtering":
        return {"filter": choose_from(params["filters"], semantic_key=semantic_key, profile_id=profile_id)}
    if profile_id == "codec_sample_rate_simulation":
        return {"codec": choose_from(params["codecs"], semantic_key=semantic_key, profile_id=profile_id)}
    if profile_id == "gain_dynamic_range_variation":
        return {
            "gain_db": round(interpolate(params["minimum_gain_db"], params["maximum_gain_db"], unit), 6),
            "peak_safety": params["peak_safety"],
        }
    if profile_id == "timing_silence_variation":
        return {
            "leading_silence_seconds": round(params["maximum_leading_silence_seconds"] * unit, 6),
            "trailing_silence_seconds": round(params["maximum_trailing_silence_seconds"] * (1.0 - unit), 6),
            "time_shift_seconds": round(interpolate(-params["maximum_time_shift_seconds"], params["maximum_time_shift_seconds"], unit), 6),
        }
    raise ValueError(f"unsupported augmentation profile {profile_id}")


def db_to_linear(db: float) -> float:
    return math.pow(10.0, db / 20.0)


def read_mono_pcm16(path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16 kHz signed 16-bit PCM WAV")
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0, frames


def write_mono_pcm16(path: Path, samples: Any) -> None:
    import numpy as np

    peak = float(np.max(np.abs(samples))) if getattr(samples, "size", 0) else 0.0
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    samples = np.clip(samples, -0.98, 0.98)
    pcm = np.rint(samples * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.part.{os.getpid()}{path.suffix}")
    with wave.open(str(temp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    os.replace(temp, path)


def rms(samples: Any) -> float:
    import numpy as np

    if getattr(samples, "size", 0) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def peak(samples: Any) -> float:
    import numpy as np

    if getattr(samples, "size", 0) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def deterministic_rng(seed_text: str) -> Any:
    import numpy as np

    seed = int(stable_sha256(seed_text)[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def resample_ratio(samples: Any, ratio: float) -> Any:
    from scipy.signal import resample_poly

    fraction = Fraction(float(ratio)).limit_denominator(1000)
    return resample_poly(samples, fraction.numerator, fraction.denominator)


def match_source_rms(source: Any, transformed: Any, *, peak_limit: float = 0.98) -> tuple[Any, dict[str, float]]:
    source_rms = rms(source)
    transformed_rms = rms(transformed)
    rms_gain = 1.0
    if source_rms > 0.0 and transformed_rms > 0.0:
        rms_gain = source_rms / transformed_rms
        transformed = transformed * rms_gain
    before_peak = peak(transformed)
    safety_gain = 1.0
    if before_peak > peak_limit:
        safety_gain = peak_limit / before_peak
        transformed = transformed * safety_gain
    return transformed, {
        "source_rms": round(source_rms, 8),
        "raw_output_rms": round(transformed_rms, 8),
        "applied_rms_gain": round(rms_gain, 8),
        "peak_safety_gain": round(safety_gain, 8),
    }


def _noise_like(samples: Any, *, seed_text: str, colour: str) -> Any:
    import numpy as np
    from scipy.signal import butter, sosfilt

    rng = deterministic_rng(seed_text)
    noise = rng.normal(0.0, 1.0, size=len(samples))
    if colour == "low":
        sos = butter(3, 700.0, btype="lowpass", fs=16000, output="sos")
        noise = sosfilt(sos, noise)
    elif colour == "pink":
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1 / 16000)
        scale = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
        noise = np.fft.irfft(spectrum * scale, n=len(noise))
    elif colour == "brown":
        noise = np.cumsum(noise)
    if peak(noise) > 0:
        noise = noise / peak(noise)
    return noise


def add_noise(samples: Any, *, snr_db: float, seed_text: str, colour: str) -> Any:
    noise = _noise_like(samples, seed_text=seed_text, colour=colour)
    signal_rms = rms(samples)
    noise_rms = rms(noise)
    if signal_rms == 0.0 or noise_rms == 0.0:
        return samples
    target_noise_rms = signal_rms / db_to_linear(snr_db)
    return samples + noise * (target_noise_rms / noise_rms)


def add_hum(samples: Any, *, snr_db: float) -> Any:
    import numpy as np

    t = np.arange(len(samples), dtype=np.float64) / 16000.0
    hum = np.sin(2 * np.pi * 50.0 * t) + 0.35 * np.sin(2 * np.pi * 100.0 * t)
    if rms(hum) == 0.0:
        return samples
    return samples + hum * (rms(samples) / db_to_linear(snr_db) / rms(hum))


def apply_filter(samples: Any, filter_name: str) -> Any:
    from scipy.signal import butter, sosfilt

    if filter_name == "mild_low_pass":
        sos = butter(4, 6200.0, btype="lowpass", fs=16000, output="sos")
    elif filter_name == "mild_high_pass":
        sos = butter(2, 120.0, btype="highpass", fs=16000, output="sos")
    elif filter_name == "band_pass":
        sos = butter(3, [180.0, 6800.0], btype="bandpass", fs=16000, output="sos")
    elif filter_name == "telephone_like":
        sos = butter(4, [300.0, 3400.0], btype="bandpass", fs=16000, output="sos")
    elif filter_name == "broad_eq_tilt":
        sos = butter(2, 2600.0, btype="highpass", fs=16000, output="sos")
    else:
        raise ValueError(f"unsupported filter {filter_name}")
    return sosfilt(sos, samples)


def apply_rir(samples: Any, *, rt60_seconds: float, seed_text: str) -> Any:
    import numpy as np
    from scipy.signal import fftconvolve

    rng = deterministic_rng(seed_text)
    length = max(256, int(16000 * min(rt60_seconds, 0.6)))
    t = np.arange(length, dtype=np.float64) / 16000.0
    decay = np.exp(-6.91 * t / max(rt60_seconds, 0.05))
    impulse = rng.normal(0.0, 0.18, size=length) * decay
    impulse[0] += 1.0
    impulse = impulse / max(peak(impulse), 1e-9)
    return fftconvolve(samples, impulse, mode="full")[: len(samples)]


def codec_simulation(samples: Any, codec: str) -> Any:
    import numpy as np

    pcm = np.rint(np.clip(samples, -0.98, 0.98) * 32767.0).astype("<i2").tobytes()
    if codec == "g711_alaw":
        return np.frombuffer(audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2), dtype="<i2").astype(np.float64) / 32768.0
    if codec == "g711_mulaw":
        return np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2), dtype="<i2").astype(np.float64) / 32768.0
    if codec == "downsample_8k_return_16k":
        return resample_ratio(resample_ratio(samples, 0.5), 2.0)[: len(samples)]
    if codec == "downsample_12k_return_16k":
        return resample_ratio(resample_ratio(samples, 0.75), 4.0 / 3.0)[: len(samples)]
    if codec == "ogg_opus_moderate":
        quantized = np.rint(np.clip(samples, -0.98, 0.98) * 4096.0) / 4096.0
        return apply_filter(quantized, "mild_low_pass")
    raise ValueError(f"unsupported codec {codec}")


def apply_profile_transform(samples: Any, profile_id: str, parameters: dict[str, Any], *, seed_text: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    details: dict[str, Any] = dict(parameters)
    if profile_id == "coupled_speed_pitch_resampling":
        transformed = resample_ratio(samples, float(parameters["rate"]))
    elif profile_id == "tempo_preserving_pitch":
        factor = float(parameters["tempo_factor"])
        if 0.98 < factor < 1.02:
            factor = 1.02 if deterministic_unit_interval(seed_text, profile_id, "tempo-side") >= 0.5 else 0.98
            details["tempo_factor_adjusted"] = factor
        transformed = resample_ratio(samples, 1.0 / factor)
    elif profile_id == "mild_pitch_formant_vtlp_proxy":
        semitone_ratio = math.pow(2.0, float(parameters["pitch_semitones"]) / 12.0)
        pitched = resample_ratio(samples, 1.0 / semitone_ratio)
        transformed = resample_ratio(pitched, len(samples) / max(len(pitched), 1))
        transformed = apply_filter(transformed, "mild_high_pass" if float(parameters["formant_warp"]) >= 1.0 else "mild_low_pass")
    elif profile_id == "procedural_room_impulse_response":
        transformed = apply_rir(samples, rt60_seconds=float(parameters["rt60_seconds"]), seed_text=seed_text)
    elif profile_id == "environmental_background_noise":
        colour = choose_from(("white", "low", "pink"), semantic_key=seed_text, profile_id=profile_id)
        details["noise_colour"] = colour
        transformed = add_noise(samples, snr_db=float(parameters["snr_db"]), seed_text=seed_text, colour=colour)
    elif profile_id == "coloured_electrical_noise":
        colour = choose_from(("white", "pink", "brown", "hum"), semantic_key=seed_text, profile_id=profile_id)
        details["noise_colour"] = colour
        transformed = add_hum(samples, snr_db=float(parameters["snr_db"])) if colour == "hum" else add_noise(samples, snr_db=float(parameters["snr_db"]), seed_text=seed_text, colour=colour)
    elif profile_id == "microphone_channel_filtering":
        transformed = apply_filter(samples, str(parameters["filter"]))
    elif profile_id == "codec_sample_rate_simulation":
        transformed = codec_simulation(samples, str(parameters["codec"]))
    elif profile_id == "gain_dynamic_range_variation":
        gain_db = float(parameters["gain_db"])
        if abs(gain_db) < 1.0:
            gain_db = 1.0 if deterministic_unit_interval(seed_text, profile_id, "gain-side") >= 0.5 else -1.0
            details["gain_db_adjusted"] = gain_db
        transformed = samples * db_to_linear(gain_db)
        before_peak = peak(transformed)
        safety_gain = 1.0
        if before_peak > 0.98:
            safety_gain = 0.98 / before_peak
            transformed = transformed * safety_gain
        details.update(
            {
                "source_rms": round(rms(samples), 8),
                "raw_output_rms": round(rms(transformed), 8),
                "applied_rms_gain": round(db_to_linear(gain_db), 8),
                "peak_safety_gain": round(safety_gain, 8),
                "output_peak_before_write": round(peak(transformed), 8),
            }
        )
        return transformed, details
    elif profile_id == "timing_silence_variation":
        leading = np.zeros(int(round(16000 * float(parameters["leading_silence_seconds"]))), dtype=np.float64)
        trailing = np.zeros(int(round(16000 * float(parameters["trailing_silence_seconds"]))), dtype=np.float64)
        shift = int(round(16000 * float(parameters["time_shift_seconds"])))
        shifted = np.pad(samples, (max(shift, 0), max(-shift, 0)), mode="constant")
        if shift < 0:
            shifted = shifted[-shift:]
        transformed = np.concatenate([leading, shifted, trailing])
    elif profile_id == "compound_realistic_condition":
        transformed = apply_rir(samples, rt60_seconds=0.22 + deterministic_unit_interval(seed_text, "rir") * 0.18, seed_text=seed_text)
        transformed = add_noise(transformed, snr_db=float(parameters["snr_db"]), seed_text=seed_text, colour="low")
        transformed = apply_filter(transformed, choose_from(("mild_low_pass", "mild_high_pass", "broad_eq_tilt"), semantic_key=seed_text, profile_id=profile_id))
    else:
        raise ValueError(f"unsupported augmentation profile {profile_id}")
    transformed, safety = match_source_rms(samples, transformed)
    details.update(safety)
    details["output_peak_before_write"] = round(peak(transformed), 8)
    return transformed, details


def render_augmented_file(
    *,
    source_audio_path: Path,
    output_audio_path: Path,
    profile_id: str,
    parameters: dict[str, Any],
    seed_text: str,
) -> dict[str, Any]:
    source, source_frames = read_mono_pcm16(source_audio_path)
    transformed, details = apply_profile_transform(source, profile_id, parameters, seed_text=seed_text)
    write_mono_pcm16(output_audio_path, transformed)
    stats = read_audio_stats(output_audio_path)
    details.update(
        {
            "source_frame_count": source_frames,
            "output_frame_count": stats.frames,
            "source_duration_seconds": round(source_frames / 16000.0, 6),
            "output_duration_seconds": round(stats.duration_seconds, 6),
            "output_audio_sha256": stats.sha256,
        }
    )
    return details
