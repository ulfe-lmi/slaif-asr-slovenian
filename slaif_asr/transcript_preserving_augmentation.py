from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

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
