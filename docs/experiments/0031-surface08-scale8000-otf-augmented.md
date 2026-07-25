# Experiment 0031: Surface08 Scale-8000 OTF Augmented

Classification: `SCALE8000_OTF_SURFACE08_NEW_BEST_DIRECTIONAL`

This stacked diagnostic changes the training data axis while retaining the Surface08 model surface. It uses scale-8000 clean synthetic audio and deterministic in-memory transcript-preserving augmentation from PR #50 infrastructure. No augmented WAV was rendered or used.

## Result

- Selected round: 5; stopped round: 9 (`three_rounds_without_new_raw_best`).
- Training exposures: 144,000; unique semantic rows: 64,000.
- OTF workers: 3 spawned processes; aggregate fill rate: 0.994036.
- Selected checkpoint SHA256: `106b114d4f975b939b34027aa403810db81ca777d1f0568ba039da0744dae46a`.
- `accepted_parent` remains `none`; this result is noncanonical and promotion-ineligible.

## Controller-Dev Curve

| Round | Step | Exposures | Unique semantic rows | Train loss | Anchor | Scale | ARTUR-dev WER | CER | Empty | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | NOT_APPLICABLE | 48.291606 | 47.956826 | 66.467 | 27.409 | 13 | false |
| 1 | 2000 | 16000 | 16000 | 12.158212 | 7.640984 | 7.803065 | 44.297 | 15.235 | 0 | false |
| 2 | 4000 | 32000 | 32000 | 8.648028 | 6.454911 | 6.343108 | 40.752 | 14.27 | 0 | false |
| 3 | 6000 | 48000 | 48000 | 7.719345 | 5.005504 | 5.501023 | 42.631 | 14.858 | 0 | false |
| 4 | 8000 | 64000 | 64000 | 7.228627 | 5.323382 | 5.420541 | 42.631 | 15.678 | 0 | false |
| 5 | 10000 | 80000 | 64000 | 6.729259 | 3.89556 | 4.699539 | 39.0 | 13.61 | 0 | true |
| 6 | 12000 | 96000 | 64000 | 6.325313 | 4.290958 | 5.042308 | 38.616 | 13.363 | 0 | true |
| 7 | 14000 | 112000 | 64000 | 6.168114 | 4.445369 | 5.64919 | 39.94 | 15.097 | 0 | false |
| 8 | 16000 | 128000 | 64000 | 6.017143 | 3.078338 | 4.256905 | 39.94 | 13.944 | 0 | false |
| 9 | 18000 | 144000 | 64000 | 5.818158 | 4.445348 | 5.163853 | 39.556 | 14.357 | 0 | false |

ARTUR controller-dev aggregate metrics alone selected the checkpoint. Anchor and scale are inherited fixed synthetic probes and were not training sources.

## Data Diversity

| Metric | Value |
|---|---:|
| Total virtual exposures seen | 144000 |
| Unique semantic rows seen | 64000 |
| Mean exposures per semantic row | 2.25 |
| P95 exposures per semantic row | 3 |
| Voice/style distribution | `{"F1": 16000, "F2": 16000, "F3": 16000, "F4": 16000, "M1": 16000, "M2": 16000, "M3": 16000, "M4": 16000, "piper-artur": 16000}` |
| Augmentation profile distribution | `{"codec_sample_rate_simulation": 13025, "coloured_electrical_noise": 12929, "compound_realistic_condition": 13218, "coupled_speed_pitch_resampling": 13068, "environmental_background_noise": 13042, "gain_dynamic_range_variation": 13277, "microphone_channel_filtering": 13194, "mild_pitch_formant_vtlp_proxy": 13058, "procedural_room_impulse_response": 13104, "tempo_preserving_pitch": 12919, "timing_silence_variation": 13166}` |

## Live OTF Loader Telemetry

| Round | Examples/s | OTF fill rate | Consumer wait % | Queue p50/p95 | Worker count | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 13.307672 | 0.995457995 | 0.4542 | 15.0 / 15.0 | 3 | in-memory only |
| 2 | 13.09185 | 0.992505593 | 0.749441 | 15.0 / 15.0 | 3 | in-memory only |
| 3 | 13.278953 | 0.993428916 | 0.657108 | 15.0 / 15.0 | 3 | in-memory only |
| 4 | 13.52551 | 0.993057476 | 0.694252 | 15.0 / 15.0 | 3 | in-memory only |
| 5 | 12.593067 | 0.992757767 | 0.724223 | 15.0 / 15.0 | 3 | in-memory only |
| 6 | 13.706704 | 0.988785107 | 1.121489 | 15.0 / 15.0 | 3 | in-memory only |
| 7 | 13.342608 | 0.995865859 | 0.413414 | 15.0 / 15.0 | 3 | in-memory only |
| 8 | 12.461126 | 0.997213804 | 0.27862 | 15.0 / 15.0 | 3 | in-memory only |
| 9 | 13.233884 | 0.996953988 | 0.304601 | 15.0 / 15.0 | 3 | in-memory only |

## Directional Metrics

| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Surface08 scale-8000 OTF |
|---|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 34.317 / 13.765 / 0 | 23.137 / 7.429 / 0 | 20.497 / 6.112 / 0 | 13.199 / 3.785 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 14.752 / 4.682 / 0 | 7.842 / 2.145 / 0 | 5.202 / 1.850 / 0 | 4.658 / 1.472 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 46.195 / 15.604 / 0 | 42.084 / 12.985 / 0 | 41.878 / 13.186 / 0 | 39.353 / 12.002 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 56.793 / 20.177 / 0 | 47.357 / 14.805 / 0 | 41.765 / 13.553 / 0 | 39.974 / 12.251 / 0 |

Values are normalized WER / CER / empty hypotheses. Directional batch-32 evaluation ran only after ARTUR-dev fixed the selected round.

## Boundaries

- No real speech, S6TTS, scale-2000 audio, or immutable gate was used for training.
- No pre-rendered augmented WAV was used or written.
- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.
- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.
- The local scale-8000 manifests are restored identities; the historical committed manifest hashes remain separately recorded.
