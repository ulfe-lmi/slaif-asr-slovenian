# Experiment 0032: Surface08 Scale-8000 OTF StrongAug v1

Classification: `STRONGAUG_V1_REAL_GATE_REGRESSION`

This diagnostic keeps PR #51's model surface, untouched-base initialization, scale-8000 clean reservoir, optimizer, effective batch, controller policy, and 144,000-exposure cap fixed. The only scientific change is deterministic StrongAug v1 intensity and curriculum. All transformed waveforms were produced in memory.

## Result

- Selected round: 6; stopped round: 9 (`three_rounds_without_new_raw_best`).
- Training exposures: 144,000; unique semantic rows: 64,000.
- Aggregate OTF fill rate: 0.996849.
- Selected checkpoint SHA256: `c5c95ffd83be32353f5321a6c7968b435c4ae88b637bef190a11936169cc5e18`.
- `accepted_parent` remains `none`; this is noncanonical diagnostic evidence.

## Augmentation Audit

| Augmentation family | Fraction | Parameter range | Notes |
|---|---:|---|---|
| Standard OTF | 0.450306 | existing 11 profiles | one standard profile per selected sample |
| Additive noise | 0.313701 | SNR 8-25 dB; usually 10-25 dB | white, pink, brown, hum, fan/HVAC procedural noise |
| Reverb | 0.235146 | RT60 0.12-0.45 s | deterministic short-to-moderate synthetic room response |
| Bandpass/filter | 0.315375 | mild low/high/bandpass/tilt; 8% telephone candidate | lower-SNR plus telephone is prohibited |
| Codec/companding | 0.078722 | G.711 mu-law/A-law, 12 kHz round-trip, moderate quantization | deterministic in memory |
| Gain/compression | 0.235146 | gain -6 to +6 dB; ratio 1.5-3.0 | mild compression/limiting; rare soft clipping |
| Speed/pitch | 0.078944 | speed 0.94-1.06 | no large pitch shift |

| Round range | Standard OTF % | StrongAug % | Purpose |
|---|---:|---:|---|
| 1-4 | 70 | 30 | broad linguistic coverage with mild robustness |
| 5-9 | 25 | 75 | robustness pressure on repeated rows |

Observed deterministic mixture:

- Coverage phase: `{"samples": 64000, "standard_fraction": 0.699031, "standard_samples": 44738, "strong_fraction": 0.300969, "strong_samples": 19262}`
- Robustness phase: `{"samples": 80000, "standard_fraction": 0.251325, "standard_samples": 20106, "strong_fraction": 0.748675, "strong_samples": 59894}`

## Controller-Dev Curve

| Round | Step | Exposures | Unique semantic rows | Train loss | Anchor | Scale | ARTUR-dev WER | CER | Empty | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | NOT_APPLICABLE | 48.291606 | 47.956826 | 66.467 | 27.409 | 13 | false |
| 1 | 2000 | 16000 | 16000 | 6.827323 | 5.014875 | 6.904157 | 42.717 | 13.806 | 0 | false |
| 2 | 4000 | 32000 | 32000 | 9.646206 | 3.816313 | 5.51031 | 41.563 | 14.038 | 0 | false |
| 3 | 6000 | 48000 | 48000 | 8.238914 | 3.317234 | 5.065149 | 40.239 | 14.763 | 0 | false |
| 4 | 8000 | 64000 | 64000 | 7.595151 | 3.086005 | 4.625936 | 42.204 | 15.329 | 0 | false |
| 5 | 10000 | 80000 | 64000 | 7.266124 | 2.395652 | 4.311823 | 39.812 | 14.379 | 0 | false |
| 6 | 12000 | 96000 | 64000 | 6.924031 | 2.248785 | 3.830563 | 38.915 | 13.646 | 0 | true |
| 7 | 14000 | 112000 | 64000 | 6.711497 | 2.652643 | 4.365921 | 41.008 | 15.163 | 0 | false |
| 8 | 16000 | 128000 | 64000 | 6.47859 | 2.787658 | 3.986893 | 39.556 | 14.096 | 0 | false |
| 9 | 18000 | 144000 | 64000 | 6.228589 | 2.489959 | 3.870898 | 39.214 | 13.61 | 0 | true |

ARTUR controller-dev aggregate metrics alone selected the checkpoint. FLEURS-v2 and ARTUR-J were evaluated only after selection.

## Live OTF Loader Telemetry

| Round | Examples/s | OTF fill rate | Consumer wait % | Queue p50/p95 | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 13.520708 | 0.997991856 | 0.200814 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 2 | 12.680959 | 0.997139742 | 0.286026 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 3 | 13.401514 | 0.995046484 | 0.495352 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 4 | 13.193375 | 0.997327553 | 0.267245 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 5 | 12.967653 | 0.996008328 | 0.399167 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 6 | 13.2075 | 0.996924543 | 0.307546 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 7 | 13.335479 | 0.996749662 | 0.325034 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 8 | 13.967576 | 0.996898908 | 0.310109 | 15.0 / 15.0 | 3 spawned workers; in-memory only |
| 9 | 12.779732 | 0.997536946 | 0.246305 | 15.0 / 15.0 | 3 spawned workers; in-memory only |

## Directional Metrics

| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Surface08 scale-8000 standard OTF | Surface08 scale-8000 StrongAug v1 |
|---|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 34.317 / 13.765 / 0 | 23.137 / 7.429 / 0 | 20.497 / 6.112 / 0 | 13.199 / 3.785 / 0 | 11.957 / 3.645 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 14.752 / 4.682 / 0 | 7.842 / 2.145 / 0 | 5.202 / 1.850 / 0 | 4.658 / 1.472 / 0 | 5.202 / 1.850 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 46.195 / 15.604 / 0 | 42.084 / 12.985 / 0 | 41.878 / 13.186 / 0 | 39.353 / 12.002 / 0 | 39.904 / 12.343 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 56.793 / 20.177 / 0 | 47.357 / 14.805 / 0 | 41.765 / 13.553 / 0 | 39.974 / 12.251 / 0 | 39.406 / 12.357 / 0 |

Values are normalized WER / CER / empty hypotheses.

## Boundaries

- No real speech, S6TTS, scale-2000 audio, or immutable gate was used for training.
- No augmented WAV was pre-rendered, written, or used.
- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.
- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.
