# Experiment 0023: Scale-2000 Decoder+Joint RNNT with S6TTS Hard-Voice Share

Classification: `S6TTS_HARDVOICE_REAL_REGRESSION`

This is synthetic-only, directional batch-32 evidence. The acoustic encoder and Slovenian prompt pathway were frozen; decoder and joint base parameters were intentionally trainable. No checkpoint is accepted and `accepted_parent` remains `none`.

## Data

- Corpus: `sl-corpus-v4-gams-16000-training-v1`
- Fixed text SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- Base scale-2000 all views SHA256: `9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca`
- Base scale-2000 schedule SHA256: `6757018f3306839ce8564ba758e13e231ab4784bf98049b65701b963b55e5842`
- S6 clean manifest SHA256: `355a85134e81d9e3ea4089ea9a941f62fb101902b4e151c394eaaf1d1de416d5`
- S6 augmented manifest SHA256: `8d39606dc276a7730e032e83c1811f6c71ece3de6f0b68aa1bd5f4c0a8f50251`
- Hardvoice schedule SHA256: `07c30ea48dd1a155aef7bb42ab804367bf53cddd6e41448db7485b475aa170b6`
- Exposures: 320000
- S6TTS exposure share: 0.2
- S6 hard-voice holdout rows: 1152

## Training

- Arm: `scale2000_s6tts_hardvoice20_decoder_joint_rnnt`
- Physical microbatch: 1
- Gradient accumulation: 8
- Effective batch size: 8
- Optimizer steps: 40000
- Max optimizer steps: 40000
- Stopped round: 20
- Stop reason: `max_rounds_completed`
- Controller-dev selected round: 14
- Trainable parameters: 24395808
- Continuation: retained model checkpoints were restored, but AdamW state was not retained; optimizer moments reset at continuation boundaries.
- Runtime scope: wall time, throughput, and exposure-count breakdowns cover the final resumed segment only.

## Training And Controller-Dev Curve

| Round | Step | Train loss | Anchor probe | Scale probe | S6 clean probe | S6 augmented probe | ARTUR-dev WER | CER | Empty | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | - | 53.649231 | 53.695999 | 60.301889 | 68.477138 | 66.467 | 27.409 | 13 | no |
| 1 | 2000 | 17.207915 | 10.406321 | 13.928462 | 21.375876 | 30.473003 | 61.085 | 23.302 | 0 | no |
| 2 | 4000 | 6.692708 | 14.012253 | 18.057565 | 26.478918 | 34.098386 | 57.668 | 22.41 | 0 | no |
| 3 | 6000 | 5.777554 | 16.632252 | 20.181915 | 27.801911 | 38.118576 | 55.745 | 21.67 | 0 | no |
| 4 | 8000 | 8.157559 | 19.001171 | 20.438686 | 22.937418 | 33.783034 | 57.027 | 22.069 | 0 | no |
| 5 | 10000 | 11.314583 | 21.036852 | 20.131347 | 9.037671 | 15.05255 | 64.331 | 28.011 | 0 | no |
| 6 | 12000 | 5.427631 | 16.051582 | 18.826655 | 14.123784 | 21.843517 | 55.959 | 21.358 | 0 | no |
| 7 | 14000 | 5.93494 | 14.751828 | 21.235803 | 19.398124 | 27.632172 | 57.54 | 22.388 | 0 | no |
| 8 | 16000 | 5.444359 | 16.851776 | 21.138084 | 22.591514 | 28.066696 | 57.24 | 22.932 | 0 | no |
| 9 | 18000 | 6.639511 | 15.954592 | 19.301472 | 22.062935 | 28.968355 | 57.07 | 21.924 | 0 | no |
| 10 | 20000 | 14.943602 | 15.288533 | 19.136929 | 8.777211 | 13.416785 | 61.598 | 26.582 | 0 | no |
| 11 | 22000 | 5.322862 | 13.869375 | 16.230389 | 13.798554 | 20.227083 | 55.446 | 20.451 | 0 | no |
| 12 | 24000 | 4.759125 | 11.527347 | 15.462071 | 15.21646 | 21.800088 | 54.677 | 21.351 | 0 | no |
| 13 | 26000 | 9.071306 | 11.556131 | 14.145682 | 15.777075 | 22.28308 | 55.617 | 21.445 | 0 | no |
| 14 | 28000 | 5.182073 | 14.930156 | 15.10753 | 16.830494 | 22.772025 | 54.165 | 20.067 | 0 | yes |
| 15 | 30000 | 15.357611 | 13.340416 | 17.826569 | 7.019772 | 12.730616 | 60.914 | 25.109 | 0 | no |
| 16 | 32000 | 6.077335 | 9.291644 | 14.441396 | 11.723668 | 17.995381 | 56.044 | 22.25 | 0 | no |
| 17 | 34000 | 5.164541 | 13.133378 | 15.058264 | 15.420893 | 20.133342 | 55.105 | 21.51 | 0 | no |
| 18 | 36000 | 6.386782 | 10.444089 | 13.515155 | 15.281622 | 20.773801 | 54.677 | 21.264 | 0 | no |
| 19 | 38000 | 5.19853 | 10.951885 | 14.958234 | 17.053322 | 21.672685 | 55.617 | 21.692 | 0 | no |
| 20 | 40000 | 14.700854 | 14.410678 | 17.30485 | 5.751143 | 11.499822 | 60.402 | 25.239 | 0 | no |

## S6 Hard-Voice Metrics

| Split | Base WER/CER | PR #36 if available | Selected round 14 WER/CER | Empty base/pr36/selected |
|---|---:|---:|---:|---:|
| s6tts_clean_holdout | 98.447/85.618 | not run | 52.484/24.923 | 65/not run/0 |
| s6tts_augmented_holdout | 96.443/77.501 | not run | 57.933/27.716 | 589/not run/0 |

## Directional Metrics

| Split | Base WER/CER | Scale-2000 joint WER/CER | PR #36 round20 WER/CER | Selected round 14 WER/CER | Empty base/scale2000/pr36/selected |
|---|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025/46.762 | 55.435/20.073 | 34.317/13.765 | 38.043/16.008 | 17/0/0/0 |
| supertonic_heldout_voice_holdout | 58.307/27.712 | 27.407/7.597 | 14.752/4.682 | 17.081/6.939 | 32/0/0/0 |
| fleurs_v2 | 52.685/16.406 | 51.589/16.238 | 46.195/15.604 | 48.162/16.725 | 1/0/0/0 |
| artur_j | 67.322/28.62 | 60.114/20.63 | 56.793/20.177 | 54.871/20.318 | 12/0/0/0 |

## Decision

- Real-regression burden: 0.319
- Strict regression trigger: FLEURS-v2 CER is 0.319 absolute points above untouched base despite WER gains on both real gates and WER/CER gains on ARTUR-J.
- Accepted parent: `none`

## Limitations

- Synthetic-only training remains diagnostic.
- Directional batch-32 metrics cannot promote a checkpoint.
- Real speech remains validation-only and decisive for acceptance.
- Training resumed from retained model checkpoints without retained AdamW state, so optimizer moments reset at continuation boundaries.
