# Experiment 0017: Scale-2000 Decoder+Joint RNNT Directional Diagnostic

Classification: `DECODER_JOINT_RNNT_BEATS_SCALE2000_DIRECTIONAL`

This is synthetic-only, directional batch-32 evidence. The acoustic encoder and Slovenian prompt pathway were frozen; decoder and joint base parameters were intentionally trainable. No checkpoint is accepted and `accepted_parent` remains `none`.

## Data

- Corpus: `sl-corpus-v4-gams-16000-training-v1`
- Fixed text SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- All views SHA256: `9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca`
- Exposure schedule SHA256: `6757018f3306839ce8564ba758e13e231ab4784bf98049b65701b963b55e5842`
- Exposures: 320000

## Training

- Arm: `scale2000_augmented_decoder_joint_rnnt`
- Physical microbatch: 1
- Gradient accumulation: 8
- Effective batch size: 8
- Optimizer steps: 40000
- Trainable parameters: 24395808

## Loss History

All losses below are RNNT losses where lower is better. All probe rows use
synthetic scale-2000 training data only; FLEURS-v2 and ARTUR-J were not used for
training, probing, early stopping, or steering.

Column meanings:

- `Round`: exposure round, equivalent to the training epoch index for this
  schedule. Round 0 is the pre-training probe before optimizer updates.
- `Train sampled mean`: mean of privacy-safe sampled current training-loss
  events emitted by the live progress reporter during that round. This is not a
  full-corpus training loss.
- `Train last rolling`: final rolling mean training loss reported near the end
  of that round by the live progress reporter.
- `Anchor probe`: no-optimizer loss on the fixed 32-row inherited synthetic
  anchor probe after the round.
- `Scale probe`: no-optimizer loss on the fixed 320-row deterministic synthetic
  scale probe after the round.

| Round | Train sampled mean | Train last rolling | Anchor probe | Scale probe |
|---:|---:|---:|---:|---:|
| 0 | - | - | 56.371 | 53.588 |
| 1 | 11.783 | 17.054 | 12.763 | 14.512 |
| 2 | 4.435 | 6.454 | 17.815 | 18.726 |
| 3 | 3.355 | 6.034 | 15.380 | 20.172 |
| 4 | 4.761 | 7.209 | 21.448 | 19.017 |
| 5 | 4.381 | 6.016 | 15.324 | 19.839 |
| 6 | 2.870 | 4.870 | 15.459 | 20.397 |
| 7 | 3.548 | 6.056 | 15.295 | 22.007 |
| 8 | 3.384 | 5.242 | 20.547 | 20.269 |
| 9 | 3.915 | 6.639 | 17.913 | 19.746 |
| 10 | 3.952 | 6.921 | 13.529 | 15.139 |
| 11 | 4.440 | 5.592 | 14.796 | 16.676 |
| 12 | 3.867 | 4.729 | 15.027 | 14.984 |
| 13 | 4.814 | 9.219 | 13.097 | 15.445 |
| 14 | 3.411 | 5.040 | 12.481 | 14.298 |
| 15 | 3.800 | 7.294 | 11.193 | 14.967 |
| 16 | 3.139 | 5.117 | 12.745 | 14.601 |
| 17 | 3.700 | 5.983 | 12.291 | 15.980 |
| 18 | 3.228 | 5.731 | 11.888 | 14.485 |
| 19 | 2.886 | 5.894 | 11.448 | 14.807 |
| 20 | 4.431 | 7.069 | 11.250 | 12.897 |

The scalar final probe fields in the JSON report were recorded by a separate
final probe pass: anchor `12.483167`, scale `13.510591`. The table above is the
per-round probe history.

## Directional Metrics

| Split | Base WER/CER | Scale-2000 joint WER/CER | Decoder+Joint RNNT WER/CER | Empty base/scale2000/decoder+joint |
|---|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025/46.762 | 55.435/20.073 | 34.317/13.765 | 17/0/0 |
| supertonic_heldout_voice_holdout | 58.307/27.712 | 27.407/7.597 | 14.752/4.682 | 32/0/0 |
| fleurs_v2 | 52.685/16.406 | 51.589/16.238 | 46.195/15.604 | 1/0/0 |
| artur_j | 67.322/28.62 | 60.114/20.63 | 56.793/20.177 | 12/0/0 |

## Decision

- Real-regression burden: 0.0
- Accepted parent: `none`

## Limitations

- Synthetic-only training remains diagnostic.
- Directional batch-32 metrics cannot promote a checkpoint.
- Real speech remains validation-only and decisive for acceptance.
