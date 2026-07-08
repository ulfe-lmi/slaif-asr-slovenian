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
