# Experiment 0015: Scale-8000 Clean-Only RTX 2080 Ti Directional

Classification: `SCALE8000_CLEAN_BEATS_BASE_BUT_NOT_SCALE2000`

This is directional, noncanonical batch-32 evidence. No batch-1 canonical evaluation was run, no checkpoint or adapter is accepted, and `accepted_parent` remains `none`.

## Data

- Corpus: `sl-corpus-v5-scale8000-training-v1`
- Text SHA256: `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd`
- Semantic rows: 64000
- Clean views: 576000
- Augmented views: 0

## Training

- Arm: `scale8000_clean_only_joint_adapter_dim32`
- Physical microbatch: 1
- Gradient accumulation: 8
- Effective batch size: 8
- Optimizer steps: 72000
- Trainable parameters: 42240

## Directional Metrics

| Split | Base WER/CER | Scale-2000 WER/CER | Scale-8000 clean WER/CER | Empty base/scale2000/scale8000 |
|---|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025/46.762 | 55.435/20.073 | 62.267/21.979 | 17/0/0 |
| supertonic_heldout_voice_holdout | 58.307/27.712 | 27.407/7.597 | 28.882/8.004 | 32/0/0 |
| fleurs_v2 | 52.685/16.406 | 51.589/16.238 | 51.268/16.259 | 1/0/0 |
| artur_j | 67.322/28.62 | 60.114/20.63 | 60.856/20.708 | 12/0/0 |

## Decision

- Real-regression burden: 0.0
- Accepted parent: `none`

## Limitations

- Synthetic-only training remains a diagnostic signal.
- Directional batch-32 metrics cannot promote a model.
- Real speech remains validation-only and decisive for acceptance.
