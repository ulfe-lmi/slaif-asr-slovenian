# Experiment 0014: GaMS 16000 Scale-2000 Text-Only Directional Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic changes semantic text count from 1,600 to 16,000 while preserving the scale-200 voices, augmentation policy, joint-adapter surface, training protocol, and batch-32 directional evaluation policy. No canonical batch-1 evaluation was run.

## Scale

- Semantic rows: 16000
- Clean files: 144000
- Augmented files: 176000
- Exposure records: 320000
- The 2000x multiplier refers to exposure count, not independent linguistic information.

## Directional Metrics

| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |
|---|---:|---:|---:|---:|
| base | 86.025/46.762 | 58.307/27.712 | 52.685/16.406 | 67.322/28.62 |
| scale200_joint_adapter | 63.509/23.83 | 36.879/11.088 | 54.386/17.573 | 64.176/22.753 |
| scale2000_joint_adapter | 55.435/20.073 | 27.407/7.597 | 51.589/16.238 | 60.114/20.63 |

## Decision

- Scale-200 burden: 2.868
- Scale-2000 burden: 0.0
- Burden change: -2.868
- Classification: `SCALE2000_TEXT_REAL_GAIN_DIRECTIONAL`
- Accepted parent: `none`

## Limitations

- Directional batch-32 metrics are not canonical acceptance evidence.
- No batch-1 evaluation was run.
- All training remains synthetic.
- The 2000x multiplier refers to exposure count, not independent linguistic information.
- No checkpoint or adapter is accepted as a parent.
- TRAINING_ELIGIBLE was not issued.
