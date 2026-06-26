# Experiment 0013: GaMS 1600 Nine-voice Scale-200 Directional Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic uses 1,600 accepted synthetic texts, nine clean synthetic voice sources, and eleven transcript-preserving augmentation views. Evaluation is fast batch-32 directional evidence only; no canonical batch-1 evaluation was run.

## Scale

- Semantic rows: 1600
- Clean files: 14400
- Augmented files: 17600
- Exposure records: 32000
- The 200x multiplier refers to exposure count, not independent linguistic information.

## Directional Metrics

| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |
|---|---:|---:|---:|---:|
| base | 86.025/46.762 | 58.307/27.712 | 52.685/16.406 | 67.322/28.62 |
| piper_joint_adapter | 69.876/28.007 | 50.543/15.041 | 64.733/25.529 | 72.958/30.021 |
| supertonic3_joint_adapter | 75.932/30.25 | 46.817/13.849 | 60.616/20.973 | 70.511/26.723 |
| batched_replay_joint_adapter | 73.137/28.091 | 48.292/13.821 | 58.643/19.023 | 68.283/24.926 |
| scale200_joint_adapter | 63.509/23.83 | 36.879/11.088 | 54.386/17.573 | 64.176/22.753 |

## Decision

- Regression burden: 2.868
- Burden reduction versus replay reference: 69.924497%
- Classification: `SCALE200_SYNTHETIC_MITIGATES_REPLAY_REGRESSION`
- Accepted parent: `none`

## Limitations

- Directional batch-32 metrics are not canonical acceptance evidence.
- No batch-1 evaluation was run.
- All training remains synthetic.
- The 200x multiplier refers to exposure count, not independent linguistic information.
- No checkpoint or adapter is accepted as a parent.
- TRAINING_ELIGIBLE was not issued.
