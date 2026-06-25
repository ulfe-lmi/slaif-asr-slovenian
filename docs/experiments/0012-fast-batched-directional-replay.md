# Experiment 0012: Fast Batched Directional Replay

Status: **completed in PR; pending strategic review**

This replay uses native batched Supertonic synthesis and batch-32 directional ASR evaluation. It is noncanonical: exact batch-1 transcript parity was intentionally not required, no batch-1 replay was run, and no release or promotion decision may use this report alone.

## Synthesis

- Batch size: 32
- Batch count: 46
- Converted rows: 1472
- OOM fallbacks: 0

## Directional Metrics

| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |
|---|---:|---:|---:|---:|
| base | 86.025/46.762 | 58.307/27.712 | 52.685/16.406 | 67.322/28.62 |
| piper_joint_adapter | 69.876/28.007 | 50.543/15.041 | 64.733/25.529 | 72.958/30.021 |
| supertonic3_joint_adapter | 75.932/30.25 | 46.817/13.849 | 60.616/20.973 | 70.511/26.723 |
| batched_replay_joint_adapter | 73.137/28.091 | 48.292/13.821 | 58.643/19.023 | 68.283/24.926 |

## Decision

- Piper burden: 28.208
- Canonical Supertonic burden: 15.687
- Replay Supertonic burden: 9.536
- Classification: `FAST_DIRECTIONAL_REPLAY_CONFIRMS_CONCLUSION`
- Accepted parent: `none`

## Limitations

- Directional batch-32 metrics are not canonical acceptance evidence.
- Exact transcript parity with batch size 1 was intentionally not required.
- No batch-1 replay was run.
- No release or promotion decision may use this report alone.
- All training remains synthetic and accepted_parent remains none.
