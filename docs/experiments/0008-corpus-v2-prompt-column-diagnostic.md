# Experiment 0008: Corpus-v2 Prompt-column Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic trains only the 2,048-value Slovenian prompt-column delta on the corpus-v2 selected synthetic training manifest. The data status is `DIAGNOSTIC_ONLY`; no checkpoint is accepted as a parent.

## Authorization

- Certificate status: `DIAGNOSTIC_ONLY`
- Certificate SHA256: `ac201abe6ea59624fa386cde4c48e2dbaaa2e6eba21a3af0ddbe42f17a488e8c`
- Selected-training manifest SHA256: `84e10587af184be92571ab84e3bd58cd676866e2bd944534c759f0fc9a07fa13`

## Decisions

- Scientific classification: `CORPUS_V2_PROMPT_COLUMN_SYNTHETIC_ONLY`
- Batching classification: `A100_PROMPT_TRAINING_BATCH_NOT_EQUIVALENT`
- Accepted parent: `none`

## Training

| Arm | Batch | Epochs | Exposures | Initial probe loss | Final probe loss | Full-loss reduction |
|---|---:|---:|---:|---:|---:|---:|
| reference_batch1 | 1 | 12 | 1920 | 50.777602 | 27.852969 | 43.87274 |
| a100_batched | 8 | 12 | 1920 | 50.777602 | 22.44438 | 54.22353 |

## Aggregate Metrics

| Model | Split | Normalized WER | Normalized CER | Empty hypotheses |
|---|---|---:|---:|---:|
| a100_batched | artur_j | 71.123 | 25.796 | 0 |
| a100_batched | fleurs_v2 | 61.47 | 20.347 | 0 |
| a100_batched | selected_training | 69.955 | 26.405 | 0 |
| a100_batched | synthetic_holdout | 73.137 | 27.474 | 2 |
| base | artur_j | 67.453 | 29.016 | 12 |
| base | fleurs_v2 | 52.703 | 16.423 | 1 |
| base | selected_training | 93.032 | 61.623 | 41 |
| base | synthetic_holdout | 84.317 | 47.295 | 17 |
| reference_batch1 | artur_j | 74.443 | 28.655 | 1 |
| reference_batch1 | fleurs_v2 | 66.229 | 23.729 | 0 |
| reference_batch1 | selected_training | 78.19 | 31.153 | 1 |
| reference_batch1 | synthetic_holdout | 78.882 | 32.941 | 3 |

## Limitations

- Single-voice synthetic training.
- No real training or calibration speech.
- Synthetic holdout metrics are diagnostic only.
- Development real gates are not a final blind test.
