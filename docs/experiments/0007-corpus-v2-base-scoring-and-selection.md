# Experiment 0007: Corpus-v2 Base Scoring and Selection

Status: **completed in PR; pending strategic review**

This experiment scores the accepted single-voice synthetic candidate source and independent synthetic holdout with the untouched Nemotron base, then constructs a selected-training manifest from the candidate source only. No model training occurred.

## Model

- Repository: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`

## Scoring Metrics

| Partition | Rows | Normalized WER | Normalized CER | Empty hypotheses |
|---|---:|---:|---:|---:|
| candidate source | 415 | 76.114 | 36.182 | 41 |
| synthetic holdout | 96 | 84.317 | 47.295 | 17 |

## Selected Training

- Rows: 160
- Hard/control: 120 / 40
- Certificate status: `SELECTED_TRAINING_MANIFEST_READY`

## Limitations

- Selected training is single-voice synthetic Piper audio.
- Selected training is not real-speech evidence.
- Training remains unauthorized until a later training work order.
- The untouched Nemotron base remains the only accepted parent.
