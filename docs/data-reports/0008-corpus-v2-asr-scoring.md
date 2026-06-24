# Corpus-v2 ASR Scoring

Status: `PASSED`

This privacy-safe report records untouched-base ASR scoring for the accepted single-voice synthetic candidate source and independent synthetic holdout. It contains aggregate metrics only and does not authorize model training.

## Runtime

- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Context: `[56, 3]`
- Batch policy: batch size 1, no duration bucketing, FP32, TF32 disabled

## Aggregate Metrics

| Partition | Rows | Normalized WER | Normalized CER | Raw WER | Raw CER | Empty hypotheses | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_source | 415 | 76.114 | 36.182 | 77.439 | 37.502 | 41 | 0.150185 |
| synthetic_holdout | 96 | 84.317 | 47.295 | 85.67 | 48.956 | 17 | 0.36453 |

## Limitations

- This is untouched-base ASR scoring of single-voice synthetic audio.
- Synthetic holdout metrics are diagnostic only and are not real-speech generalization evidence.
- No selected-training data is TRAINING_ELIGIBLE in this report.
- Raw generated text, hypotheses, candidate IDs, audio paths, local manifests, and monitor CSV files remain ignored local artifacts.
