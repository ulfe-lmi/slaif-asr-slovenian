# Experiment 0009: Corpus-v2 Speaker-range Augmentation Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic changes only deterministic speaker-range resampling of the selected-training waveforms relative to Experiment 0008's clean `a100_batched` arm. The data status remains `DIAGNOSTIC_ONLY`; no checkpoint is accepted as a parent.

## Authorization

- Certificate status: `DIAGNOSTIC_ONLY`
- Certificate SHA256: `b9163264c5bd48fa877d5cf799255db410b1a978582f3b70fd5dfef72f0e0bc8`
- Baseline report SHA256: `117ec8bbb97580db3e9ccf13a118a8472aa06930f42417171046e487e8ba411a`

## Augmentation

- Source rows: 160
- Non-clean generated files: 640
- Scheduled exposures: 1920

## Aggregate Metrics

| Split | Base WER/CER | Clean batch-8 WER/CER | Augmented WER/CER | Empty base/clean/augmented |
|---|---:|---:|---:|---:|
| selected_training | 93.032/61.623 | 69.955/26.405 | 72.76/28.104 | 41/0/1 |
| synthetic_holdout | 84.317/47.295 | 73.137/27.474 | 72.981/28.708 | 17/2/2 |
| fleurs_v2 | 52.703/16.423 | 61.47/20.347 | 61.93/20.661 | 1/0/0 |
| artur_j | 67.453/29.016 | 71.123/25.796 | 71.56/24.749 | 12/0/0 |

## Decision

- Scientific classification: `SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED`
- Accepted parent: `none`
- Clean real-regression burden: 16.361
- Augmented real-regression burden: 17.572

## Limitations

- One original Piper voice family.
- Resampling is only an acoustic proxy and does not establish age, gender, or multi-speaker coverage.
- No real calibration speech.
- Synthetic holdout is not real-generalization evidence.
- FLEURS-v2 and ARTUR-J are development gates.
