# Corpus-v2 Scoring Authorization

Status: `SCORING_AUTHORIZED`

This privacy-safe report authorizes ASR scoring and selected-training construction only. It does not authorize model training or `TRAINING_ELIGIBLE` status.

## Inputs

- Candidate source rows: 415
- Candidate source text SHA256: `b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6`
- Candidate source audio manifest SHA256: `c1d366e1d05b6f728af51b3350556b6d915fabf5a6b584a6aa2f9fdc0df538bc`
- Synthetic holdout rows: 96
- Synthetic holdout text SHA256: `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`
- Synthetic holdout audio manifest SHA256: `7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d`

## Partition Independence

- Text overlap counts: `{}`
- Audio overlap counts: `{"audio_path_overlaps": 0, "audio_sha256_overlaps": 0}`
- Fuzzy review pairs: 0
- Protected overlap counts: `{"number_masked_overlaps": 0, "surface_overlaps": 0}`

## Authorized Next Actions

- ASR scoring of the accepted candidate source
- ASR scoring of the accepted synthetic holdout
- selected-training construction from the accepted candidate source
- diversity-preserving hard-example selection in a later work order

## Prohibited Actions

- model training
- TRAINING_ELIGIBLE certification
- checkpoint promotion
- public performance claims
- use as real-speech generalization evidence

## Limitations

- This is not a training certificate.
- Both partitions are single-voice Piper synthetic audio.
- Synthetic holdout evidence is diagnostic only and not real-speech generalization evidence.
- ASR scoring and selected-training construction require a later work order.
