# Work Order 0038: Fixed-Data Surface05 Last Two Encoder Blocks

## Goal

Continue the fixed scale-2000 trainable-surface sweep by testing only
`SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS` from the untouched
Nemotron base.

## Authorized Change

Train `decoder`, `joint`, `encoder.layers.22`, and `encoder.layers.23`. Use
learning rates `5e-4`, `5e-4`, and `2e-5` for both encoder blocks. All lower
encoder blocks, frontend/preprocessor, prompt pathway, tokenizer, adapters, and
temporary heads remain frozen or absent.

## Fixed Data And Control

- Corpus: `sl-corpus-v4-gams-16000-training-v1`.
- Exposures: 320,000 under the committed scale-2000 schedule.
- ARTUR controller-dev: aggregate batch-1 run-control under ADR 0008.
- FLEURS-v2 and ARTUR-J: post-selection directional batch-32 only.
- No S6TTS, scale-8000, database-extension, or real-speech training data.

## Status Boundaries

This work is `DIAGNOSTIC_ONLY`, cannot issue `TRAINING_ELIGIBLE`, cannot accept
or publish a checkpoint, and keeps `accepted_parent` as `none`. Local
checkpoints and manifests remain ignored. No raw audio, predictions,
references, hypotheses, or local paths enter Git. Surface06 remains
unauthorized after this work order pending strategic review.
