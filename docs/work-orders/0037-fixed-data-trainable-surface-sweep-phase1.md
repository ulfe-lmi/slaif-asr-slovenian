# Work Order 0037: Fixed-Data Trainable-Surface Sweep, Phase 1

## Goal

Hold original scale-2000 augmented data and all evaluation controls fixed while
testing `SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK` from the untouched
Nemotron base.

## Authorized Change

Train `decoder`, `joint`, and exactly `encoder.layers.23`. Use separate learning
rates of `5e-4`, `5e-4`, and `2e-5`, respectively. All lower encoder blocks,
frontend/preprocessor, prompt pathway, tokenizer, adapters, and temporary heads
remain frozen or absent.

## Data And Control

- Corpus: `sl-corpus-v4-gams-16000-training-v1`.
- Exposures: 320,000 under the committed scale-2000 schedule.
- ARTUR controller-dev: aggregate batch-1 run-control under ADR 0008.
- FLEURS-v2 and ARTUR-J: post-selection directional batch-32 only.
- No S6TTS, scale-8000, database-extension, or real-speech training data.

## Status Boundaries

This work is `DIAGNOSTIC_ONLY`, cannot issue `TRAINING_ELIGIBLE`, cannot accept
or publish a checkpoint, and keeps `accepted_parent` as `none`. Local
checkpoints and manifests remain ignored. No raw audio, predictions,
references, hypotheses, or local paths enter Git.
