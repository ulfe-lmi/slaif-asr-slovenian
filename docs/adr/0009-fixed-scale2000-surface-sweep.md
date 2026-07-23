# ADR 0009: Fixed Scale-2000 Trainable-Surface Sweep

## Status

Accepted for the bounded diagnostic program defined by Work Order 0037.

## Decision

Authorize a fixed-data trainable-surface sweep on original scale-2000 augmented
data. Each experiment changes one named trainable surface while preserving the
corpus, exposure schedule, validation protocol, target language, streaming
context, and reporting contract.

Phase 1 authorizes only
`SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK`: the RNNT decoder, RNNT joint,
and exactly the final encoder block. ARTUR controller-dev may provide aggregate
run-control under ADR 0008. FLEURS-v2 and ARTUR-J remain post-selection
directional gates and cannot select a checkpoint.

## Reason

The strongest clean WER/CER reduction came from original scale-2000 augmented
audio with decoder+joint RNNT training. Later data mixtures introduced
tradeoffs. The next controlled question is which model surface should move,
not which data variant should be added.

## Permitted In Phase 1

- Train `model.decoder`.
- Train `model.joint`.
- Train exactly the final encoder block, discovered as `encoder.layers.23` in
  the pinned live model.
- Use the fixed scale-2000 exposure schedule without replacement or expansion.
- Use ARTUR controller-dev aggregate metrics for run-control under ADR 0008.

## Forbidden

- Full encoder training or any lower encoder block.
- Subsampling, frontend, or preprocessor training.
- Tokenizer or prompt labels, tables, or embeddings changes.
- S6TTS, scale-8000, database-extension, or real-speech training data.
- FLEURS-v2 or ARTUR-J checkpoint selection.
- Text-only objectives, temporary LM heads, or adapters.
- Checkpoint acceptance, model release, `TRAINING_ELIGIBLE`, or an
  `accepted_parent` change.

## Consequences

This is a diagnostic exception to the default synthetic-only encoder freeze,
not a general authorization to train encoder parameters. Later encoder depths
require separate work orders and evidence. Full-encoder training remains
prohibited.
