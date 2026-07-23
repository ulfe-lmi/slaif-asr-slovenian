# ADR 0009: Fixed Scale-2000 Trainable-Surface Sweep

## Status

Accepted for the bounded diagnostic program defined by Work Orders 0037 and
0038.

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

## Phase 2 / Work Order 0038

Authorize
`SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS`: the RNNT decoder, RNNT
joint, and exactly the final two encoder blocks. In the pinned live model these
must resolve to `encoder.layers.22` and `encoder.layers.23`; execution fails
closed if that identity cannot be proved.

Phase 2 permits:

- training the decoder and joint;
- training exactly the final two encoder blocks at the lower encoder learning
  rate;
- using only the original scale-2000 augmented corpus v4 and its fixed
  exposure schedule;
- using ARTUR controller-dev aggregate run-control under ADR 0008.

Phase 2 forbids:

- full encoder training or encoder blocks below the final two;
- frontend, subsampling, or preprocessor training;
- tokenizer or prompt labels, tables, embeddings, or fusion-path changes;
- S6TTS, scale-8000, database-extension, or real-speech training data;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- Surface06, checkpoint acceptance, model publication, `TRAINING_ELIGIBLE`, or
  an `accepted_parent` change.

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

These are diagnostic exceptions to the default synthetic-only encoder freeze,
not a general authorization to train encoder parameters. Surface06 and later
encoder depths require separate work orders and evidence. Full-encoder training
remains prohibited.
