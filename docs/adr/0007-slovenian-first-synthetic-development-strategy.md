# ADR 0007: Slovenian-First Synthetic Development Strategy

## Status

Accepted

## Date

2026-07-05

## Context

The original project strategy emphasized preserving multilingual behavior while
making the smallest possible Slovenian-specific update. That remains a useful
safety instinct, but the product objective has narrowed: build a usable
Slovenian, and eventually Slovenian-English, low-latency streaming ASR system.

The project also has a hard data constraint: no real Slovenian acoustic samples
may be used for model training. Real Slovenian speech is reserved for
validation and acceptance evidence. Synthetic text and synthetic TTS audio are
therefore the only currently authorized training signal.

Recent scale diagnostics show that larger and more acoustically varied synthetic
training can reduce real-gate regression directionally. The relevant acceptance
question is not whether training audio was synthetic; it is whether the
completed challenger passes validation-only real-speech gates without leakage
and under a committed protocol.

## Decision

The project adopts a Slovenian-first synthetic development track:

1. The primary engineering target is Slovenian streaming ASR, with
   Slovenian-English as the likely first bilingual extension.
2. Preserving the base model's multilingual behavior is secondary to improving
   Slovenian utility, except where a work order explicitly includes a
   multilingual regression gate.
3. Real Slovenian acoustic data is validation-only. It must not enter training,
   synthetic prompt construction, selected-training membership, early stopping,
   hyperparameter tuning, per-sample steering, or adapter-surface selection.
4. Aggregate real-gate metrics may compare completed challengers and decide
   whether a completed challenger is useful enough for the next governed step.
5. While training remains synthetic-only, the acoustic encoder stays frozen by
   default. Any encoder training requires a later ADR and explicit human
   approval.
6. Broader non-encoder emission surfaces are permitted under explicit work
   orders. These include larger RNNT joint adapters, decoder adapters,
   joint-plus-decoder adapters, and frozen-encoder joint/decoder fine-tuning.
7. Tokenizer replacement remains a late-stage architectural change. It requires
   a separate ADR, decoder impact analysis, and human approval.
8. Batch-32 directional evaluation is useful for iteration speed, but canonical
   batch-1 evaluation remains required before any acceptance or release
   discussion.

## Consequences

- Future work orders may authorize synthetic-only training without using real
  acoustic samples for training. Those work orders must state whether the run is
  diagnostic-only or whether a completed challenger may be considered by
  validation-only real gates.
- Synthetic metrics alone cannot accept a parent checkpoint. A synthetic-trained
  challenger may be considered only if the committed real validation and release
  protocol permits it.
- The training-data constitution continues to require privacy-safe certificates,
  partition independence, and acoustic validation before synthetic training.
- Real gates are development validation instruments. Repeated use makes them
  unsuitable as a final blind test.
- A later release candidate will need a fresh, human-approved final blind
  protocol or other release-grade real-speech evidence.

## Non-Decisions

- This ADR does not authorize training on real Slovenian acoustic samples.
- This ADR does not accept any existing checkpoint, adapter, or synthetic corpus
  as production-ready.
- This ADR does not authorize public model publication.
- This ADR does not replace the tokenizer.
