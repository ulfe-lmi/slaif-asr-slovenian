# Work Order 0045: Surface08 Scale-8000 OTF Augmented

Status: completed historical authorization record

This record was added during Work Order 0050 because the original PR #51 stack
did not contain a `docs/work-orders/0045*` file. It records the governing
boundaries of the already completed run and does not create new authorization or
change Experiment 0031 evidence.

## Goal

Run one Surface08 data-axis diagnostic from the untouched Nemotron base using
the admitted 64,000-row scale-8000 clean synthetic pool and deterministic,
in-memory transcript-preserving augmentation.

## Fixed Controls

- Surface08: decoder, joint, all 24 encoder layers, and `prompt_kernel`.
- Frontend, tokenizer, prompt identity, language mapping, and every other
  surface remain frozen.
- Effective batch: 8.
- ARTUR controller-dev provides aggregate run-control only.
- FLEURS-v2 and ARTUR-J are post-selection directional gates only.
- No pre-rendered augmented WAVs.
- No real speech, S6TTS, scale-2000, or new rows as training sources.

## Configured Schedule

- Primary exposure budget: 160,000.
- Absolute maximum: 320,000.
- Controller early stopping: enabled under ADR 0008.
- Actual stopped exposure count: 144,000.

## Result

Experiment 0031 selected round 5 and stopped at round 9 after 144,000 virtual
exposures. It classified
`SCALE8000_OTF_SURFACE08_NEW_BEST_DIRECTIONAL`, scoring 39.353/12.002/0 on
FLEURS-v2 and 39.974/12.251/0 on ARTUR-J.

This was a bounded diagnostic. It did not accept a checkpoint, issue
`TRAINING_ELIGIBLE`, authorize Surface09 or full-model training, or authorize
publication.
