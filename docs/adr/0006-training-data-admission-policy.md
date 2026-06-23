# ADR 0006: Training-data admission policy

- Status: Accepted
- Date: 2026-06-23

## Context

Slovenian Curriculum Round 1 passed narrow schema and duplicate validation while
remaining structurally repetitive, linguistically defective, and
train/holdout template-confounded.

Hard-example selection subsequently amplified artifacts that had already been
admitted into the source pool. High ASR error on those artifacts did not prove
they were good training examples.

Synthetic improvement from that corpus therefore did not establish real
Slovenian generalization.

## Decision

Adopt [`docs/training-data-constitution.md`](../training-data-constitution.md)
as the detailed constitutional companion for training-data doctrine.

Require an explicit data-status state machine:

```text
DRAFT
TEXT_REJECTED
TEXT_ACCEPTED
AUDIO_REJECTED
AUDIO_ACCEPTED
TRAINING_ELIGIBLE
DIAGNOSTIC_ONLY
RETIRED
```

Promotion-oriented model training is prohibited unless the data has reached the
required status and a privacy-safe acceptance certificate has been committed.

The following Round 1 v1 corpus identities are permanently retired:

- candidate pool:
  `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17`
- synthetic holdout:
  `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9`
- selected training manifest:
  `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4`

Corpus acceptance is separate from model experimentation. A training run can be
runtime-valid and parameter-integrity-valid while still being
data-confounded.

## Consequences

Corpus validation becomes a first-class pre-GPU stage.

Future data work requires stronger structural, partition, acoustic, and
Slovenian linguistic evidence before TTS, candidate scoring, selection, or
training.

Historical experiments remain auditable, including their metrics, hashes,
commands, and promotion decisions.

Architecture conclusions drawn from corpus-confounded experiments are narrowed.
Experiment 0004 remains a valid rejection of its challenger. Experiment 0005
remains valid as execution and parameter-integrity evidence, but it must not be
cited as proof that residual adapters, their placement, or added capacity are
intrinsically unsuitable.

Reusable validation tooling, adversarial fixtures, and the first data
acceptance certificate are later implementation work.
