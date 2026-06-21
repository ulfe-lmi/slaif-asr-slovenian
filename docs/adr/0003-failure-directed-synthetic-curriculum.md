# ADR 0003: Failure-directed synthetic curriculum

- Status: Accepted
- Date: 2026-06-21

## Context

GaMS can generate effectively unbounded Slovenian text, and the existing TTS can render it. A large static synthetic corpus would be easy to create but inefficient, difficult to audit, and likely to overrepresent synthetic style.

## Decision

Use a closed active-learning loop:

1. evaluate the last accepted checkpoint;
2. cluster current failures and coverage gaps;
3. ask GaMS for a bounded candidate batch;
4. validate and synthesize the batch;
5. pre-evaluate all candidates;
6. train mainly on actual failures plus replay and general controls;
7. run immutable real-speech, latency, and transfer gates;
8. accept or roll back;
9. generate the next batch from remaining failures.

The accepted checkpoint, not the newest challenger, is the parent of the next round.

## Consequences

Positive:

- compute and TTS output are focused on observed errors;
- every round is auditable;
- curriculum adapts to the current model;
- synthetic data does not become an unbounded static asset.

Costs and risks:

- orchestration is more complex than one-shot dataset generation;
- controller-development metrics become adaptive and are not unbiased;
- leakage controls and replay are mandatory;
- GaMS can become myopic without general controls.
