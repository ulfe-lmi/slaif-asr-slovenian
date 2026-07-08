# ADR 0008: ARTUR Controller-Dev Run Control

Status: Accepted

Date: 2026-07-08

## Context

The project has immutable real gates for FLEURS-v2 and ARTUR-J. Those gates are
validation-only acceptance evidence and must not be spent on early stopping,
hyperparameter tuning, or checkpoint selection. Experiment 0017 showed that
synthetic RNNT probe loss was non-monotonic across exposure rounds, so synthetic
loss alone is not a reliable checkpoint selector for future longer
decoder+joint work.

## Decision

Create `artur-controller-dev-v1`, an ARTUR public-speech
controller-development partition for real-acoustic run-control. It must be
source-recording disjoint from `artur-j-public-gate-v1`, protected-text
disjoint from the current immutable gates, and represented publicly only by
privacy-safe aggregate certificates and reports unless a later data-release
review authorizes more.

## Permitted Uses

- aggregate per-epoch/per-round validation WER;
- aggregate CER;
- aggregate empty hypothesis count;
- aggregate insertion, deletion, and substitution rates;
- aggregate RNNT validation loss if implemented;
- early stopping and checkpoint selection in future explicitly authorized
  training work orders.

## Forbidden Uses

- training;
- gradient updates;
- synthetic prompt construction;
- GaMS prompt content;
- selected-training construction;
- hard-example mining from raw references or hypotheses;
- immutable-gate acceptance;
- public quality claims;
- model release claims.

## Consequences

Using this partition for early stopping makes it spent development data. It is
not unbiased acceptance evidence and must not be reported as a final gate. The
immutable `artur-j-public-gate-v1`, immutable `fleurs-sl-si-test-full-v2`, and
any final blind test remain unavailable for early stopping, hyperparameter
selection, synthetic prompt construction, and model selection.

Future work orders may run a second-GPU validation process against completed
checkpoint snapshots, but only as independent single-visible-GPU processes with
read-only checkpoint access and aggregate output.
