# Roadmap

The roadmap is organized as evidence-bearing milestones, not calendar promises.

## M0 — Strategic scaffold

Status: **complete**

Deliverables:

- repository constitution;
- architecture decisions;
- data, evaluation, testing, and release policies;
- detailed adaptation plan;
- issue, PR, and work-order templates.

Exit gate:

- repository created;
- first scaffold review completed;
- no model or data artifacts committed.

## M1 — Runtime contract and baseline inference

Status: **in progress**

Deliverables:

- pinned NeMo environment: **implemented as pinned setup path**;
- pinned base checkpoint downloader: **implemented**;
- runtime contract inspector: **implemented**;
- tokenizer audit: **implemented**;
- forced `sl-SI` baseline: **implemented as wrappers, pending GPU evidence**;
- streaming evaluation at all released context settings: **wrappers implemented, pending GPU evidence**;
- small public fixture manifest without sensitive audio: **schema and text-only example implemented**.

Exit gate:

- reproducible inference command;
- checkpoint and dependency hashes recorded;
- baseline outputs archived;
- no training yet.

Work order:
[`work-orders/0001-runtime-contract-and-baseline-inference.md`](work-orders/0001-runtime-contract-and-baseline-inference.md)

## M2 — Data and TTS ingestion

Deliverables:

- candidate schema;
- TTS adapter interface;
- audio and manifest validator;
- provenance records;
- partition and leakage checks;
- synthetic text deduplication.

Exit gate:

- a small generated batch passes validation;
- no word alignment requirement;
- no immutable-gate leakage.

## M3 — Selective adaptation proof

Deliverables:

- prompt-specific trainable-surface implementation;
- trainable-parameter diff verifier;
- tiny-set overfit test;
- saved and reloadable `.nemo` challenger;
- first real-Slovenian gate comparison.

Exit gate:

- intended parameter region is the only changed region;
- resulting checkpoint runs streaming inference;
- no transfer or latency regression beyond declared limits.

## M4 — Active GaMS/TTS loop

Deliverables:

- error taxonomy;
- bounded GaMS failure brief;
- candidate pre-scoring;
- active selector;
- replay reservoir;
- fixed-budget round trainer;
- acceptance and rollback automation.

Exit gate:

- at least three traceable rounds;
- accepted-parent discipline enforced;
- real-speech metrics guide acceptance.

## M5 — Accuracy and transfer campaign

Deliverables:

- comparison of prompt-specific, prompt-kernel, emission, and limited-encoder stages;
- multilingual regression suite;
- latency/accuracy matrix;
- real/synthetic gap analysis;
- failure-driven curriculum report.

Exit gate:

- selected architecture justified by evidence;
- results reproducible from committed configs;
- final blind test remains unused for tuning.

## M6 — Public model release candidate

Deliverables:

- adapter or merged checkpoint;
- Hugging Face model card;
- license and attribution review;
- evaluation report;
- inference quickstart;
- known limitations;
- release decision brief.

Exit gate:

- all release gates pass;
- human release authority approves;
- public claims match evidence.

## Initial PR sequence

1. Repository strategic scaffold.
2. Runtime contract and baseline inference.
3. Manifest and audio validation.
4. Evaluation normalization and metrics.
5. Selective prompt-specific adaptation.
6. Checkpoint-diff integrity tests.
7. GaMS candidate schema and text validation.
8. TTS adapter and provenance.
9. Active candidate selector.
10. Bounded training-round runner.
11. Acceptance/rollback gates.
12. Multilingual and latency regression harness.
13. Model release tooling and model-card template.

A failing or unresolved PR is repaired before the next feature slice.
