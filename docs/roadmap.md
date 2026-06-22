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

Status: **complete**

Deliverables:

- pinned NeMo environment: **implemented as `.venv` CUDA 12.6 setup path**;
- pinned base checkpoint downloader: **implemented**;
- runtime contract inspector: **implemented**;
- tokenizer audit: **implemented**;
- forced `sl-SI` baseline: **implemented and smoke-verified on one RTX 2080 Ti**;
- streaming evaluation at all released context settings: **implemented and smoke-verified on one RTX 2080 Ti**;
- small public fixture manifest without sensitive audio: **schema and text-only example implemented**.

Exit gate:

- reproducible inference command;
- checkpoint and dependency hashes recorded;
- baseline outputs archived;
- no training yet.

M1 completion means the runtime contract and one short single-GPU smoke path work.
It does not establish Slovenian ASR quality.

Work order:
[`work-orders/0001-runtime-contract-and-baseline-inference.md`](work-orders/0001-runtime-contract-and-baseline-inference.md)

Repair and verification work order:
[`work-orders/0002-m1-runtime-repair-and-2080ti-verification.md`](work-orders/0002-m1-runtime-repair-and-2080ti-verification.md)

CPU CI baseline work order:
[`work-orders/0003-cpu-ci-baseline.md`](work-orders/0003-cpu-ci-baseline.md)

CPU CI is the durable pull-request baseline for repository hygiene and unit
checks. It does not install NeMo, restore the model, use GPUs, or prove
Slovenian ASR quality. GPU verification remains separate manual or future
self-hosted evidence.

## M2 — Data and TTS ingestion

Status: **vertical slice complete; scalable governance pending**

Execution hardware policy: use one RTX 2080 Ti process-visible GPU unless a later work order explicitly permits different hardware.

Deliverables:

- candidate schema;
- TTS adapter interface: **Piper vertical slice complete on one RTX 2080 Ti**;
- audio and manifest validator;
- provenance records;
- partition and leakage checks;
- synthetic text deduplication.

Exit gate:

- a small generated batch passes validation;
- no word alignment requirement;
- no immutable-gate leakage.

Current completed vertical-slice work order:
[`work-orders/0004-piper-slovenian-tts-ingestion.md`](work-orders/0004-piper-slovenian-tts-ingestion.md)

The current M2 slice proves real Piper `sl_SI-artur-medium` synthesis to a
Nemotron smoke manifest on one RTX 2080 Ti. It does not implement GaMS
generation, failure-directed selection, large-batch synthesis, leakage controls,
replay, or production training manifests.

## Real Evaluation Gates

Status: **complete for initial development gates**

Deliverables:

- complete FLEURS Slovenian test gate: **implemented and baseline-evaluated**;
- deterministic ARTUR-J public-speech gate: **implemented and baseline-evaluated**;
- Slovenian normalizer `sl-asr-normalization-v1`: **implemented**;
- untouched Nemotron aggregate baseline: **recorded**;
- unaccepted micro-proof diagnostic on both gates: **recorded; regressed**.

These are immutable development gates, not final blind tests or release
criteria. Raw references, audio, manifests, hypotheses, and per-sample outputs
remain ignored local artifacts.

Report:
[`experiments/0003-real-slovenian-baseline.md`](experiments/0003-real-slovenian-baseline.md)

Work order:
[`work-orders/0007-real-slovenian-evaluation-suite.md`](work-orders/0007-real-slovenian-evaluation-suite.md)

## M3 — Selective adaptation proof

Status: **prompt-column micro-proof complete; active-curriculum generalization tooling in progress**

The first prompt-specific proof is expected to attempt one RTX 2080 Ti with FP16 AMP. A100 is requested only after measured memory, throughput, or authoritative benchmark evidence supports escalation.

Deliverables:

- prompt-specific trainable-surface implementation: **implemented for one
  prompt column**;
- trainable-parameter diff verifier: **implemented**;
- tiny-set overfit test: **completed for the eight Piper smoke candidates**;
- saved and reloadable `.nemo` challenger: **verified locally, ignored**;
- first real-Slovenian gate comparison: **diagnostic public-smoke comparison
  run; it regressed and is not a benchmark**.

Exit gate:

- intended parameter region is the only changed region;
- resulting checkpoint runs streaming inference;
- no transfer or latency regression beyond declared limits.

The first micro-experiment supports the narrow claim that a 2048-scalar
`sl-SI` prompt-column delta can overfit a tiny synthetic set while changing only
the selected prompt column. It does not validate release quality. The full
FLEURS and ARTUR-J diagnostics both regressed for the ignored micro-proof
checkpoint, so it remains unaccepted and is not a valid parent.

Work order:
[`work-orders/0005-m3-prompt-column-adaptation-proof.md`](work-orders/0005-m3-prompt-column-adaptation-proof.md)

Aggregate report:
[`experiments/0001-prompt-column-micro-overfit.md`](experiments/0001-prompt-column-micro-overfit.md)

Active-curriculum work order:
[`work-orders/0006-gams-prompt-column-active-curriculum.md`](work-orders/0006-gams-prompt-column-active-curriculum.md)

Active-curriculum report:
[`experiments/0002-prompt-column-active-curriculum.md`](experiments/0002-prompt-column-active-curriculum.md)

The active-curriculum PR adds the bounded GaMS -> Piper -> Nemotron protocol,
metric corrections, and promotion/rollback machinery. It does not by itself make
a challenger accepted; real GPU rounds and FLEURS plus ARTUR-J fixed-gate
outcomes remain the deciding evidence.

## M4 — Active GaMS/TTS loop

Status: **not started as a release loop**

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
3. CPU-only CI baseline.
4. Piper Slovenian TTS ingestion.
5. Manifest and audio validation.
6. Evaluation normalization and metrics.
7. Selective prompt-specific adaptation.
8. Checkpoint-diff integrity tests.
9. GaMS candidate schema and text validation.
10. Active candidate selector.
11. Bounded training-round runner.
12. Acceptance/rollback gates.
13. Multilingual and latency regression harness.
14. Model release tooling and model-card template.

A failing or unresolved PR is repaired before the next feature slice.
