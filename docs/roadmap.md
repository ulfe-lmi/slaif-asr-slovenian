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

Status: **vertical slice complete; text and audio validators implemented; corpus v2 selected-training manifest ready**

Execution hardware policy: use exactly one process-visible GPU. Current
project-owned helpers accept one visible A100 or RTX 2080 Ti and reject CPU
fallback or multiple visible GPUs. Historical M2 evidence used one RTX 2080 Ti;
current A100 experiments use physical GPU 1 selected with
`CUDA_VISIBLE_DEVICES=1`.

Deliverables:

- candidate schema;
- TTS adapter interface: **Piper vertical slice complete on one RTX 2080 Ti**;
- audio and manifest validator;
- provenance records;
- partition and leakage checks;
- synthetic text deduplication;
- reusable text-stage training-data admission validator: **implemented**;
- GaMS corpus-v2 candidate reservoir: **implemented and text-admitted as
  `TEXT_ACCEPTED` after whole-file human review**;
- acoustic validator and privacy-safe audio certificate: **implemented for the
  415-row single-voice candidate reservoir as `AUDIO_ACCEPTED`**.
- independent synthetic diagnostic holdout: **implemented as a 96-row GaMS-9B
  partition and text-admitted as `TEXT_ACCEPTED` after whole-file human
  review, then waveform-validated as `AUDIO_ACCEPTED`**.
- scoring authorization certificate: **implemented as `SCORING_AUTHORIZED`;
  ASR scoring and selected-training construction have run under that boundary,
  and model training remains prohibited**.
- selected-training manifest: **implemented as
  `SELECTED_TRAINING_MANIFEST_READY` for 160 candidate-source rows**.

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

The training-data constitution is now a prerequisite for promotion-oriented
data work. The retired Round 1 v1 corpus identities must not be reused for
training, steering, model comparison, or promotion. The text-stage validator can
produce `TEXT_ACCEPTED`; the synthetic audio validator can produce
`AUDIO_ACCEPTED`. The first corpus-v2 GaMS reservoir has reached both states
for 415 reviewed, single-voice Piper-rendered candidates, and the separate
96-row GaMS-9B diagnostic holdout has reached both states. A scoring
authorization certificate permits ASR scoring of both partitions and
selected-training construction from the candidate source. Scoring and
selected-training construction are complete, with 160 candidate-source rows in
the selected-training manifest. No `TRAINING_ELIGIBLE` data certificate exists.

## Real Evaluation Gates

Status: **FLEURS v2 and ARTUR-J untouched-base baselines complete; A100 batch policy measured**

Deliverables:

- complete FLEURS Slovenian test gate: **implemented as
  `fleurs-sl-si-test-full-v2` with 834 unique occurrences and a valid
  untouched-base ASR baseline**;
- deterministic ARTUR-J public-speech gate: **implemented and baseline-evaluated**;
- Slovenian normalizer `sl-asr-normalization-v1`: **implemented**;
- untouched Nemotron aggregate baseline: **recorded for FLEURS-v2 and ARTUR-J;
  historical FLEURS-v1 baseline is deprecated**;
- A100 batch policy: **measured as batch size 1 without duration bucketing;
  larger tested batches are not transcript-equivalent**;
- unaccepted micro-proof diagnostic on ARTUR-J: **recorded; regressed**.

These are immutable development gates, not final blind tests or release
criteria. Raw references, audio, manifests, hypotheses, and per-sample outputs
remain ignored local artifacts. Historical `fleurs-sl-si-test-full-v1` files
remain available only for auditability and must not be used as complete-split
quality evidence.

Report:
[`experiments/0003-real-slovenian-baseline.md`](experiments/0003-real-slovenian-baseline.md)
and
[`experiments/0006-a100-batched-streaming-evaluation.md`](experiments/0006-a100-batched-streaming-evaluation.md)

Work order:
[`work-orders/0007-real-slovenian-evaluation-suite.md`](work-orders/0007-real-slovenian-evaluation-suite.md)

## M3 — Selective adaptation proof

Status: **prompt-column micro-proof complete; corpus-v2 prompt-column diagnostic synthetic-only**

Historical prompt-specific evidence first attempted one RTX 2080 Ti. Current
A100-hosted experiments still use exactly one visible logical GPU and do not use
multi-GPU training unless a future work order permits it.

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
the selected prompt column. It does not validate release quality. ARTUR-J
regressed for the ignored micro-proof checkpoint, and the historical
FLEURS-v1 component is deprecated, so it remains unaccepted and is not a valid
parent.

The corpus-v2 prompt-column diagnostic used the reviewed selected-training
manifest once under a named `DIAGNOSTIC_ONLY` exception. The batch-size-1
reference arm and the throughput-selected batch-8 arm improved synthetic
diagnostics but failed real-gate non-regression. The scientific classification
is `CORPUS_V2_PROMPT_COLUMN_SYNTHETIC_ONLY`, the batching classification is
`A100_PROMPT_TRAINING_BATCH_NOT_EQUIVALENT`, and the accepted parent remains
the untouched Nemotron checkpoint.

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

Project-generated Slovenian curriculum Round 1 has now run without GaMS or an
external LLM. It improved selected synthetic training examples but failed the
fixed synthetic-holdout threshold and regressed ARTUR-J. The historical
FLEURS-v1 component is deprecated. The challenger is rejected and is not a valid
parent.

Later review found the Round 1 v1 corpus structurally repetitive,
linguistically defective, and train/holdout template-confounded. This narrows
the scientific interpretation: the rejection remains valid, but the result must
not be cited as evidence that a clean curriculum would fail.

Round 1 work order:
[`work-orders/0008-slovenian-curriculum-round-1.md`](work-orders/0008-slovenian-curriculum-round-1.md)

Round 1 report:
[`experiments/0004-slovenian-curriculum-round-1.md`](experiments/0004-slovenian-curriculum-round-1.md)

The Slovenian residual-adapter proof reused the exact Round 1 corpus and fixed
real gates while preserving every pretrained Nemotron parameter. Rank 16 and
rank 64 adapters improved fixed synthetic-holdout metrics, but both regressed
ARTUR-J. The historical FLEURS-v1 component is deprecated. The result remains
synthetic-only and no adapter is accepted.

Because this proof reused the retired Round 1 v1 corpus, it is
corpus-confounded. Its runtime and parameter-integrity evidence remain useful,
but it must not be cited as proof that residual adapters, their placement, or
added capacity are intrinsically unsuitable.

Residual-adapter work order:
[`work-orders/0009-slovenian-residual-adapter-proof.md`](work-orders/0009-slovenian-residual-adapter-proof.md)

Residual-adapter report:
[`experiments/0005-slovenian-residual-adapter-proof.md`](experiments/0005-slovenian-residual-adapter-proof.md)

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
