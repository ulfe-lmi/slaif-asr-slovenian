<div align="center">
  <a href="https://www.slaif.si">
    <img src="https://slaif.si/img/logos/SLAIF_logo_ANG_barve.svg" width="400" alt="SLAIF">
  </a>
</div>

# SLAIF Slovenian Streaming ASR

[![License: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)
[![CPU CI](https://github.com/ulfe-lmi/slaif-asr-slovenian/actions/workflows/ci.yml/badge.svg)](https://github.com/ulfe-lmi/slaif-asr-slovenian/actions/workflows/ci.yml)
[![Project status](https://img.shields.io/badge/status-real%20gates%20established-yellow.svg)](docs/roadmap.md)

SLAIF Slovenian Streaming ASR is a reproducible research and engineering project for adapting, evaluating, and releasing open-weight streaming automatic speech recognition models for Slovenian.

The first supported base model is [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), trained and served through NVIDIA NeMo. The intended adaptation loop is:

```text
GaMS generates a small Slovenian candidate batch
    -> Slovenian TTS renders audio
    -> current ASR checkpoint transcribes and scores it
    -> the system selects actual failures and coverage gaps
    -> a bounded fine-tuning update is trained
    -> immutable real-speech and transfer gates accept or reject it
    -> the next batch is generated from the remaining failures
```

## Status

This repository is currently at **real-gate-guided adaptation research**. The
prompt-column micro-proof is complete, active-curriculum tooling exists, the
first immutable real Slovenian development gates establish the untouched
Nemotron baseline, and the first project-generated curriculum round plus a
residual-adapter capacity proof have both been rejected as accepted parents by
the fixed real gates. The first GaMS corpus-v2 candidate reservoir has reached
`TEXT_ACCEPTED` and `AUDIO_ACCEPTED` as a reviewed, single-voice synthetic
candidate pool. A separately generated 96-row synthetic diagnostic holdout has
reached `TEXT_ACCEPTED` after a whole-file human decision and `AUDIO_ACCEPTED`
after Piper synthesis plus waveform validation. A privacy-safe scoring
authorization certificate permitted ASR scoring and selected-training
construction; those steps have produced aggregate untouched-base scoring
evidence and a selected-training manifest with
`SELECTED_TRAINING_MANIFEST_READY` status. A named `DIAGNOSTIC_ONLY`
prompt-column experiment has now run on that selected-training manifest. It
improved the synthetic diagnostic holdout but failed real-gate non-regression,
so no checkpoint is accepted and no corpus is `TRAINING_ELIGIBLE`. A follow-up
speaker-range resampling diagnostic changed only deterministic training-audio
resampling against the same batch-8 prompt-column protocol. It still improved
the synthetic holdout but did not prevent or mitigate real-gate regression, so
the accepted parent remains the untouched Nemotron checkpoint. A frozen-base
Slovenian RNNT joint-adapter diagnostic then trained one NeMo-native adapter in
`model.joint` while every pretrained Nemotron tensor remained frozen. It also
learned the synthetic holdout but regressed the real gates, so it is
synthetic-only and no adapter or checkpoint is accepted.

Present:

- project constitution for coding agents;
- CPU-only GitHub Actions baseline for repository hygiene and unit checks;
- architecture and trust-boundary decisions;
- data, testing, evaluation, and release policies;
- a training-data constitution governing corpus admission before TTS, scoring,
  selection, or training;
- a fail-closed text-stage training-corpus validator, protected-gate hash-index
  builder, and retired-corpus registry;
- a governed GaMS corpus-v2 candidate-reservoir generator, local review-pack
  builder, and privacy-safe aggregate DRAFT report;
- a corpus-v2 review-admission command and privacy-safe aggregate
  post-review report;
- a whole-file human review-decision mode for exact-hash bounded corpora;
- a corpus-v2 Piper synthesis bridge, bounded worker benchmark, acoustic
  validator, and privacy-safe `AUDIO_ACCEPTED` certificate;
- a GaMS-9B independent synthetic holdout generator, privacy-safe DRAFT
  generation report, and privacy-safe `TEXT_ACCEPTED` review-admission report
  for the 96-row holdout;
- privacy-safe `AUDIO_ACCEPTED` evidence for the 96-row holdout and a
  `SCORING_AUTHORIZED` certificate for candidate-source and synthetic-holdout
  ASR scoring;
- aggregate untouched-base ASR scoring reports for the accepted candidate
  source and independent synthetic holdout;
- a privacy-safe selected-training manifest certificate with
  `SELECTED_TRAINING_MANIFEST_READY` status;
- a corpus-v2 prompt-column diagnostic report showing synthetic-only behavior
  under the named `DIAGNOSTIC_ONLY` exception;
- a corpus-v2 speaker-range augmentation diagnostic report showing that
  deterministic resampling proxies did not mitigate real-gate regression;
- a corpus-v2 Slovenian joint-adapter diagnostic report showing that one
  frozen-base RNNT joint-hidden adapter learned synthetic diagnostics but
  regressed real gates;
- shared privacy-safe live progress infrastructure used by long-running
  training and evaluation commands;
- an A100 batched streaming evaluation substrate with batch-1 parity checks,
  duration-bucketed sweeps, and a measured policy for future real-gate
  evaluation;
- a detailed Nemotron/NeMo adaptation plan;
- PR, issue, review, and work-order templates;
- pinned baseline runtime configuration;
- official checkpoint download and checksum helper;
- runtime contract and Slovenian tokenizer-audit commands;
- forced `sl-SI` cache-aware streaming inference wrappers.
- a pinned external Piper TTS configuration for the `sl_SI-artur-medium`
  Slovenian voice;
- a small synthetic-smoke candidate fixture and local TTS-to-ASR vertical-slice
  helpers.
- a prompt-column-only Slovenian adaptation proof that changes only the derived
  `sl-SI` first prompt-projection input column in an ignored local checkpoint.
- pinned GaMS generator configuration and deterministic active-curriculum
  validation for the next prompt-column generalization experiment.
- immutable real-speech development gates for the complete FLEURS Slovenian
  test split (`fleurs-sl-si-test-full-v2`) and a deterministic ARTUR-J
  public-speech project gate.
- a completed project-generated Round 1 curriculum experiment whose
  prompt-column challenger improved selected synthetic training examples but
  regressed FLEURS and ARTUR-J, so it is not an accepted parent.
- a completed Slovenian residual-adapter proof on one A100 logical GPU. Rank 16
  and rank 64 adapters both improved the fixed synthetic holdout but regressed
  FLEURS and ARTUR-J, so neither is an accepted parent.
- a valid untouched-base FLEURS-v2 ASR baseline and A100 batch-policy
  experiment. Batch sizes above 1 were faster but not transcript-equivalent;
  batch size 1 without duration bucketing remains the selected A100 policy and
  the scientific reference mode.
- a completed corpus-v2 prompt-column diagnostic. The batch-size-1 reference
  and batch-8 training arm both trained only the 2,048-value `sl-SI` prompt
  column; both remained unaccepted. True minibatch training improved throughput
  but was not scientifically equivalent to the reference arm.
- a completed corpus-v2 speaker-range augmentation diagnostic. It used fixed
  batch size 8, did not sweep hyperparameters or batches, evaluated with
  batch size 1, and classified the resampling intervention as not supported.
- a completed corpus-v2 Slovenian joint-adapter diagnostic. It trained exactly
  one RNNT joint-hidden adapter with batch size 8 on the original clean Piper
  audio, evaluated with batch size 1, emitted live progress, and classified the
  result as synthetic-only.

Current GPU execution code supports exactly one visible NVIDIA A100 or RTX 2080
Ti. The current A100 development host uses physical GPU 1 selected with
`CUDA_VISIBLE_DEVICES=1`, which maps to PyTorch logical `cuda:0`. Historical
M1/M2 evidence used one RTX 2080 Ti. Other physical GPUs remain unused unless a
work order explicitly permits them.

Not yet present:

- model weights;
- a `TRAINING_ELIGIBLE` corpus or data acceptance certificate;
- a released Slovenian checkpoint.

No accuracy or readiness claim should be inferred from the baseline runtime tooling.
The M1 smoke evidence proves functional restoration and single-GPU inference only;
it is not a Slovenian quality benchmark.
CPU CI does not install NeMo, download checkpoints or audio, use either GPU, or
prove model restoration or GPU inference.
The M2 TTS slice renders only ignored local synthetic-smoke audio and does not
authorize publishing synthetic audio or model artifacts.
The M3 prompt-column proof is a tiny synthetic micro-overfit result, not a
benchmark or production-readiness claim. The public real-smoke diagnostic
regressed after the micro-update. The active-curriculum protocol must pass fixed
synthetic and real gates before any challenger can become an accepted parent.
The real FLEURS and ARTUR-J gates are immutable development gates, not final
blind tests and not release-quality claims. Their raw references, audio,
manifests, hypotheses, and per-sample outputs remain ignored local artifacts.
Historical FLEURS v1 aggregate metrics are deprecated because v1 used
non-unique upstream source IDs for sample IDs and WAV filenames; its 834
manifest rows represented only 347 unique sample identities. ARTUR-J evidence is
unaffected. FLEURS v2 now has a valid untouched-base baseline in
[`docs/experiments/0006-a100-batched-streaming-evaluation.md`](docs/experiments/0006-a100-batched-streaming-evaluation.md).
The Round 1 curriculum text, audio, hypotheses, delta, checkpoint, residual
adapter artifacts, and raw reports are also ignored local artifacts and are not
published by this repository.
The Round 1 v1 candidate pool, synthetic holdout, and selected-training
manifest are permanently retired for future training, steering, model
comparison, and promotion because later review found structural repetition,
train/holdout template-family overlap, and pervasive Slovenian quality defects.
The corpus-v2 prompt-column diagnostic is also not a release or public-quality
claim: its data status is `DIAGNOSTIC_ONLY`, the training audio is single-voice
synthetic Piper output, evaluation still uses batch size 1, and the accepted
parent remains the untouched NVIDIA Nemotron checkpoint.
The speaker-range augmentation diagnostic is likewise not a release or
public-quality claim. Its five resampling profiles are acoustic proxies only,
not real child, elderly, gender, or multi-speaker coverage. It did not accept a
checkpoint and did not authorize publication of augmented audio.
The Slovenian joint-adapter diagnostic is also not a release or public-quality
claim. It preserved the encoder, tokenizer, prompt kernel, decoder, and RNNT
joint base weights, trained only the named adapter, used the original clean
single-voice Piper audio, and did not accept or publish an adapter.
Future promotion-oriented training requires `TRAINING_ELIGIBLE` data under the
training-data constitution. The corpus-v2 reservoir is not committed as raw
text or audio. It now has privacy-safe aggregate `TEXT_ACCEPTED` and
`AUDIO_ACCEPTED` evidence for 415 reviewed Piper-rendered items, and the
separately generated 96-row GaMS-9B synthetic diagnostic holdout has reached
`TEXT_ACCEPTED` and `AUDIO_ACCEPTED`. The partition-level scoring certificate
authorized ASR scoring of both partitions and selected-training construction
from the accepted candidate source. Aggregate scoring and selected-training
construction are now complete, with the selected manifest classified as
`SELECTED_TRAINING_MANIFEST_READY`. There is still no model-training
authorization or `TRAINING_ELIGIBLE` corpus.

## Repository role

This repository contains the **method, orchestration, configurations, tests, and release evidence**.

It is deliberately not:

- a fork of NeMo;
- a copy of the Nemotron checkpoint;
- a storage location for raw speech corpora;
- a storage location for private or generated training audio;
- the primary distribution channel for trained model weights.

NeMo remains an external, pinned dependency. Model artifacts will be released separately through Hugging Face with clear base-model attribution and the applicable model license.
Piper remains an external GPL-3.0-or-later executable dependency installed into
`.venv-piper`; it is not imported into the Apache-licensed package and is not
vendored into this repository.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap and PR sequence](docs/roadmap.md)
- [Detailed training plan](docs/training-plan.md)
- [Data policy](docs/data-policy.md)
- [Training-data constitution](docs/training-data-constitution.md)
- [Training corpus text validator](docs/data-quality-validator.md)
- [Testing strategy](docs/testing-strategy.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Evaluation datasets](docs/evaluation-datasets.md)
- [Release policy](docs/release-policy.md)
- [Third-party licenses and attribution](docs/third-party-licenses.md)
- [Baseline inference quickstart](docs/baseline-inference.md)
- [Current project handoff](docs/project-handoff.md)
- [Architecture decisions](docs/adr/)
- [Execution work orders](docs/work-orders/)

Coding agents must read [`AGENTS.md`](AGENTS.md) before changing the repository. Agents that use the companion instruction file must also read [`CLAUDE.md`](CLAUDE.md).

## Planned artifact names

GitHub project:

```text
ulfe-lmi/slaif-asr-slovenian
```

Initial Hugging Face adapter release:

```text
ulfe-lmi/slaif-asr-slovenian-nemotron-3.5-adapter
```

Possible later merged checkpoint:

```text
ulfe-lmi/slaif-asr-slovenian-nemotron-3.5
```

Names are project decisions, not claims of ownership over NVIDIA technology.

## Current validation task

The corpus-v2 ASR scoring and selected-training construction task is defined by
[`docs/work-orders/0019-corpus-v2-asr-scoring-and-selection.md`](docs/work-orders/0019-corpus-v2-asr-scoring-and-selection.md).
The bulk review and acoustic admission task is defined by
[`docs/work-orders/0015-bulk-review-and-acoustic-admission.md`](docs/work-orders/0015-bulk-review-and-acoustic-admission.md).
The independent corpus-v2 holdout task is defined by
[`docs/work-orders/0017-corpus-v2-independent-holdout.md`](docs/work-orders/0017-corpus-v2-independent-holdout.md).
The corpus-v2 linguistic-review admission task is defined by
[`docs/work-orders/0014-corpus-v2-linguistic-review-admission.md`](docs/work-orders/0014-corpus-v2-linguistic-review-admission.md).
The GaMS corpus-v2 candidate-reservoir task is defined by
[`docs/work-orders/0013-gams-corpus-v2-candidate-reservoir.md`](docs/work-orders/0013-gams-corpus-v2-candidate-reservoir.md).
The fail-closed training-corpus validator task is defined by
[`docs/work-orders/0012-fail-closed-training-corpus-validator.md`](docs/work-orders/0012-fail-closed-training-corpus-validator.md).
The training-data constitution adoption task is defined by
[`docs/work-orders/0011-adopt-training-data-constitution.md`](docs/work-orders/0011-adopt-training-data-constitution.md).
The Slovenian residual-adapter proof is defined by
[`docs/work-orders/0009-slovenian-residual-adapter-proof.md`](docs/work-orders/0009-slovenian-residual-adapter-proof.md).
The project-generated Slovenian curriculum Round 1 task is defined by
[`docs/work-orders/0008-slovenian-curriculum-round-1.md`](docs/work-orders/0008-slovenian-curriculum-round-1.md).
The real Slovenian evaluation-suite task is defined by
[`docs/work-orders/0007-real-slovenian-evaluation-suite.md`](docs/work-orders/0007-real-slovenian-evaluation-suite.md).
The prompt-column micro-overfit task is defined by
[`docs/work-orders/0005-m3-prompt-column-adaptation-proof.md`](docs/work-orders/0005-m3-prompt-column-adaptation-proof.md).
The GaMS-directed prompt-column active-curriculum task is defined by
[`docs/work-orders/0006-gams-prompt-column-active-curriculum.md`](docs/work-orders/0006-gams-prompt-column-active-curriculum.md).
The Piper Slovenian TTS ingestion task is defined by
[`docs/work-orders/0004-piper-slovenian-tts-ingestion.md`](docs/work-orders/0004-piper-slovenian-tts-ingestion.md).
The CPU CI baseline task is defined by
[`docs/work-orders/0003-cpu-ci-baseline.md`](docs/work-orders/0003-cpu-ci-baseline.md).
The runtime repair-and-verification task is defined by
[`docs/work-orders/0002-m1-runtime-repair-and-2080ti-verification.md`](docs/work-orders/0002-m1-runtime-repair-and-2080ti-verification.md).
The original runtime baseline work order is
[`docs/work-orders/0001-runtime-contract-and-baseline-inference.md`](docs/work-orders/0001-runtime-contract-and-baseline-inference.md).

## Licensing

Code and original documentation in this repository are licensed under the [Apache License 2.0](LICENSE).

The NVIDIA base checkpoint is **not included** and is licensed separately under NVIDIA Open Model Development and Weight License 1.1 (OpenMDW 1.1). Users must obtain the base model from its official distribution point and comply with its license.

The selected Piper TTS engine is **not included** and is licensed separately
under GPL-3.0-or-later. The selected `sl_SI-artur-medium` voice is downloaded
from `rhasspy/piper-voices`; its repository, model-card, and ARTUR source
license metadata disagree, so this project applies the conservative ARTUR
CC BY-SA 4.0 attribution and publication policy. See
[third-party licenses](docs/third-party-licenses.md).

A future adapter or merged checkpoint is a derived model artifact and must carry the applicable base-model license, attribution, model card, training disclosure, and evaluation disclosure. See [release policy](docs/release-policy.md).

## Acknowledgement

We acknowledge the support of the EC/EuroHPC JU and the Slovenian Ministry of HESI via the project SLAIF (grant number 101254461).

## Security and responsible disclosure

Do not open a public issue containing credentials, personal speech, private transcripts, or undisclosed dataset details. See [`SECURITY.md`](SECURITY.md).
