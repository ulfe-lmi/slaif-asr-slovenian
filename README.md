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

This repository is currently at **real-gate baseline evaluation**. The
prompt-column micro-proof is complete, active-curriculum tooling exists, and
the first immutable real Slovenian development gates now establish the
untouched Nemotron baseline before any further curriculum work. The first
project-generated Slovenian curriculum round has run and its challenger was
rejected by the fixed gates.

Present:

- project constitution for coding agents;
- CPU-only GitHub Actions baseline for repository hygiene and unit checks;
- architecture and trust-boundary decisions;
- data, testing, evaluation, and release policies;
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
  test split and a deterministic ARTUR-J public-speech project gate, with
  untouched-base aggregate metrics.
- a completed project-generated Round 1 curriculum experiment whose
  prompt-column challenger improved selected synthetic training examples but
  regressed FLEURS and ARTUR-J, so it is not an accepted parent.

Current M1/M2 development hardware is one physical NVIDIA RTX 2080 Ti selected
with `CUDA_VISIBLE_DEVICES=0`. A second RTX 2080 Ti may be present in the
development host but remains unused unless a later work order explicitly permits
it. A100 hardware is not a default prerequisite.

Not yet present:

- model weights;
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
The Round 1 curriculum text, audio, hypotheses, delta, and checkpoint are also
ignored local artifacts and are not published by this repository.

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
