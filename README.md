<div align="center">
  <a href="https://www.slaif.si">
    <img src="https://slaif.si/img/logos/SLAIF_logo_ANG_barve.svg" width="400" alt="SLAIF">
  </a>
</div>

# SLAIF Slovenian Streaming ASR

[![License: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)
[![Project status](https://img.shields.io/badge/status-runtime%20baseline-yellow.svg)](docs/roadmap.md)

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

This repository is currently at **M1 complete: runtime contract and baseline inference smoke verified**.

Present:

- project constitution for coding agents;
- architecture and trust-boundary decisions;
- data, testing, evaluation, and release policies;
- a detailed Nemotron/NeMo adaptation plan;
- PR, issue, review, and work-order templates;
- pinned baseline runtime configuration;
- official checkpoint download and checksum helper;
- runtime contract and Slovenian tokenizer-audit commands;
- forced `sl-SI` cache-aware streaming inference wrappers.

Current M1/M2 development hardware is one physical NVIDIA RTX 2080 Ti selected
with `CUDA_VISIBLE_DEVICES=0`. A second RTX 2080 Ti may be present in the
development host but remains unused unless a later work order explicitly permits
it. A100 hardware is not a default prerequisite.

Not yet present:

- training code;
- datasets;
- model weights;
- benchmark results;
- a released Slovenian checkpoint.

No accuracy or readiness claim should be inferred from the baseline runtime tooling.
The M1 smoke evidence proves functional restoration and single-GPU inference only;
it is not a Slovenian quality benchmark.

## Repository role

This repository contains the **method, orchestration, configurations, tests, and release evidence**.

It is deliberately not:

- a fork of NeMo;
- a copy of the Nemotron checkpoint;
- a storage location for raw speech corpora;
- a storage location for private or generated training audio;
- the primary distribution channel for trained model weights.

NeMo remains an external, pinned dependency. Model artifacts will be released separately through Hugging Face with clear base-model attribution and the applicable model license.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap and PR sequence](docs/roadmap.md)
- [Detailed training plan](docs/training-plan.md)
- [Data policy](docs/data-policy.md)
- [Testing strategy](docs/testing-strategy.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Release policy](docs/release-policy.md)
- [Baseline inference quickstart](docs/baseline-inference.md)
- [Current project handoff](docs/project-handoff.md)
- [Architecture decisions](docs/adr/)
- [Execution work orders](docs/work-orders/)

Coding agents must read [`AGENTS.md`](AGENTS.md) before changing the repository. Claude Code must also read [`CLAUDE.md`](CLAUDE.md).

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

## Current runtime task

The current runtime repair-and-verification task is defined by
[`docs/work-orders/0002-m1-runtime-repair-and-2080ti-verification.md`](docs/work-orders/0002-m1-runtime-repair-and-2080ti-verification.md).
The original runtime baseline work order is
[`docs/work-orders/0001-runtime-contract-and-baseline-inference.md`](docs/work-orders/0001-runtime-contract-and-baseline-inference.md).

## Licensing

Code and original documentation in this repository are licensed under the [Apache License 2.0](LICENSE).

The NVIDIA base checkpoint is **not included** and is licensed separately under NVIDIA Open Model Development and Weight License 1.1 (OpenMDW 1.1). Users must obtain the base model from its official distribution point and comply with its license.

A future adapter or merged checkpoint is a derived model artifact and must carry the applicable base-model license, attribution, model card, training disclosure, and evaluation disclosure. See [release policy](docs/release-policy.md).

## Acknowledgement

We acknowledge the support of the EC/EuroHPC JU and the Slovenian Ministry of HESI via the project SLAIF (grant number 101254461).

## Security and responsible disclosure

Do not open a public issue containing credentials, personal speech, private transcripts, or undisclosed dataset details. See [`SECURITY.md`](SECURITY.md).
