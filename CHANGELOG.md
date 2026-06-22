# Changelog

All notable project changes will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases should use semantic versioning where it fits the software artifact.

## [Unreleased]

### Added

- Initial SLAIF project constitution.
- Architecture, data, testing, evaluation, and release policies.
- Nemotron 3.5 Slovenian adaptation plan.
- Work-order, issue, and pull-request templates.
- Pinned Nemotron 3.5 ASR runtime configuration and official checkpoint checksum.
- Baseline download, runtime-contract inspection, Slovenian tokenizer-audit, and forced `sl-SI` streaming inference wrappers.
- Baseline inference quickstart and text-only manifest schema.
- M1 repair work order for one-RTX-2080-Ti verification.
- CUDA 12.6/PyTorch `.venv` runtime requirements and constraints.
- CPU-only GitHub Actions baseline for repository hygiene, unit tests, Python
  compilation, and shell syntax.
- Tracked-file repository validation command for JSON/TOML syntax, Markdown
  links, forbidden artifacts, oversized files, local paths, secrets, and
  trailing whitespace.
- Piper Slovenian TTS ingestion configuration, external environment setup,
  pinned `sl_SI-artur-medium` voice downloader, smoke candidates, rendering
  wrapper, provenance and NeMo manifest generation.
- ADR and third-party attribution documentation for the Piper GPL boundary and
  ARTUR voice license discrepancy.
- Prompt-column-only Slovenian adaptation utilities, metrics, experiment
  configuration, training/evaluation drivers, and privacy-safe aggregate M3
  micro-overfit report.
- GaMS active-curriculum configuration, strict candidate validation,
  deterministic active selection, corpus metric summaries, and promotion or
  rollback helpers.
- `.venv-gams` setup and durable CUDA 12.6 Nemotron training-environment
  verification helpers for the Numba/NVVM stack used in M3.
- ADR, work order, and experiment report scaffold for the GaMS-directed
  prompt-column active-curriculum protocol.
- Real Slovenian evaluation-suite configuration, normalizer, FLEURS full-test
  gate builder, ARTUR-J standardized-transcript parser and gate builder,
  baseline evaluator, privacy-safe gate metadata, and aggregate baseline report.
- Project-generated Slovenian curriculum Round 1 configuration, validation,
  selection, execution runner, tests, and privacy-safe aggregate experiment
  report.

### Changed

- Corrected the Nemotron checkpoint SHA256 and retained the prior Hugging Face LFS ETag separately.
- Runtime contract now separates checkpoint-detected contexts from configured supported contexts.
- Tokenizer audit now distinguishes required Slovenian samples from extended-symbol warnings.
- Single-file inference now resolves relative audio paths and persists per-context `result.json` plus logs.
- M2 status is now in progress for the Piper-to-Nemotron vertical slice; no
  training, GaMS integration, or public audio/model publication is included.
- M3 prompt-column proof records FP32 fallback after FP16 AMP loss-scale
  overflow events and reports the tiny synthetic result separately from holdout
  and public real-smoke diagnostics.
- M3 generalization status now distinguishes protocol/tooling from completed
  two-round GPU evidence.
- Future challenger promotion now requires non-regression on both full FLEURS
  Slovenian test and ARTUR-J public-speech development gates.
- Round 1 prompt-column curriculum evidence now records a rejected challenger:
  selected synthetic training improved, fixed synthetic holdout did not meet
  the promotion threshold, and both real gates regressed.
