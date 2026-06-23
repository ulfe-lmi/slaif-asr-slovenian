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
- GaMS command-line probe and BF16 generation-debugging report for the A100
  runtime.
- Slovenian residual-adapter proof configuration, shared single-GPU hardware
  policy helper, adapter implementation, tests, execution runner, and
  privacy-safe aggregate experiment report.
- FLEURS full-test v2 gate metadata, occurrence-index identity planning,
  full-gate verifier, and v1 deprecation record.
- Training-data constitution, data-admission ADR, and governance links for
  future corpus validation and acceptance certificates.
- Fail-closed text-stage training-corpus validator, protected-gate hash-index
  builder, retired-corpus registry, adversarial fixtures, and validator usage
  documentation.
- GaMS corpus-v2 candidate-reservoir configuration, batched generation harness,
  native-speaker review-pack builder, CPU tests, and privacy-safe DRAFT data
  report.
- Corpus-v2 linguistic-review admission command, local accepted-subset and
  review-decision outputs, CPU tests, and privacy-safe aggregate post-review
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
- GaMS generation now uses the model-compatible Transformers stack and 4-bit
  BF16 compute, with explicit attention masks and correct padding semantics.
- GPU execution helpers no longer assume physical GPU 0 or RTX 2080 Ti only;
  they now require exactly one visible A100 or RTX 2080 Ti and use logical
  `cuda:0`.
- Residual-adapter evidence records synthetic-only behavior: rank 16 and rank
  64 adapters improved the fixed synthetic holdout but regressed both real
  gates, so no adapter is accepted as a parent.
- Canonical FLEURS documentation now points to `fleurs-sl-si-test-full-v2`.
  Historical FLEURS v1 evidence is deprecated because non-unique upstream
  source IDs caused duplicate sample IDs and WAV overwrites; v1 metrics must
  not be used as complete-split quality evidence.
- Round 1 v1 corpus identities are permanently retired from future training,
  steering, model comparison, early stopping, generator steering, promotion,
  and public corpus-quality claims. Experiments 0004 and 0005 remain auditable
  historical evidence, but their architecture-level conclusions are narrowed by
  the corpus-confounding finding.
- New corpus admission must use the text-stage validator rather than the
  historical Round 1 schema/duplicate checker. The validator can produce
  `TEXT_ACCEPTED`, but it does not prove acoustic suitability or issue
  `TRAINING_ELIGIBLE`.
- The first corpus-v2 GaMS reservoir is DRAFT only. It is not committed as raw
  generated text, has no fabricated linguistic review, and is not authorized
  for TTS, ASR scoring, selection, or training.
- The current edited corpus-v2 review sheet records 415 `ACCEPT` outcomes but
  omits the required `review_revision` on every row. Review admission therefore
  remains `DRAFT`; no accepted review sidecar, data certificate, TTS, scoring,
  selection, or training is authorized.
