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

### Changed

- Corrected the Nemotron checkpoint SHA256 and retained the prior Hugging Face LFS ETag separately.
- Runtime contract now separates checkpoint-detected contexts from configured supported contexts.
- Tokenizer audit now distinguishes required Slovenian samples from extended-symbol warnings.
- Single-file inference now resolves relative audio paths and persists per-context `result.json` plus logs.
