# Work Order 0001: Runtime contract and baseline Slovenian inference

## Governing instructions

- Read `AGENTS.md` and `CLAUDE.md`.
- Read ADR 0001 and ADR 0002.
- Read `docs/testing-strategy.md` and `docs/evaluation-protocol.md`.
- If the official model or NeMo interfaces differ from the documented assumptions, report the difference and adjust only enough to complete this work order.

## Current verified state

- Repository contains strategic scaffold only.
- Base checkpoint is selected but not pinned in executable code.
- No environment lock, downloader, inspection script, or inference wrapper exists.
- No training is permitted in this task.

## Goal

Create a reproducible baseline that downloads the official checkpoint, inspects the loaded runtime contract, audits Slovenian tokenizer behavior, and runs cache-aware streaming inference with the forced `sl-SI` prompt.

## Domain behavior to preserve

- Slovenian characters `č`, `š`, and `ž`;
- punctuation and capitalization expected by the base model;
- native cache-aware streaming behavior;
- explicit NVIDIA base-model attribution;
- no exposure of private audio or transcripts.

## Scope

Implement:

- a pinned development environment or reproducible installation path;
- an official-checkpoint download helper;
- a runtime inspection command;
- tokenizer round-trip audit;
- single-file and manifest-based streaming inference wrappers;
- a small public text-only example manifest schema;
- focused tests that do not require private data;
- baseline documentation.

## Non-goals

- Do not fine-tune.
- Do not implement GaMS or TTS integration.
- Do not add active selection.
- Do not publish weights.
- Do not replace the tokenizer.
- Do not commit audio or checkpoint files.
- Do not add a hosted API or UI.
- Do not claim Slovenian quality beyond measured outputs.

## Files and areas to inspect

- current repository scaffold;
- official Nemotron 3.5 ASR Streaming model card and files;
- official NeMo prompt-aware RNNT model;
- official cache-aware inference script;
- official fine-tuning example only to understand restoration/config behavior.

## Required behavior

### Environment

- Pin a NeMo revision known to load this checkpoint.
- Record Python, PyTorch, CUDA, and NeMo requirements.
- Prefer a source or container workflow that is reproducible on A100.
- Avoid an unpinned `main` install.

### Download

- Download only from the official model repository.
- Pin the model revision.
- Store the checkpoint under an ignored directory.
- Write or verify a SHA256 checksum.

### Inspection

Produce a machine-readable contract containing at least:

- loaded class;
- total parameter count;
- encoder layer count and width;
- tokenizer vocabulary size;
- sample rate;
- `sl-SI` and `sl` prompt indices;
- prompt-kernel structure;
- available/default streaming context configuration;
- checkpoint and environment revisions.

### Tokenizer audit

Test representative Slovenian:

```text
abcčdefghijklmnoprsštuvzž
ABCČDEFGHIJKLMNOPRSŠTUVZŽ
Čez cesto švigne žaba.
Ljubljana, 21. junij 2026.
Zaženi Docker Compose in preveri GPU.
Cena je 12,50 €, temperatura pa 23,7 °C.
```

Record IDs, decoded text, and a clear pass/fail result.

### Inference

Provide:

- mono 16 kHz audio conversion instructions;
- forced `target_lang=sl-SI`;
- one-file streaming inference;
- manifest streaming inference;
- wrappers for `[56,0]`, `[56,1]`, `[56,3]`, `[56,6]`, and `[56,13]`;
- privacy-safe output handling.

## Acceptance criteria

- A clean A100 environment can follow the documented setup.
- The checkpoint loads from the pinned revision.
- Runtime contract is generated.
- `sl-SI` resolves correctly.
- tokenizer audit results are explicit.
- a user-supplied Slovenian WAV can be transcribed at all five settings.
- checkpoint/audio/output artifacts remain ignored.
- no training code is introduced.
- docs state exact limitations and environment blockers.

## Tests and evidence required

- configuration syntax validation;
- unit tests for contract serialization and tokenizer-audit result handling where practical;
- shell lint or equivalent for wrappers;
- `git diff --check`;
- forbidden-large-file scan;
- secret scan;
- one real GPU inference smoke test when environment permits.

GPU report must state exact commands, GPU, CUDA, PyTorch, NeMo, checkpoint revision, context setting, and result.

## Documentation required

Update:

- `README.md` status;
- `docs/project-handoff.md`;
- `docs/roadmap.md` M1 status;
- a new quickstart or inference document;
- `CHANGELOG.md`.

Do not describe training as implemented.

## Workflow

- Create branch `feat/runtime-contract-baseline`.
- Commit only this task.
- Open one pull request.
- Do not merge.

## Final report

Use the structured report from `AGENTS.md`, with a separate table for the five streaming settings.
