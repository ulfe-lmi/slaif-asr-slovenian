# Work Order 0002: M1 Runtime Repair and RTX 2080 Ti Verification

## Governing Instructions

- Read `AGENTS.md`, `CLAUDE.md`, `docs/project-handoff.md`, `docs/roadmap.md`,
  `docs/baseline-inference.md`, `docs/testing-strategy.md`,
  `docs/evaluation-protocol.md`, ADR 0001, and ADR 0002.
- Treat live `origin/main` as authoritative.
- Verify merge commit `b86407f42e61ddc89b1eb4dca666ae4f46ebf3a8` is an ancestor of
  `origin/main`.
- Use one repository-local `.venv`.
- Use exactly one GPU: physical GPU 0 through `CUDA_VISIBLE_DEVICES=0`.
- Do not use Conda, global Python installs, Docker, multiple GPUs, DDP, NCCL,
  model sharding, or the second RTX 2080 Ti.

## Goal

Repair the M1 runtime defects found during verification and close M1 with real
single-GPU evidence on one RTX 2080 Ti.

The PR must provide:

- correct checkpoint checksum metadata;
- repeatable CUDA 12.6/PyTorch virtual environment setup;
- correct relative and absolute audio path handling;
- machine-readable single-file inference results;
- separated detected-vs-configured streaming-context contract semantics;
- required-vs-extended Slovenian tokenizer audit classification;
- successful inference evidence for all five contexts on one RTX 2080 Ti;
- updated governance, roadmap, baseline, and handoff documentation.

## Required Environment

- 48 GB class system RAM;
- 2 x NVIDIA RTX 2080 Ti, 11 GB each;
- physical GPU 0 selected with `CUDA_VISIBLE_DEVICES=0`;
- physical GPU 1 unused;
- Python 3.12;
- `.venv` created with `python3 -m venv`;
- `torch==2.7.1+cu126`;
- matching CUDA 12.6 `torchaudio`;
- pinned NeMo revision `8044a3924bfcfe8ef71d792bb73bf274fe853575`.

## Required Repairs

1. Repair `scripts/setup_runtime_env.sh` so it is idempotent, explicit about
   `.venv`, pins CUDA 12.6 PyTorch before installing NeMo, can recreate broken
   local environments through a documented option, and runs `pip check`.
2. Correct checkpoint SHA256 to the Hugging Face LFS file digest
   `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
   Preserve the prior HTTP ETag only in a separately named field if retained.
3. Resolve `--audio-file`, manifests, checkpoint, NeMo root, and output paths
   consistently before inference starts.
4. Persist every successful single-file context result to
   `runs/inference/<run-id>/context_<left>_<right>/result.json` with a
   human-readable log beside it.
5. Represent checkpoint-detected streaming contexts separately from configured
   project-supported contexts.
6. Make tokenizer audit default success depend on required Slovenian samples,
   while retaining extended-symbol failures as warnings. Add `--strict-all` for
   nonzero exit on any sample failure.
7. Update governance and handoff documentation for one-RTX-2080-Ti M1/M2 and
   first M3 proof execution.

## Required Verification

Run and report:

- preflight commands: `git rev-parse HEAD`, `git status --short`,
  `git log --oneline --decorate -n 10`, `nvidia-smi -L`, `nvidia-smi`,
  `free -h`, `df -h /`, and `python3 --version`;
- clean `.venv` setup through `scripts/setup_runtime_env.sh --recreate`;
- `.venv/bin/python -m pip check`;
- NeMo ASR import verification;
- one-visible-GPU assertion under `CUDA_VISIBLE_DEVICES=0`;
- checkpoint checksum verification;
- CPU runtime-contract generation;
- tokenizer audit in default and strict modes;
- all repository unit tests;
- all repository Python `py_compile`;
- `bash -n scripts/*.sh`;
- JSON and TOML syntax validation;
- Markdown relative-link validation;
- forbidden artifact, large-file, secret, temporary container path, and
  local-path scans;
- ignored-artifact verification;
- actual five-context GPU smoke on one RTX 2080 Ti in the order:
  `[56,0]`, `[56,3]`, `[56,13]`, `[56,1]`, `[56,6]`;
- at least one relative audio path run and equivalent absolute audio path run;
- one `--all-contexts` invocation when memory and runtime permit;
- second-GPU non-use evidence from `nvidia-smi` before, during, and after runs;
- `git diff --check`;
- `git diff --cached --check`.

## Non-Goals

Do not train, fine-tune, integrate GaMS or TTS, implement active learning,
change the base model, replace or retrain the tokenizer, use the second GPU,
use an A100, add DDP/NCCL/FSDP/DeepSpeed/model parallelism, add an API or UI,
commit checkpoints/audio/logs/generated results, publish a model or dataset,
modify external NeMo source, or claim Slovenian production quality.

## Final Report

Use the structured Agent Report requested by this work order and `AGENTS.md`.
