# Work Order 0003: CPU CI Baseline

## Governing Instructions

- Read `AGENTS.md`, `CLAUDE.md`, `docs/project-handoff.md`,
  `docs/roadmap.md`, `docs/testing-strategy.md`, ADR 0001, and work order
  0002 before editing.
- Treat live `origin/main` as authoritative.
- Verify expected merged M1 commit
  `c03ede4fa997b8549ed4fa439a6e89b1d682562c` before starting.
- Use a repository-local `.venv` for local verification.
- Do not use either RTX 2080 Ti, NeMo, checkpoint downloads, audio downloads,
  Hugging Face access, GPU CI, or repository secrets.

## Goal

Add a deterministic GitHub Actions workflow that runs repository checks which do
not require GPU access, NVIDIA NeMo installation, model checkpoints, audio
artifacts, external model hosting, or secrets.

The workflow is the durable remote validation baseline for subsequent pull
requests. It does not replace M1 GPU verification and does not prove model
restoration, streaming inference, or Slovenian recognition quality.

## Required Changes

- Add `.github/workflows/ci.yml`.
- Trigger on pull requests targeting `main`, pushes to `main`, and manual
  dispatch.
- Use least-privilege `contents: read` permissions.
- Cancel obsolete runs on the same branch through workflow concurrency.
- Use only official GitHub-maintained actions pinned to immutable commit SHAs,
  with comments documenting the corresponding action versions.
- Use Python 3.12.
- Create `.venv` in CI.
- Install only the editable repository package.
- Do not install NeMo, CUDA PyTorch wheels, torchaudio, checkpoints, or audio.
- Add a reusable tracked-file validation command.
- Add focused unit tests for the validation behavior.
- Update README, handoff, roadmap, testing strategy, and changelog
  documentation.

## Required Checks

CI and local verification must run:

- `.venv/bin/python -m unittest discover -s tests`
- `.venv/bin/python -m py_compile <all tracked repository Python files>`
- `.venv/bin/python scripts/check_repository.py`
- `bash -n scripts/*.sh`
- `git diff --check`
- `git diff --cached --check` before commit

The repository validation command must operate only on tracked files and check:

- tracked JSON and JSONL parse;
- tracked TOML parses;
- relative Markdown links resolve;
- no tracked forbidden model artifacts;
- no tracked audio artifacts;
- no tracked oversized files;
- no tracked temporary container paths or private local paths;
- no obvious credential or private-key material;
- no trailing whitespace.

## Non-Goals

Do not implement M2 data features, modify inference or tokenizer behavior,
modify checkpoint metadata, install or invoke NeMo, download the model, download
audio, add GPU CI, add a self-hosted runner, change repository settings or
secrets, add third-party marketplace actions, add linting/formatting
dependencies, use either RTX 2080 Ti, publish, or merge.

## Acceptance Criteria

- Local CPU checks pass.
- GitHub Actions runs on the PR and passes.
- The workflow requires no GPU, NeMo, checkpoint, audio, Hugging Face access, or
  repository secrets.
- The workflow uses least-privilege permissions.
- Official actions are immutably pinned.
- The CI badge resolves to the new workflow.
- No runtime artifacts are committed.
- Documentation accurately distinguishes CPU CI from GPU evidence.
