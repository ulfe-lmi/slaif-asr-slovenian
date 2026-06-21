# Contributing

Thank you for contributing to SLAIF Slovenian Streaming ASR.

## Before opening a change

1. Read [`AGENTS.md`](AGENTS.md).
2. Read the relevant architecture decision records in [`docs/adr/`](docs/adr/).
3. Check the current [roadmap](docs/roadmap.md) and [project handoff](docs/project-handoff.md).
4. Open or reference an issue for non-trivial behavior changes.
5. Keep one pull request focused on one coherent outcome.

## Pull requests

A pull request should include:

- the problem and intended behavior;
- explicit non-goals;
- files changed;
- tests and exact results;
- GPU/runtime details when relevant;
- documentation impact;
- risks and known limitations;
- confirmation that no model weights, private speech, generated corpora, secrets, or unrelated files were committed.

Use the repository pull-request template.

## Experiments

Experiment PRs must be reproducible. Record:

- base-model and NeMo revisions;
- checkpoint and manifest hashes;
- configuration;
- trainable parameter list;
- seeds;
- hardware;
- commands;
- results;
- acceptance or rollback decision.

Raw experiment output belongs in ignored local storage or approved external artifact storage, not Git.

## Code style

Project-specific Python and shell tooling will be introduced by a dedicated work order. Until then:

- keep Markdown readable;
- use UTF-8;
- use LF line endings;
- avoid trailing whitespace;
- keep filenames lowercase with hyphens except established files such as `AGENTS.md`.

## Licensing

By contributing original code or documentation, you agree that it may be distributed under Apache-2.0.

Do not contribute third-party code unless its license is compatible and its attribution is preserved. Model artifacts are governed separately from repository code.
