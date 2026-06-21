# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability involving:

- credentials or access tokens;
- private speech or transcripts;
- personal data;
- unpublished dataset locations;
- unsafe model-download or checkpoint-loading behavior;
- model artifact licensing or provenance concerns;
- a path to mutate production or external systems.

Report privately to:

```text
janez.pers@fe.uni-lj.si
```

Include a concise description, affected revision, reproduction conditions, and impact. Do not attach sensitive audio unless explicitly requested through an approved secure channel.

## Supported versions

The project is pre-release. Security fixes target the current `main` branch and the latest tagged release once releases begin.

## Project security boundaries

This repository must not contain:

- production credentials;
- private SSH or cloud keys;
- personal speech recordings;
- non-public transcripts;
- raw participant data;
- NVIDIA base weights;
- trained model checkpoints;
- generated training corpora.

Execution agents operate in disposable environments without production secrets. Public releases require a separate release decision and model-card review.

## Supply-chain expectations

- Pin NeMo and model revisions.
- Prefer official sources.
- Record checksums for downloaded checkpoints.
- Review licenses before adding dependencies.
- Treat deserialization of model checkpoints as trusted-code execution unless proven otherwise.
