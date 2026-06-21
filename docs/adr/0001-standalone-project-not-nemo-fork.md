# ADR 0001: Standalone project rather than a NeMo fork

- Status: Accepted
- Date: 2026-06-21

## Context

The project adapts an NVIDIA NeMo checkpoint but owns a comparatively small amount of orchestration, selective-training, evaluation, and release logic.

Forking NeMo would create a large maintenance surface, obscure upstream provenance, and make it harder to distinguish SLAIF code from framework code.

## Decision

Create a standalone repository:

```text
ulfe-lmi/slaif-asr-slovenian
```

NeMo is an external dependency pinned to a revision or release.

Small upstream scripts may be copied and modified only when necessary. Their original license headers, upstream path, and revision must be preserved.

## Consequences

Positive:

- small review surface;
- clear SLAIF ownership boundary;
- easier upstream updates;
- clearer licensing;
- no misleading claim that SLAIF maintains NeMo.

Costs:

- compatibility must be tested against pinned NeMo revisions;
- some wrapper or patch code may be required;
- breaking upstream changes require deliberate migration.

## Rejected alternatives

- Full NeMo fork.
- Copying the whole training example tree into this repository.
- Using an unpinned floating NeMo `main` branch in released experiments.
