# ADR 0002: Nemotron 3.5 ASR Streaming 0.6B as the initial base

- Status: Accepted
- Date: 2026-06-21

## Context

The project requires a streaming recognizer whose supported-language performance is in the same class as Voxtral Realtime but whose training path is easier to reproduce.

The selected model exposes:

- cache-aware streaming;
- a standard RNNT objective;
- official NeMo training scaffolding;
- language prompt conditioning including `sl-SI`;
- controllable latency contexts;
- a substantially smaller parameter count than Voxtral Realtime.

## Decision

Use:

```text
nvidia/nemotron-3.5-asr-streaming-0.6b
```

as the first supported base checkpoint and NVIDIA NeMo as the training/inference framework.

The exact checkpoint and NeMo revisions are established in M1 by runtime inspection and then pinned.

## Consequences

Positive:

- no custom word-timing target builder;
- ordinary audio/transcript manifests;
- official cache-aware inference;
- direct Slovenian prompt pathway;
- feasible M1/M2 experimentation on one RTX 2080 Ti, with larger GPUs requested only when measured evidence requires escalation.

Risks:

- newer NeMo interfaces may still change;
- prompt and shared RNNT components require transfer-regression gates;
- model artifacts use a license separate from repository code;
- zero-shot Slovenian quality must be measured rather than assumed.

## Rejected alternatives

- Voxtral Realtime as the first implementation target.
- Training a streaming recognizer from scratch.
- Replacing the base tokenizer before auditing it.
