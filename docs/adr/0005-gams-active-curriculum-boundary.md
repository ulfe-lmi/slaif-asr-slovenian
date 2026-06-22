# ADR 0005: GaMS Active-Curriculum Boundary

- Status: Accepted
- Date: 2026-06-22

## Context

M3 experiment 0001 proved that a 2,048-parameter `sl-SI` prompt-column delta can
memorize a tiny synthetic set while preserving parameter isolation. It did not
show generalization: synthetic holdout WER did not improve and one public real
diagnostic regressed.

The next controlled question is whether the same trainable surface can improve
when candidate text is generated from actual failures instead of fixed smoke
sentences.

## Decision

Use GaMS as an external local generator for two bounded active-learning rounds:

- primary model: `cjvt/GaMS3-12B-Instruct`;
- primary revision: `1d0b27af5748784482600d24779409e7e1dc9adc`;
- fallback model: `cjvt/GaMS-9B-Instruct`;
- fallback revision: `292744023fa0b7ccc7ae2c3c885a67468e49fa03`;
- license: Gemma Terms of Use;
- execution: `.venv-gams`, Transformers, Accelerate, bitsandbytes, 4-bit NF4,
  double quantization, BF16 compute, one operator-selected visible GPU;
- CPU offload, unauthorized additional GPUs, model sharding, and multiple
  generators in one scientific comparison are forbidden.

GaMS returns strict JSON candidate records. The repository validates UTF-8 NFC
text, `sl-SI`, unique safe IDs, duplicate and near-duplicate text,
protected-gate overlap, and bounded text length. Invalid rows are rejected and
counted.

Real FLEURS gate reference text is protected. It may be used for evaluation, but
raw real-gate references must not be sent to GaMS. Round-2 steering may use only
synthetic candidate-pool failures and aggregate real-gate category counts.

## Consequences

Positive:

- the active loop is auditable and bounded;
- generator, TTS, ASR scoring, and training environments remain isolated;
- synthetic gains are separated from real-gate behavior;
- rollback is automatic when promotion gates fail.

Costs and risks:

- GaMS3 primary generation may require A100-class memory for reliable BF16
  generation;
- generation is slower because model phases run sequentially;
- two active rounds remain too small for production claims;
- the same prompt-column trainable surface may still fail to generalize.
