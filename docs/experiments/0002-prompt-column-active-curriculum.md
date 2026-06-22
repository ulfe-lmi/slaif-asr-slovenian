# Experiment 0002: Prompt-Column Active Curriculum

Status: **protocol and tooling added; generator probe completed; full two-round
GPU experiment not yet completed in this commit**

## Purpose

This experiment tests whether the M3 prompt-column-only adaptation generalizes
when candidate text is generated from actual ASR failures by a bounded GaMS ->
Piper -> Nemotron active-learning loop.

The trainable surface remains the 2,048-value `sl-SI` first prompt-projection
column delta. No encoder, decoder, joint, tokenizer, full prompt-kernel, or
non-Slovenian prompt parameter is trainable.

## Configuration

- Experiment config:
  [`configs/experiments/prompt_column_active_curriculum.json`](../../configs/experiments/prompt_column_active_curriculum.json)
- Generator config:
  [`configs/generation/gams_prompt_curriculum.json`](../../configs/generation/gams_prompt_curriculum.json)
- Primary GaMS model: `cjvt/GaMS3-12B-Instruct`
- Primary GaMS revision: `1d0b27af5748784482600d24779409e7e1dc9adc`
- Fallback GaMS model: `cjvt/GaMS-9B-Instruct`
- Fallback GaMS revision: `292744023fa0b7ccc7ae2c3c885a67468e49fa03`
- GaMS license: Gemma Terms of Use
- GaMS quantization: 4-bit NF4, double quantization, FP16 compute
- Piper engine revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Piper voice revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Base checkpoint revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Base checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`

## Fixed Gates

The fixed gates are defined before round-specific training:

| Gate | Size | Source | Steering policy |
|---|---:|---|---|
| synthetic holdout | 64 | GaMS independent general Slovenian prompt | never train; never steer from raw errors |
| real FLEURS gate | 64 | `google/fleurs` `sl_si` test split | never train; raw references never sent to GaMS |

Only local ignored manifests and audio contain gate references and paths. Public
reports may include row IDs, hashes, aggregate duration statistics, and aggregate
metrics when license policy permits.

## Metric Contract

This experiment reports all of the following separately:

- corpus WER;
- corpus CER;
- mean utterance WER;
- mean utterance CER;
- median utterance WER;
- median utterance CER;
- empty-hypothesis count.

Mean utterance WER is not labeled as corpus WER.

## Current Evidence

Implemented and locally tested in this commit:

- pinned generator and experiment configs;
- strict GaMS JSON parsing and candidate validation;
- duplicate, near-duplicate, and protected-hash rejection;
- deterministic active hard-example ranking;
- seeded general-control selection;
- round-2 brief schema that excludes real-gate references and synthetic-holdout
  raw errors;
- corpus and mean metric distinction;
- promotion and rollback decision logic;
- reproducible Nemotron training environment helper;
- GaMS setup helper;
- FLEURS gate builder script;
- active selection and promotion-evaluation scripts.

Not yet completed in this commit:

- Piper synthesis of the active candidate pools;
- Nemotron pre-scoring of active candidates;
- round-1 and round-2 prompt-column training;
- final promotion decision and scientific classification.

### Generator Probe

Both GaMS candidates were tested with `CUDA_VISIBLE_DEVICES=0`, 4-bit NF4,
double quantization, FP16 compute, and no CPU offload. GPU 1 stayed at idle
baseline.

| Model | Result | Peak VRAM | Wall time | Notes |
|---|---|---:|---:|---|
| `cjvt/GaMS3-12B-Instruct` | failed strict JSON | 7425.9 MiB | 35.512 s | loaded and generated, but output was multilingual non-JSON text |
| `cjvt/GaMS-9B-Instruct` | passed one-candidate probe | 6080.2 MiB | 25.123 s | generated one valid strict-JSON Slovenian candidate after chat-template formatting |

The fallback result proves the generator interface can produce a valid candidate
on one RTX 2080 Ti. It does not complete the required 64-item synthetic holdout,
128-candidate round pools, Piper synthesis, Nemotron pre-scoring, training, or
fixed-gate evaluation.

## Scientific Classification

`EXPERIMENT_INVALID`

This is a placeholder classification for the unexecuted comparison. It must be
replaced only after both active rounds either complete or fail under the protocol.

## Limitations

- The repository tooling does not itself prove that GaMS3 fits in 11 GiB.
- Synthetic-only gains cannot be described as Slovenian ASR improvement.
- No challenger checkpoint from this experiment is accepted until the fixed real
  gate passes promotion criteria.
