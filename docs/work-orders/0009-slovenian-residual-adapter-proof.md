# Work Order 0009: Slovenian Residual-Adapter Proof

Status: completed in PR draft

## Purpose

Test whether a Slovenian-specific residual adapter can generalize better than
the rejected prompt-column Round 1 challenger while preserving every pretrained
Nemotron parameter.

This work order reuses the exact synthetic corpus, selected training manifest,
and immutable real evaluation gates from the rejected project-generated Round 1
experiment. It does not generate new sentences, run GaMS, or train on real
speech.

## Required Branch And Commit

- Branch: `feat/slovenian-residual-adapter-proof`
- Commit: `feat: add Slovenian residual-adapter proof`
- Pull request title: `feat: add Slovenian residual-adapter proof`

## Hardware Policy

Use only physical GPU 1 on the current A100 host:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Inside PyTorch this must appear as exactly one logical CUDA device, `cuda:0`.
The code must reject CPU fallback and multiple visible GPUs. It must accept both
NVIDIA A100 and RTX 2080 Ti devices when exactly one is visible, and must not
assume that the physical device selector is zero.

Do not use GPUs 0, 2, or 3, CPU or disk offload, DDP, NCCL, FSDP, DeepSpeed, or
model sharding.

## Runtime Repair

Replace historical helper assumptions such as `CUDA_VISIBLE_DEVICES=0` and
`2080 Ti`-only checks with a shared project-owned single-GPU policy helper.

The helper must:

- require exactly one visible CUDA device;
- accept A100 and RTX 2080 Ti;
- record the physical selector from `CUDA_VISIBLE_DEVICES`;
- use logical `cuda:0` internally;
- reject multiple visible GPUs and CPU fallback;
- report device name, capability, VRAM, selector, and PyTorch CUDA runtime.

## Adapter Architecture

Wrap the frozen prompt kernel with a Slovenian-only residual path:

```text
base_output = frozen_prompt_kernel(inputs)
sl_active = inputs[..., selected_prompt_column].unsqueeze(-1)
hidden = activation(down(base_output))
residual = up(hidden)
output = base_output + sl_active * residual
```

Requirements:

- derive `sl-SI` prompt index, encoder width, selected prompt column, and prompt
  kernel output width at runtime;
- do not hardcode prompt index `62`, encoder width `1024`, selected column
  `1086`, or output width `2048`;
- keep every original Nemotron parameter frozen;
- optimize only adapter parameters;
- activate only for the `sl-SI` prompt column;
- initialize the output projection to zero so step-zero wrapped output matches
  the base;
- use no bias, dropout, or running-statistics normalization;
- do not modify external NeMo source.

Compare two independent arms:

- rank 16;
- rank 64.

Both start from the untouched accepted base checkpoint.

## Data

Verify and reuse these artifacts from experiment 0004:

- candidate pool SHA256:
  `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17`
- synthetic holdout SHA256:
  `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9`
- selected training manifest SHA256:
  `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4`
- FLEURS manifest SHA256:
  `07838a58222b9a0f6a4f4639b66d678ee38f87254518e43b742a143ef4aeaf4e`
- ARTUR-J manifest SHA256:
  `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763`

Use the same 160 selected synthetic training utterances and the same locked
96-utterance synthetic holdout. Do not use FLEURS or ARTUR-J speech for
training, steering, or early stopping.

## Training

Common configuration:

- precision: FP32;
- TF32: disabled;
- optimizer: AdamW;
- weight decay: 0;
- learning rate: 0.001;
- batch size: 1;
- seed: 1234;
- prompt mode: `langID`;
- target prompt: `sl-SI`;
- maximum optimizer steps: 1800;
- training order: identical deterministic order.

If an arm shows less than 20% training-loss reduction after 600 steps and all
runtime and integrity checks pass, restart that arm once with learning rate
0.0003. Do not perform an open-ended learning-rate search.

## Evaluation

Evaluate on GPU 1 at context `[56,3]`, batch size 1:

- untouched accepted base;
- rank-16 residual adapter;
- rank-64 residual adapter.

Evaluate the rejected prompt-column challenger only if its transferred artifact
and integrity report validate.

Splits:

- selected synthetic training set;
- locked synthetic holdout;
- complete FLEURS Slovenian test gate;
- ARTUR-J public-speech gate.

Report normalized and raw corpus WER/CER, mean and median utterance WER/CER,
empty-hypothesis count, wall time, real-time factor, and peak VRAM separately
per split. Never combine synthetic and real metrics.

## Promotion Criteria

An arm is eligible only when all are true:

- base-parameter integrity passes;
- synthetic holdout normalized corpus WER or CER improves by at least 10%
  relative;
- FLEURS normalized corpus WER does not regress by more than 1.0 absolute point;
- ARTUR-J normalized corpus WER does not regress by more than 1.0 absolute
  point;
- FLEURS normalized corpus CER does not regress by more than 1.5 absolute
  points;
- ARTUR-J normalized corpus CER does not regress by more than 1.5 absolute
  points;
- empty-hypothesis count does not increase on either real gate;
- at least one real gate improves by either 1.0 absolute WER point or 1.5
  absolute CER points.

If neither rank passes, no adapter becomes an accepted parent.

## Scientific Classification

Use exactly one:

- `SL_RESIDUAL_GENERALIZATION_SUPPORTED`;
- `SL_RESIDUAL_SYNTHETIC_ONLY`;
- `SL_RESIDUAL_NOT_SUPPORTED`;
- `EXPERIMENT_INVALID`.

## Required Evidence

Run:

```bash
export CUDA_VISIBLE_DEVICES=1

.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
```

Run real GPU training and all four evaluation splits for both ranks.

## Non-Goals

Do not generate new corpus text, run GaMS, train the prompt column or original
prompt kernel, train encoder/decoder/joint weights, change tokenizer, use real
speech for training, use synthetic holdout for training or steering, publish an
adapter or model, publish generated audio, redesign CI, or create a service API.
