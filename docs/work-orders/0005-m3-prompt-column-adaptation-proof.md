# Work Order 0005: M3 Prompt-Column Adaptation Proof

## Goal

Begin M3 with the smallest possible Slovenian adaptation proof:

```text
modify only the contribution of the sl-SI prompt column
freeze acoustic, streaming, decoder, joint, tokenizer, and all other prompt parameters
```

This is a real one-GPU training experiment on physical GPU 0 of the RTX 2080 Ti
development host. The experiment must prove or falsify whether a single
Slovenian prompt-column delta can overfit a tiny synthetic Slovenian set without
changing any other checkpoint tensor.

## Required branch and metadata

- Branch: `feat/m3-prompt-column-proof`
- Commit: `feat: add Slovenian prompt-column adaptation proof`
- Pull-request title: `feat: add Slovenian prompt-column adaptation proof`
- Do not include agent or tool branding in Git or GitHub metadata.

## Inputs

- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Piper engine revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Piper voice revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Synthetic candidates: the eight committed Piper smoke candidate IDs.
- Real smoke sample: one ignored, public FLEURS Slovenian sample under CC BY 4.0.

## Required implementation

- Add a project-owned prompt-column delta module.
- Derive the Slovenian prompt index, encoder width, first prompt linear shape,
  and selected input column from the restored checkpoint at runtime.
- Train only an additive delta vector equivalent to changing one first-linear
  input column.
- Reject nonzero weight decay.
- Keep all original model parameters frozen.
- Merge the learned delta into only the selected column, remove the wrapper, and
  save an ignored adapted `.nemo` checkpoint.
- Restore the adapted checkpoint and prove tensor integrity:
  only the selected Slovenian prompt column may differ.
- Evaluate base and adapted checkpoints at `[56,3]` on:
  six synthetic training utterances, two synthetic holdout utterances, and one
  public real smoke utterance.
- Keep pipeline status and recognition status separate.
- Write a privacy-safe committed aggregate experiment report.

## Split

One-sample Phase A:

```text
piper-smoke-0007
```

Six-sample Phase B training set:

```text
piper-smoke-0001
piper-smoke-0003
piper-smoke-0004
piper-smoke-0005
piper-smoke-0007
piper-smoke-0008
```

Synthetic holdout:

```text
piper-smoke-0002
piper-smoke-0006
```

## Non-goals

- no encoder, decoder, joint, tokenizer, prompt-kernel-wide, or non-Slovenian
  prompt training;
- no GaMS integration;
- no large corpus generation;
- no active selection;
- no public model, delta, checkpoint, data, or generated-audio publication;
- no real-speech training;
- no GPU 1 use;
- no A100 use;
- no Docker, Conda, DDP, NCCL, FSDP, DeepSpeed, or model sharding;
- no CI redesign;
- no production or benchmark claim.

## Acceptance criteria

- one RTX 2080 Ti is visible through `CUDA_VISIBLE_DEVICES=0`;
- the model loads on GPU 0;
- prompt index and selected column are derived, not hardcoded;
- effective trainable parameter count is recorded;
- only the delta receives gradients;
- Phase A is classified by predeclared loss and decoding criteria;
- Phase B runs only if Phase A is supported or partially supported;
- saved adapted checkpoint restores on GPU 0;
- integrity report has no unexpected tensor changes;
- base/adapted evaluation tables are written for each split;
- conclusion is one of `PROMPT_COLUMN_SUPPORTED`,
  `PROMPT_COLUMN_PARTIALLY_SUPPORTED`, `PROMPT_COLUMN_NOT_SUPPORTED`, or
  `EXPERIMENT_INVALID`;
- no checkpoint, delta, audio, runtime output, private path, or secret is
  committed.
