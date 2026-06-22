# Work Order 0008: Slovenian Curriculum Round 1

## Goal

Execute the first project-generated Slovenian curriculum round without GaMS,
external LLM APIs, or any other corpus-generation service.

## Required branch and metadata

- Branch: `exp/slovenian-curriculum-round-1`
- Commit: `exp: run Slovenian curriculum round 1`
- Pull-request title: `exp: run Slovenian curriculum round 1`
- Do not include tool branding in Git or GitHub metadata.

## Parent state

Round 1 starts from current `origin/main` after the real Slovenian evaluation
suite is merged. The accepted parent remains the untouched
`nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint at revision
`3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`.

The ignored micro-overfit checkpoint is not an accepted parent.

## Required data

- Generate and freeze 96 independent synthetic holdout utterances.
- Generate and validate a separate 320-utterance candidate pool in eight local
  batches.
- Keep generated JSONL, audio, local manifests, hypotheses, deltas,
  checkpoints, and raw reports under ignored `runs/` storage.
- Commit only generation specification, hashes, aggregate counts, and
  privacy-safe metrics.

## Required execution

Use only physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`.

Execute sequentially:

1. validate holdout and candidate corpora;
2. synthesize both corpora through Piper on GPU 0;
3. pre-score only the candidate pool with the untouched Nemotron checkpoint;
4. select 120 hard examples and 40 deterministic controls;
5. train only the 2048-scalar `sl-SI` prompt-column delta;
6. prove state-dictionary integrity;
7. evaluate selected synthetic training, fixed synthetic holdout, complete
   FLEURS, and ARTUR-J gates;
8. classify the challenger as accepted, synthetic-only, rejected, or invalid.

## Non-goals

- no GaMS;
- no external LLM;
- no Round 2 generation;
- no prompt-kernel, encoder, decoder, joint, tokenizer, LoRA, or real-speech
  training;
- no GPU 1 or A100 use;
- no generated corpus, audio, checkpoint, delta, or raw output publication;
- no service API, database, CI redesign, model release, or merge.
