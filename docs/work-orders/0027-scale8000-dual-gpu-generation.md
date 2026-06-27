# Work Order 0027: Scale-8000 Dual-GPU Synthetic Generation

Status: `ENVIRONMENT_BLOCKED`

Branch: `exp/scale8000-dual-gpu-generation`

## Goal

Run the canonical repository validation pass and prepare a dual-GPU scale-8000
synthetic dataset generation path that includes the reconciled scale-2000 corpus
by construction.

Scale-8000 follows the existing project convention where the scale name denotes
the exposure multiplier against the original 160-item reference. The intended
dataset therefore contains:

- 64,000 semantic rows;
- 576,000 clean voice views;
- 704,000 augmented views;
- 1,280,000 total view/exposure records.

This is dataset construction evidence only. It does not issue
`TRAINING_ELIGIBLE`, does not accept a model parent, and does not authorize a
public quality claim.

## Parent Evidence

The parent subset is the scale-2000 corpus:

- corpus: `sl-corpus-v4-gams-16000-training-v1`;
- text SHA256:
  `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`;
- rows: 16,000;
- audio certificate:
  `docs/data-certificates/sl-corpus-v4-scale2000-audio-v1.json`;
- directional report:
  `docs/experiments/0014-gams16000-scale2000-text-only-directional.json`.

The scale-8000 design preserves scale-2000 as a prefix: all inherited rows must
remain byte-for-byte unchanged and appear before newly generated rows in the
combined local corpus.

## Canonical Pass

The canonical repository pass was run before generation planning. Commands were
recorded in the public data report. The first wrapper invocation of
`py_compile` was invalid because newline-separated file names were embedded in
a shell string; the required command was rerun directly and passed.

No canonical-pass result blocked generation. The only blocker was storage
preflight.

## Dual-GPU Plan

The work order authorizes two independent single-visible-GPU workers:

- worker `gpu0`: `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`;
- worker `gpu1`: `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`.

The planned generation work contains 1,200 prompt-shard tasks split evenly:

- `gpu0`: 600 tasks, 36,000 requested rows;
- `gpu1`: 600 tasks, 36,000 requested rows.

Both workers write only to ignored local staging paths. No worker writes
directly to tracked report or certificate paths.

## Storage Preflight

The resource preflight found:

- inherited scale-2000 data size: 82,341,008,788 bytes;
- projected new scale-8000 data size: 247,023,026,364 bytes;
- required free bytes with 25% safety margin: 308,778,782,955 bytes;
- available bytes: 211,107,655,680 bytes.

Generation did not start because available storage did not satisfy the required
safety margin.

## Result

Classification: `environment_blocked`

No GaMS, Piper, Supertonic, Nemotron, synthesis, training, evaluation, or corpus
generation stage was started after the storage blocker was detected.

## Safety

- No generated text was committed.
- No generated audio was committed.
- No model, adapter, checkpoint, prediction, local manifest, monitor CSV, or TSV
  artifact was committed.
- `TRAINING_ELIGIBLE` was not issued.
- No model was promoted.
