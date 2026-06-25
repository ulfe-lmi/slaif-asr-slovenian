# Work Order 0024: Fast Batched Directional Replay

Status: in progress

## Scope

Make the synthetic ASR iteration loop faster by changing only Supertonic
dataset synthesis/conversion/validation throughput and directional ASR
evaluation throughput, then replay Experiment 0011 with unchanged training.

## Fixed Boundaries

- Supertonic synthesis uses native batches of 32 on GPU.
- Conversion and validation use a bounded CPU worker pool.
- Training implementation and protocol remain byte-identical to `main`.
- Training remains the fixed Supertonic frozen-base joint-adapter protocol:
  batch 8, 12 epochs, 1920 sample exposures, 240 optimizer steps, AdamW at
  0.001, FP32, TF32 disabled.
- Evaluation is directional only: batch size 32, duration bucketing enabled,
  no exact transcript parity requirement, and no batch-1 replay.
- Canonical Experiment 0011 evidence is not replaced.

## Protected Training Files

The replay configuration records Git blob SHA and byte SHA256 values for:

- `slaif_asr/corpus_v2_training.py`
- `slaif_asr/slovenian_joint_adapter.py`
- `scripts/run_supertonic3_joint_adapter_diagnostic.py`

The replay verifier checks these values before training and before final
reporting. Any change invalidates the replay.

## Acceptance

The PR is ready for strategic review only when all 1472 Supertonic WAVs are
batch-synthesized and validated, unchanged training completes once, batch-32
directional evaluation runs one combined suite per model, the directional
classification is recorded, no raw data or model artifact is committed, and the
PR remains unmerged.

## Result

Pending execution.
