# Experiment 0028: Surface07 Canonical Batch-1 Evaluation

Classification: `CANONICAL_SURFACE07_CONFIRMED_NEW_BEST`

This evaluation-only proof compares named, preselected challengers under the scientific reference protocol. It did not train, tune, or reselect any checkpoint.

## Candidate Inventory

| Candidate | Available | SHA256 | Source experiment | Evaluated |
|---|---:|---|---|---:|
| base | true | `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` | untouched-base | true |
| pr36_round20 | true | `b40b01027383a2c8b6886accbd7504e7b62ffbfa52d5cbcfbbb9553f809d3c9a` | 0017-scale2000-decoder-joint-rnnt-directional | true |
| surface06_round05 | true | `e82bd833fcbb622c73b7acb8f295e18d32fd605fcd49915286fd84ebee19cdf1` | 0026-fixed-scale2000-surface06-last-four-encoder-blocks | true |
| surface07_round13 | true | `349d06dd517b6e99b71a74f15a04d6020afe56223ef946014d3bdca1440706b0` | 0027-fixed-scale2000-surface07-topencoder-fusion | true |

## Canonical Metrics

| Split | Base canonical WER/CER/empty | PR #36 canonical WER/CER/empty | Surface06 canonical WER/CER/empty | Surface07 canonical WER/CER/empty |
|---|---:|---:|---:|---:|
| fleurs_v2 | 52.703 / 16.423 / 1 | 46.219 / 15.617 / 0 | 44.512 / 13.533 / 0 | 42.090 / 12.988 / 0 |
| artur_j | 67.453 / 29.016 / 12 | 57.055 / 20.432 / 0 | 50.939 / 16.093 / 0 | 47.532 / 15.025 / 0 |

Values are normalized corpus WER / CER / empty-hypothesis count using `sl-asr-normalization-v1`.

## Directional Context

| Split | PR #36 directional | Surface06 directional | Surface07 directional | Canonical interpretation |
|---|---:|---:|---:|---|
| fleurs_v2 | 46.195 / 15.604 / 0 | 44.506 / 13.528 / 0 | 42.084 / 12.985 / 0 | Canonical batch-1 ordering is reported above; directional metrics did not alter candidate selection. |
| artur_j | 56.793 / 20.177 / 0 | 50.590 / 15.803 / 0 | 47.357 / 14.805 / 0 | Canonical batch-1 ordering is reported above; directional metrics did not alter candidate selection. |

## Protocol

- Policy: `single-gpu-canonical-batch1-v1`.
- Batch size: 1.
- Duration bucketing: false.
- Precision: fp32.
- TF32: false.
- Target language: `sl-SI`.
- Attention context: `[56, 3]`.
- Normalization: `sl-asr-normalization-v1`.
- GPU: NVIDIA GeForce RTX 3090; one visible CUDA device.

## Boundaries

- `accepted_parent` remains `none`.
- Promotion eligibility and `TRAINING_ELIGIBLE` remain false.
- No controller-development partition was loaded.
- No checkpoint was accepted and no model was published.
- Checkpoints, predictions, manifests, logs, raw references, and raw hypotheses remain ignored local artifacts.
