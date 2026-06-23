# Experiment 0005: Slovenian Residual-Adapter Proof

Status: **completed; synthetic-only; no accepted adapter**

> **FLEURS v1 deprecation:** this report used historical
> `fleurs-sl-si-test-full-v1` metrics. That gate is now deprecated because
> repeated upstream FLEURS source IDs caused duplicate sample IDs and WAV
> overwrites; its 834 rows represented only 347 unique sample identities. The
> FLEURS numbers below are preserved for auditability but must not be used as
> complete-split quality evidence. ARTUR-J independently failed promotion and
> remains unaffected.

## Purpose

This experiment tests whether a Slovenian-specific residual adapter with more
capacity than the 2048-scalar prompt-column delta can generalize from the same
project-generated synthetic Round 1 corpus to the fixed real Slovenian gates.

No new corpus was generated, GaMS was not run, and no real speech entered
training. The experiment isolates trainable-surface capacity while keeping the
accepted base checkpoint, Piper audio, corpus hashes, training order,
optimizer, synthetic holdout, and real gates fixed.

## Configuration

- Experiment config:
  [`configs/experiments/slovenian_residual_adapter_proof.json`](../../configs/experiments/slovenian_residual_adapter_proof.json)
- Work order:
  [`docs/work-orders/0009-slovenian-residual-adapter-proof.md`](../work-orders/0009-slovenian-residual-adapter-proof.md)
- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Base checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Hardware: one NVIDIA A100-SXM4-80GB selected with
  `CUDA_VISIBLE_DEVICES=1`; PyTorch saw one logical device, `cuda:0`
- Precision: FP32
- TF32: disabled
- Prompt mode: `langID`
- Target prompt: `sl-SI`

## Adapter Surface

The adapter wraps the frozen prompt kernel with a Slovenian-gated residual path:

```text
base_output = frozen_prompt_kernel(inputs)
sl_active = inputs[..., selected_prompt_column].unsqueeze(-1)
residual = up(GELU(down(base_output)))
output = base_output + sl_active * residual
```

The selected prompt index, encoder width, selected input column, and prompt
kernel output width are derived at runtime. For this checkpoint they resolved
to:

| Field | Value |
|---|---:|
| `sl-SI` prompt index | 62 |
| Encoder width | 1024 |
| Selected first-projection column | 1086 |
| Prompt-kernel output width | 1024 |

The output projection is zero-initialized, so step-zero wrapped output matches
the untouched base within tolerance. The residual is exactly inactive for
non-`sl-SI` prompt columns.

## Data Integrity

| Artifact | SHA256 |
|---|---|
| Candidate pool | `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17` |
| Synthetic holdout | `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9` |
| Selected training manifest | `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4` |
| FLEURS manifest (deprecated v1) | `07838a58222b9a0f6a4f4639b66d678ee38f87254518e43b742a143ef4aeaf4e` |
| ARTUR-J manifest | `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763` |

The transferred manifests were verified against the committed hashes first,
then localized into ignored run storage because the transferred files retained
their original host absolute paths. The localized copies are ignored local
artifacts and were used only for execution on the current machine.

| Dataset | Rows | Unique sample IDs |
|---|---:|---:|
| Selected synthetic training | 160 | 160 |
| Fixed synthetic holdout | 96 | 96 |
| FLEURS full test (deprecated v1) | 834 | 347 |
| ARTUR-J gate | 256 | 256 |

The synthetic holdout, FLEURS, and ARTUR-J did not enter training, selection, or
steering.

## Training

Both arms started independently from the untouched accepted base checkpoint.

| Rank | Trainable parameters | LR | Steps | Initial loss | Final loss | Loss reduction | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32768 | 0.001 | 1800 | 98.540 | 12.888 | 86.920% | 3582.4 MiB |
| 64 | 131072 | 0.001 | 1800 | 98.540 | 9.552 | 90.306% | 3583.5 MiB |

No fallback learning-rate arm was needed because both arms exceeded the
precommitted loss-reduction check.

## Parameter Integrity

| Arm | Base tensors identical | Prompt kernel identical | Encoder identical | Decoder/joint identical | Tokenizer/config identical |
|---|---|---|---|---|---|
| rank 16 | passed | passed | passed | passed | passed |
| rank 64 | passed | passed | passed | passed | passed |

Only adapter tensors changed from initialization. No pretrained Nemotron tensor
changed. The adapter artifacts, integrity reports, predictions, and raw logs
remain ignored local artifacts.

The rejected prompt-column Round 1 artifact was present in transferred local
storage, but its older integrity record does not carry the full base-identity
metadata required by this work order's adapter comparison. It was not rerun or
used for promotion; experiment 0004 remains the committed prompt-column
diagnostic.

## Metrics

Normalized metrics use `sl-asr-normalization-v1`. Corpus WER/CER are computed
from summed edit counts, not averaged utterance percentages. Mean and median
utterance metrics are named separately.

### Selected Synthetic Training

| Model | Normalized corpus WER | Normalized corpus CER | Mean utt. WER | Median utt. WER | Empty hypotheses |
|---|---:|---:|---:|---:|---:|
| base | 89.070 | 62.622 | 89.157 | 94.118 | 35 |
| rank 16 | 19.340 | 5.613 | 19.394 | 17.647 | 0 |
| rank 64 | 12.633 | 3.264 | 12.645 | 11.765 | 0 |

### Fixed Synthetic Holdout

| Model | Normalized corpus WER | Normalized corpus CER | Mean utt. WER | Median utt. WER | Empty hypotheses |
|---|---:|---:|---:|---:|---:|
| base | 77.563 | 39.092 | 77.507 | 77.778 | 20 |
| rank 16 | 63.926 | 30.183 | 63.779 | 63.636 | 5 |
| rank 64 | 54.836 | 22.513 | 55.165 | 52.273 | 0 |

### FLEURS Full Test (Deprecated v1)

| Model | Normalized corpus WER | Normalized corpus CER | Mean utt. WER | Median utt. WER | Empty hypotheses |
|---|---:|---:|---:|---:|---:|
| base | 52.734 | 16.423 | 53.541 | 52.941 | 0 |
| rank 16 | 67.076 | 23.741 | 67.884 | 68.019 | 0 |
| rank 64 | 70.430 | 26.248 | 71.010 | 70.833 | 0 |

### ARTUR-J Gate

| Model | Normalized corpus WER | Normalized corpus CER | Mean utt. WER | Median utt. WER | Empty hypotheses |
|---|---:|---:|---:|---:|---:|
| base | 67.453 | 29.016 | 76.555 | 75.000 | 12 |
| rank 16 | 78.943 | 36.483 | 86.813 | 83.333 | 8 |
| rank 64 | 81.739 | 36.108 | 89.527 | 85.714 | 0 |

## Runtime Performance

| Model | Split/gate | Wall time | Real-time factor | Peak VRAM |
|---|---|---:|---:|---:|
| base | selected synthetic training | 146.169 s | 0.0991 | 2548.1 MiB |
| base | fixed synthetic holdout | 44.486 s | 0.1048 | 2572.2 MiB |
| base | FLEURS | 780.913 s | 0.0955 | 2596.8 MiB |
| base | ARTUR-J | 102.531 s | 0.0977 | 2621.0 MiB |
| rank 16 | selected synthetic training | 141.445 s | 0.0959 | 2645.9 MiB |
| rank 16 | fixed synthetic holdout | 43.026 s | 0.1014 | 2670.1 MiB |
| rank 16 | FLEURS | 784.774 s | 0.0960 | 2694.7 MiB |
| rank 16 | ARTUR-J | 105.075 s | 0.1001 | 2710.7 MiB |
| rank 64 | selected synthetic training | 145.039 s | 0.0984 | 2719.4 MiB |
| rank 64 | fixed synthetic holdout | 43.656 s | 0.1029 | 2720.3 MiB |
| rank 64 | FLEURS | 789.178 s | 0.0966 | 2720.6 MiB |
| rank 64 | ARTUR-J | 104.636 s | 0.0997 | 2720.3 MiB |

## Promotion Decision

| Rank | Decision | Reasons |
|---:|---|---|
| 16 | rejected | historical FLEURS-v1 WER/CER regression beyond threshold; ARTUR-J WER/CER regression beyond threshold |
| 64 | rejected | historical FLEURS-v1 WER/CER regression beyond threshold; ARTUR-J WER/CER regression beyond threshold |

Selected adapter: none

Accepted parent: none

Scientific conclusion:

```text
SL_RESIDUAL_SYNTHETIC_ONLY
```

Both residual adapters materially improved the fixed synthetic holdout, and
rank 64 improved it more than rank 16. Neither adapter passed the real-gate
promotion policy. The larger trainable surface increased synthetic memorization
and synthetic-holdout fit but did not transfer to ARTUR-J; the historical
FLEURS-v1 component is deprecated.

## Interpretation

This experiment supports only a synthetic-capacity claim: the residual adapter
can fit the transferred project-generated synthetic corpus better than the
2048-scalar prompt column. It does not support accepting a Slovenian adapter,
because both real gates regressed substantially.

The result suggests that simply increasing prompt-side capacity against the same
single-voice synthetic corpus is not sufficient. A next controlled experiment
should address acoustic and TTS diversity or carefully partitioned real
Slovenian training data before increasing trainable scope further.

## Safety

- Only one logical CUDA device was visible to PyTorch.
- Physical GPU selector was `CUDA_VISIBLE_DEVICES=1`.
- No model was intentionally loaded on CPU.
- No CPU or disk offload was used.
- Every pretrained Nemotron parameter remained frozen and bitwise identical.
- No real speech entered training.
- No synthetic holdout entered training or steering.
- No FLEURS or ARTUR text was sent to a generator.
- GaMS was not executed.
- No new corpus was generated.
- No model, adapter, audio, checkpoint, local manifest, or raw output is
  committed.
- No model or dataset was published.

## Limitations

- The experiment uses one synthetic corpus and one Piper voice. It does not
  test broader speaker, acoustic, or TTS diversity.
- The rejected prompt-column challenger was not promoted and is not a valid
  parent.
- The real gates are development gates, not final blind tests or release
  criteria.
- No production quality, release, or benchmark claim is made.

## Next Recommendation

Do not accept either residual adapter as a parent. If synthetic-only behavior is
the dominant pattern, prioritize TTS/acoustic diversity or a carefully governed
real Slovenian training partition before broadening the model surface further.
