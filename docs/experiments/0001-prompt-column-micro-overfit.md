# Experiment 0001: Prompt-Column Micro-Overfit

Status: **completed for the M3 micro-experiment**

## Purpose

This experiment tests the smallest Slovenian adaptation surface for Nemotron 3.5
ASR Streaming: an additive delta equivalent to changing only the `sl-SI` prompt
input column of the first prompt-projection linear layer.

The experiment does not train the encoder, decoder, joint network, tokenizer,
full prompt kernel, or any non-Slovenian prompt parameter.

## Configuration

- Experiment config:
  [`configs/experiments/prompt_column_micro_overfit.json`](../../configs/experiments/prompt_column_micro_overfit.json)
- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Base checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Piper revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Voice revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Hardware target: one RTX 2080 Ti selected with `CUDA_VISIBLE_DEVICES=0`
- Precision: FP32 after FP16 AMP produced loss-scale overflow events in the
  one-sample proof
- Repository commit tested before this PR commit:
  `1badea5283cb4a7d6080971b8ec4db65ee35d668`
- Effective trainable parameter count: 2048
- Derived prompt index: 62
- Derived encoder width: 1024
- Derived selected column: 1086
- First prompt linear shape: `[2048, 1152]`

## Data IDs

Phase A one-sample proof:

| Role | ID |
|---|---|
| synthetic training | `piper-smoke-0007` |

Phase B synthetic micro-training set:

| Role | IDs |
|---|---|
| synthetic training | `piper-smoke-0001`, `piper-smoke-0003`, `piper-smoke-0004`, `piper-smoke-0005`, `piper-smoke-0007`, `piper-smoke-0008` |

Diagnostic holdout:

| Role | IDs |
|---|---|
| synthetic holdout | `piper-smoke-0002`, `piper-smoke-0006` |
| public real smoke | `fleurs-sl-si-smoke` |

The real public smoke sample is never used for training.

## Planned Commands

```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH:-}"
.venv/bin/python scripts/train_prompt_column.py
.venv/bin/python scripts/evaluate_prompt_column_experiment.py
```

## Results

Generated manifests, deltas, checkpoints, logs, and per-utterance JSON remain
ignored under `runs/experiments/prompt-column-micro-overfit-0001/`.

### Runtime Repair

The first FP16 AMP attempt reached the RNNT loss path but recorded loss-scale
overflow events. The run was repeated in FP32. The local `.venv` also required
the CUDA 12.6 NVCC/NVVM wheel and a Numba/llvmlite downgrade so NeMo's Numba
RNNT CUDA kernels did not pick up the host CUDA 13.3 compiler path.

Local runtime versions for the successful run:

| Component | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu126 |
| CUDA runtime | 12.6 |
| NeMo | 3.1.0+8044a3924b |
| Numba | 0.61.2 |
| NumPy | 2.2.6 |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Driver | 580.167.08 |

### Phase A

| Learning rate | Steps | Initial loss | Final loss | Reduction | Base hypothesis | Adapted hypothesis | Classification |
|---:|---:|---:|---:|---:|---|---|---|
| 0.01 | 200 | 32.590385 | 1.631234 | 94.995% | empty | `Kratek stavek za hiter preizkus.` | Supported |

### Phase B

| Executed | Learning rate | Steps | Initial loss | Final loss | Reduction | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| yes | 0.01 | 500 | 45.908367 | 15.699705 | 65.802% | 2678.4 MiB |

### Integrity

| Check | Result |
|---|---|
| Changed tensors | `prompt_kernel.0.weight` |
| Selected column changed | yes |
| Other columns bitwise identical | yes |
| Bias bitwise identical | yes |
| Unexpected changed tensors | none |
| Unexpected changed elements | 0 |
| Restored adapted checkpoint | passed |

### Streaming Evaluation

All rows used context `[56,3]`, `target_lang=sl-SI`, batch size 1, and GPU 0.
Metrics are raw and are not benchmark claims.

| Split | Base WER | Adapted WER | Base CER | Adapted CER | Base empty | Adapted empty |
|---|---:|---:|---:|---:|---:|---:|
| synthetic training | 92.5 | 38.333 | 67.286 | 11.685 | 3 | 0 |
| synthetic holdout | 87.5 | 87.5 | 66.184 | 43.961 | 0 | 0 |
| public real smoke | 75.0 | 85.0 | 34.286 | 61.905 | 0 | 0 |

Synthetic training improved by 54.167 absolute WER points and by 58.559%
relative WER. The count of empty synthetic-training hypotheses did not
increase; it dropped from 3 to 0.

Synthetic holdout WER was unchanged. The public real-smoke diagnostic regressed
and must not be presented as an accepted real-speech gate.

## Scientific Classification

`PROMPT_COLUMN_SUPPORTED`

This classification applies only to the micro-experiment success criteria. It
does not make a broader Slovenian quality claim and does not make the adapted
checkpoint an accepted parent.

## Limitations

- This is a micro-overfit proof, not a Slovenian benchmark.
- Synthetic and real-smoke metrics remain separate.
- The work order does not authorize model publication.
- The real public smoke sample regressed.
- The next adaptation step needs stronger real-speech gates before accepting
  any challenger.
