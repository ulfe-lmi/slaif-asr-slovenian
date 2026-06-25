# Work Order 0022: Corpus-v2 Slovenian Joint-adapter Diagnostic

Status: completed in PR; pending strategic review

## Scope

This work order authorizes one named `DIAGNOSTIC_ONLY` experiment for
`ulfe-lmi/slaif-asr-slovenian`.

The diagnostic tests whether a small Slovenian-specific NeMo-native
`LinearAdapter` in the frozen RNNT joint hidden representation can improve the
corpus-v2 synthetic diagnostics or reduce the real-gate regression observed in
the clean prompt-column diagnostic.

## Preconditions

- PR #25 must be merged before branch creation.
- `main` must contain ancestor
  `a143e64a82e35a1779e0c834dd1ec77b6639d2f0`.
- The branch is `exp/corpus-v2-slovenian-joint-adapter`.
- The pull request title is
  `exp: test frozen-base Slovenian joint adapter`.

## Authorization

Commit 1 must create and commit the `DIAGNOSTIC_ONLY` certificate before any
Nemotron model loading or training:

```text
docs: authorize Slovenian joint-adapter diagnostic
```

The certificate is:

```text
docs/data-certificates/sl-corpus-v2-joint-adapter-diagnostic-v1.json
```

Training may begin only after that commit is tracked, pushed, matches `HEAD`,
and passes authorization tests plus repository checks.

## Fixed Inputs

- Selected-training certificate:
  `docs/data-certificates/sl-corpus-v2-selected-training-v1.json`
- Selected-training certificate SHA256:
  `a561ee4c76ddbc5baacca1d5f10aa3beb1749dded7f2f6a1b8fd0e893ab79602`
- Selected-training manifest SHA256:
  `84e10587af184be92571ab84e3bd58cd676866e2bd944534c759f0fc9a07fa13`
- Selected-training audio manifest SHA256:
  `4fe8ab008dd9725c65da510ed801a46299e1c03db0c00cb3fbf5dea40ff0be7b`
- Selected rows: 160, composed of 120 hard rows and 40 controls.
- Synthetic-holdout text SHA256:
  `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`
- Synthetic-holdout audio manifest SHA256:
  `7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d`
- Synthetic holdout rows: 96.

The holdout must not enter training, steering, early stopping, adapter
selection, or hyperparameter selection.

## Model and Runtime

- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Model revision:
  `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision:
  `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Target language: `sl-SI`
- Streaming context: `[56,3]`

GPU execution must use only:

```text
CUDA_VISIBLE_DEVICES=1
NVIDIA_TF32_OVERRIDE=0
PYTHONUNBUFFERED=1
```

PyTorch must see one logical device, `cuda:0`, backed by physical
`NVIDIA A100-SXM4-80GB` GPU 1. FP32 and TF32-disabled execution are required.
CPU or disk model offload, multiple GPUs, DDP, FSDP, NCCL, DeepSpeed, and model
sharding are prohibited.

## Trainable Surface

Install one NeMo-native adapter on `model.joint`:

- Adapter name: `sl-si-joint-adapter-v1`
- Type: `nemo.collections.common.parts.adapter_modules.LinearAdapter`
- Strategy: `ResidualAddAdapterStrategy`
- Bottleneck dimension: 32
- Activation: `swish`
- Normalization position: `pre`
- Dropout: 0.0
- Stochastic depth: 0.0
- L2 auxiliary penalty: 0.0

Every pretrained tensor must remain frozen and bitwise unchanged. Only named
adapter tensors may be trainable.

The exact trainable-parameter count is derived at runtime from:

```text
2 * joint_hidden * 32 + 2 * joint_hidden
```

## Training Protocol

Run exactly one arm:

```text
sl_si_joint_adapter_dim32
```

Settings:

- clean original Piper selected-training audio;
- batch size: 8;
- duration-compatible batching;
- epochs: 12;
- rows per epoch: 160;
- sample exposures: 1920;
- optimizer steps: 240;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0;
- scheduler: none;
- gradient accumulation: none;
- gradient clipping: none;
- seed: 1234;
- precision: FP32;
- TF32: disabled;
- SpecAugment: disabled;
- waveform augmentation: none.

No prompt-column arm, residual-adapter rank sweep, learning-rate search, early
stop, GaMS, Piper, or speaker-range augmentation is authorized.

## Progress

Long-running stages must emit privacy-safe progress to stderr and local ignored
NDJSON. No stage may remain silent for more than 10 seconds. Progress must not
include raw text, references, hypotheses, candidate IDs, holdout IDs, or local
absolute paths.

## Evaluation

Evaluate the trained joint adapter, with the adapter explicitly enabled for
`sl-SI`, on:

- selected synthetic training;
- independent synthetic holdout;
- FLEURS-v2 full test;
- ARTUR-J.

Evaluation must use batch size 1, no duration bucketing, context `[56,3]`,
FP32, and TF32 disabled.

Use committed untouched-base and Experiment 0008 clean prompt-column metrics
for comparison. Do not rerun those reference models.

## Decisions

Use exactly one scientific classification:

- `SL_JOINT_ADAPTER_REAL_GAIN_DIAGNOSTIC`
- `SL_JOINT_ADAPTER_MITIGATES_PROMPT_REGRESSION`
- `SL_JOINT_ADAPTER_SYNTHETIC_ONLY`
- `SL_JOINT_ADAPTER_NOT_SUPPORTED`
- `EXPERIMENT_INVALID`

Regardless of metrics:

```text
accepted_parent = none
```

The untouched Nemotron checkpoint remains the only accepted parent.

## Result

The experiment completed in this PR. The adapter was added only to
`model.joint`, the exact runtime trainable count was 42,240 parameters, and
all pretrained Nemotron tensors remained frozen and bitwise identical. The run
used the original clean Piper selected-training audio, no augmentation, fixed
batch size 8 for training, and batch size 1 for evaluation. Shared
privacy-safe live progress was emitted during long training and evaluation
stages.

The trained joint adapter improved the independent synthetic holdout relative
to the untouched base but regressed both real gates. The scientific
classification is `SL_JOINT_ADAPTER_SYNTHETIC_ONLY`; `accepted_parent` remains
`none`, and no adapter or checkpoint is promoted.

## Required Commands

Authorization:

```text
.venv/bin/python scripts/authorize_corpus_v2_joint_adapter_diagnostic.py \
  --work-order-id 0022 \
  --selected-certificate docs/data-certificates/sl-corpus-v2-selected-training-v1.json \
  --adapter-config configs/adapters/sl_si_joint_adapter_v1.json \
  --experiment-config configs/experiments/corpus_v2_slovenian_joint_adapter_v1.json \
  --require-status DIAGNOSTIC_ONLY
```

GPU execution:

```text
.venv/bin/python -u scripts/run_corpus_v2_joint_adapter_diagnostic.py --stage verify --progress-interval-seconds 5
.venv/bin/python -u scripts/run_corpus_v2_joint_adapter_diagnostic.py --stage train --progress-interval-seconds 5
.venv/bin/python -u scripts/run_corpus_v2_joint_adapter_diagnostic.py --stage evaluate --progress-interval-seconds 5
.venv/bin/python -u scripts/run_corpus_v2_joint_adapter_diagnostic.py --stage summarize
```

Repository verification:

```text
.venv/bin/python -m unittest tests.test_slovenian_joint_adapter
.venv/bin/python -m unittest tests.test_live_progress
.venv/bin/python -m unittest tests.test_corpus_v2_training
.venv/bin/python -m unittest tests.test_batched_streaming
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
```

## Non-goals

Do not run GaMS or Piper, regenerate text or audio, use speaker-range variants,
change selected-training membership, benchmark batch sizes, tune rank or
learning rate, train prompt-column parameters, train encoder/prompt-kernel/
decoder/joint base weights, add more than one adapter, issue
`TRAINING_ELIGIBLE`, accept or publish an adapter, use GPUs 0/2/3, or merge the
PR.
