# Work Order 0020: Corpus-v2 Prompt-column Diagnostic

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

Expected current main at issuance:

```text
e311184d3aa4fab1dd4a6aad61f8de5de646586f
```

Branch:

```text
exp/corpus-v2-prompt-column-diagnostic
```

Pull request title:

```text
exp: run corpus-v2 prompt-column diagnostic
```

## Scope

Run a named, narrow `DIAGNOSTIC_ONLY` exception for the exact corpus-v2
selected-training manifest. The experiment trains only the 2,048-value
`sl-SI` prompt-column delta, evaluates aggregate behavior on selected
synthetic training, independent synthetic holdout, FLEURS-v2, and ARTUR-J, and
decides whether the prompt-column signal is real-gate useful, synthetic-only,
or unsupported.

The work order requires two commits:

1. `docs: authorize corpus-v2 prompt-column diagnostic`
2. `exp: run corpus-v2 prompt-column diagnostic`

Training may start only after the first commit exists, the diagnostic
certificate is tracked in Git, the certificate matches `HEAD` exactly, and the
authorization checks pass.

## Diagnostic Exception

The selected-training data is authorized for this experiment only as:

```text
DIAGNOSTIC_ONLY
```

This is not `TRAINING_ELIGIBLE`, does not amend the training-data constitution,
and cannot be reused by another training experiment.

## Fixed Identities

- Selected-training certificate:
  `a561ee4c76ddbc5baacca1d5f10aa3beb1749dded7f2f6a1b8fd0e893ab79602`
- Selected-training manifest:
  `84e10587af184be92571ab84e3bd58cd676866e2bd944534c759f0fc9a07fa13`
- Selected-training audio manifest:
  `4fe8ab008dd9725c65da510ed801a46299e1c03db0c00cb3fbf5dea40ff0be7b`
- Synthetic holdout text:
  `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`
- Synthetic holdout audio manifest:
  `7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d`
- Base checkpoint:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision:
  `8044a3924bfcfe8ef71d792bb73bf274fe853575`

Any mismatch invalidates the experiment.

## Hardware

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0
```

PyTorch must see exactly one logical CUDA device, `cuda:0`, backed by physical
GPU selector `1`, an NVIDIA A100-SXM4-80GB. Training and evaluation are FP32
with TF32 disabled. CPU or disk model offload, multiple GPUs, DDP, FSDP, NCCL,
DeepSpeed, model sharding, AMP, FP16, and BF16 are out of scope.

## Required Execution

Authorization phase:

```bash
.venv/bin/python scripts/authorize_corpus_v2_diagnostic_training.py \
  --selected-certificate docs/data-certificates/sl-corpus-v2-selected-training-v1.json \
  --experiment-config configs/experiments/corpus_v2_prompt_column_diagnostic_v1.json \
  --work-order-id 0020 \
  --require-status DIAGNOSTIC_ONLY
```

GPU phases:

```bash
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage verify
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage benchmark-batch
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage train-reference
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage train-batched
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage evaluate
.venv/bin/python scripts/run_corpus_v2_prompt_column_diagnostic.py --stage summarize
```

Repository verification:

```bash
.venv/bin/python -m unittest tests.test_corpus_v2_training
.venv/bin/python -m unittest tests.test_prompt_column
.venv/bin/python -m unittest tests.test_batched_streaming
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
git ls-files | grep -E '\.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan|csv|tsv)$' || true
```

## Non-goals

Do not issue `TRAINING_ELIGIBLE`, accept any checkpoint as a parent, publish a
delta or checkpoint, run GaMS or Piper, alter text/audio, use the holdout for
steering, train residual adapters or shared prompt-kernel parameters, train
decoder/joint/encoder weights, change tokenizer, use evaluation batch size
above 1, use GPUs 0, 2, or 3, run multi-GPU training, or merge the PR.

## Completion Criteria

- The `DIAGNOSTIC_ONLY` certificate is committed before model training.
- Batch-size-1 reference training completes.
- The minibatch arm completes or is honestly unavailable.
- Both arms start independently from the untouched base.
- Fixed-probe losses are comparable.
- Parameter-integrity checks pass.
- All four evaluation splits are measured for every valid arm.
- Scientific and batching classifications are reproducible.
- `accepted_parent` is `none`.
- No raw data, predictions, model, delta, checkpoint, or monitor CSV is
  committed.
