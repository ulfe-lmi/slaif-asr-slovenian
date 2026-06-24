# Work Order 0016: A100 Batched Streaming Evaluation

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/data-policy.md`
- `docs/evaluation-protocol.md`
- `docs/testing-strategy.md`
- `docs/training-plan.md`
- relevant ADRs

Verify current main, open pull requests, and the working tree before editing.

Expected current main at issuance:

```text
8a198ecb1cf93ed05dbce8264128b5cbd6dbbd66
```

## Branch And Pull Request

Branch:

```text
perf/a100-batched-streaming-evaluation
```

Commit:

```text
perf: establish batched A100 evaluation policy
```

Pull request title:

```text
perf: establish batched A100 evaluation policy
```

Open a draft PR. Do not merge it.

## Goal

Build and execute a parity-proven, duration-bucketed, batched Nemotron
streaming inference substrate for one A100.

The work must:

- establish the first valid untouched-base `fleurs-sl-si-test-full-v2` ASR
  baseline;
- determine the highest-throughput batch policy that preserves batch-1
  hypotheses exactly;
- verify the selected policy on `artur-j-public-gate-v1`;
- expose the same substrate for future accepted-candidate ASR scoring;
- measure actual A100 utilization and throughput;
- preserve batch size 1 as the scientific reference mode.

Do not score the corpus-v2 candidate reservoir in this work order.

## Fixed Identities

Base model:

```text
nvidia/nemotron-3.5-asr-streaming-0.6b
```

Base model revision:

```text
3fc30f3e2ae5d78d462441f3ce89dda694f89bd7
```

Checkpoint SHA256:

```text
210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74
```

NeMo revision:

```text
8044a3924bfcfe8ef71d792bb73bf274fe853575
```

FLEURS v2 gate:

```text
fleurs-sl-si-test-full-v2
```

FLEURS v2 manifest SHA256:

```text
8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513
```

Expected FLEURS rows: `834`

ARTUR-J gate:

```text
artur-j-public-gate-v1
```

ARTUR-J manifest SHA256:

```text
66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763
```

Expected ARTUR-J rows: `256`

Streaming context: `[56,3]`

Target language: `sl-SI`

Any identity mismatch is `EXPERIMENT_INVALID`.

## Hardware Policy

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0
```

Required:

- physical GPU selector 1;
- exactly one visible CUDA device;
- logical device `cuda:0`;
- `NVIDIA A100-SXM4-80GB`;
- FP32 cache-aware inference;
- TF32 disabled;
- no AMP, FP16 or BF16;
- no CPU or disk offload;
- no multi-GPU execution;
- no DDP, FSDP, NCCL, DeepSpeed or model sharding.

## Required Implementation

Add reusable project-owned code for:

- manifest validation;
- deterministic duration bucketing by `(duration, sample_id)`;
- child-process execution through the pinned official NeMo cache-aware runner;
- output parsing and sample-ID association;
- restoration to source-manifest order;
- parity checks;
- GPU monitoring through `nvidia-smi`;
- aggregate privacy-safe reporting.

Do not copy or vendor NeMo inference source.

## Required Execution

Run:

```bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0

.venv/bin/python scripts/run_a100_batched_streaming_benchmark.py --stage verify
.venv/bin/python scripts/run_a100_batched_streaming_benchmark.py --stage official-parity
.venv/bin/python scripts/run_a100_batched_streaming_benchmark.py --stage sweep
.venv/bin/python scripts/run_a100_batched_streaming_benchmark.py --stage validate-selected
.venv/bin/python scripts/run_a100_batched_streaming_benchmark.py --stage summarize
```

Then run:

```bash
.venv/bin/python -m unittest tests.test_batched_streaming
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
git ls-files | grep -E '\.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan|csv|tsv)$' || true
```

## Non-Goals

Do not:

- run GaMS;
- synthesize more Piper audio;
- change Piper concurrency policy;
- score the corpus-v2 candidate reservoir;
- generate a synthetic holdout;
- select training examples;
- issue `TRAINING_ELIGIBLE`;
- train a prompt column or adapter;
- change model parameters;
- rerun rejected historical adapters;
- change context `[56,3]`;
- enable TF32, AMP, FP16 or BF16;
- use multiple GPUs;
- publish data or model artifacts;
- merge the PR.

## Definition Of Done

The PR is ready for strategic review only when:

- old and new batch-1 paths have exact parity;
- the complete FLEURS-v2 sweep is recorded;
- a common FLEURS/ARTUR batch is selected or batch 1 is explicitly retained;
- selected-batch hypotheses exactly match batch 1;
- valid untouched-base FLEURS-v2 metrics are committed;
- GPU utilization and throughput evidence is complete;
- duration bucketing is evaluated;
- future scoring can reuse the substrate;
- no protected raw outputs are committed;
- all repository checks pass;
- the PR remains unmerged.
