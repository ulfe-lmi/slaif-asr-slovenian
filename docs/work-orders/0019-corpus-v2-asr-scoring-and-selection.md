# Work Order 0019: Corpus-v2 ASR Scoring and Selected-Training Construction

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

## Governing instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/data-policy.md`
- `docs/data-quality-validator.md`
- `docs/evaluation-protocol.md`
- `docs/training-plan.md`
- `docs/testing-strategy.md`
- relevant ADRs

Verify live main, open PRs, and the working tree before editing. Expected
current main at issuance:

```text
85fb873dafea190b9c527bff474bf5eb35414993
```

PR #22 is expected to be merged. Live repository state is authoritative.

## Branch and PR

Branch:

```text
feat/corpus-v2-asr-scoring-and-selection
```

Commit:

```text
feat: score corpus-v2 audio and build selected-training manifest
```

PR title:

```text
feat: score corpus-v2 audio and build selected-training manifest
```

Open a draft PR. Do not merge it.

## Goal

Run untouched-base ASR scoring for the accepted synthetic candidate source and
independent synthetic holdout, then build a diversity-constrained
selected-training manifest from the candidate source only.

This PR must stop before model training.

It may produce candidate ASR scoring evidence, holdout ASR scoring evidence,
a selected-training manifest, a selected-training selection report, and a
privacy-safe selected-training readiness certificate.

It must not produce `TRAINING_ELIGIBLE`, model checkpoints, trained adapters,
model promotion, or public performance claims.

## Fixed authorization

Require:

```text
docs/data-certificates/sl-corpus-v2-scoring-authorization-v1.json
```

Expected status:

```text
SCORING_AUTHORIZED
```

Expected SHA256:

```text
42c57975a77594d68cd1b1250a8edc17643bbc254e29642364fc9e4be680664b
```

The certificate authorizes ASR scoring and selected-training construction only.
If missing, stale, hash-mismatched, or any other status, stop with
`EXPERIMENT_INVALID`.

## Candidate source identities

- Corpus ID: `sl-corpus-v2-gams-candidate-reservoir-v1`
- Role: `synthetic_candidate`
- Expected rows/audio: 415
- Accepted text SHA256:
  `b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6`
- Accepted review sidecar SHA256:
  `4dc16336dc9404d48cab196b862e4aa2a4558b20d2728fc954dd7fbb88fed732`
- Audio manifest SHA256:
  `c1d366e1d05b6f728af51b3350556b6d915fabf5a6b584a6aa2f9fdc0df538bc`
- Audio certificate SHA256:
  `25737f59397d5c5acdd99e6af83e1129587199cb1a184eaec43dd27139bb1692`
- Required statuses: `TEXT_ACCEPTED`, `AUDIO_ACCEPTED`

## Holdout identities

- Corpus ID: `sl-corpus-v2-independent-synthetic-holdout-v1`
- Role: `synthetic_holdout`
- Expected rows/audio: 96
- Accepted text SHA256:
  `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`
- Audio manifest SHA256:
  `7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d`
- Audio certificate SHA256:
  `d5c1660b8b11b8b250d04034dfb2abe14a96dda33d48560875a51d7168865297`
- Required statuses: `TEXT_ACCEPTED`, `AUDIO_ACCEPTED`

The holdout must never enter selected training.

## Model/runtime identities

- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Streaming context: `[56,3]`
- Language prompt: `sl-SI`
- Evaluation batch policy: `configs/evaluation/a100_streaming_batch_policy.json`
- Required policy: batch size 1, no duration bucketing, reference mode

Do not override the batch-1 A100 policy.

## Hardware policy

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0
```

Required:

- physical GPU selector 1;
- exactly one visible CUDA device;
- logical device `cuda:0`;
- NVIDIA A100-SXM4-80GB;
- FP32 inference;
- TF32 disabled;
- no AMP, FP16 or BF16;
- no CPU or disk model offload;
- no multi-GPU;
- no DDP/FSDP/NCCL/DeepSpeed/model sharding.

Before model execution, verify that physical GPU 1 is not materially occupied by
an unrelated process. A contaminated run is `ENVIRONMENT_BLOCKED` and must not
produce scoring claims.

## Implementation

Add reusable project-owned code for corpus-v2 scoring and selected-training
construction. Raw predictions, hypotheses, references, manifests and per-row
scores remain ignored local artifacts.

Committed privacy-safe outputs:

- `docs/data-reports/0008-corpus-v2-asr-scoring.md`
- `docs/data-reports/0008-corpus-v2-asr-scoring.json`
- `docs/data-reports/0009-corpus-v2-selected-training.md`
- `docs/data-reports/0009-corpus-v2-selected-training.json`
- `docs/data-certificates/sl-corpus-v2-selected-training-v1.json`
- `docs/experiments/0007-corpus-v2-base-scoring-and-selection.md`
- `docs/experiments/0007-corpus-v2-base-scoring-and-selection.json`

## Scoring requirements

Score both partitions using the untouched base model:

1. candidate source: 415 rows;
2. synthetic holdout: 96 rows.

For each partition verify manifest hash, row count, audio files, audio SHA256,
text hash, prediction association, missing/duplicate/unexpected prediction
absence, corpus WER/CER, normalized corpus WER/CER, mean/median utterance
WER/CER, empty hypotheses, duration, wall time, RTF, GPU utilization
aggregates, and model/runtime identity.

Use normalizer `sl-asr-normalization-v1`. Corpus metrics must use summed edit
counts.

## Candidate selection policy

Build selected training from the 415-row candidate source only.

Target:

- 120 hard examples;
- 40 deterministic controls;
- 160 total.

Hard score:

```text
empty_bonus + normalized_wer + 0.25 * normalized_cer
```

where `empty_bonus` is 1000 for an empty hypothesis and 0 otherwise.

Tie-breakers:

1. higher normalized CER;
2. longer duration;
3. `SHA256(candidate_id)` ascending.

Diversity constraints:

- no duplicate text/audio/family identity;
- maximum 1 selected row per discovered template family;
- maximum 1 selected row per utterance family;
- maximum 5% of selected set from any source family unless unavoidable;
- no domain exceeds 25% of selected rows unless unavoidable;
- every prompt/generation cell with accepted candidates should contribute at
  least 4 selected rows if possible;
- every major domain should have at least 5 selected rows if possible;
- include both short and long utterances.

If strict constraints prevent 120 hard examples, relax only in this order:

1. domain cap;
2. source-family cap;
3. cell minimum;
4. discovered-family cap last.

Never relax holdout exclusion or exact duplicate exclusion.

Controls are selected from remaining candidate rows by deterministic
stratified sampling and must be disjoint from hard examples.

## Selected-training certificate

Commit:

```text
docs/data-certificates/sl-corpus-v2-selected-training-v1.json
```

Certificate status:

```text
SELECTED_TRAINING_MANIFEST_READY
```

Do not use `TRAINING_ELIGIBLE`.

The certificate may authorize the next work order to create a training run plan,
but it must not authorize training by itself.

## Privacy

Committed scoring reports may contain only aggregate metrics and hashes.

Do not commit raw generated text, raw reference text, raw hypotheses, candidate
IDs, holdout IDs, audio paths, local absolute paths, per-row scores, monitor
CSVs, or local manifests.

## Required execution

```bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0

.venv/bin/python scripts/score_corpus_v2_audio.py --stage verify --require-authorization SCORING_AUTHORIZED
.venv/bin/python scripts/score_corpus_v2_audio.py --corpus-role synthetic_candidate --stage score
.venv/bin/python scripts/score_corpus_v2_audio.py --corpus-role synthetic_holdout --stage score
.venv/bin/python scripts/score_corpus_v2_audio.py --stage summarize
.venv/bin/python scripts/build_corpus_v2_selected_training.py --target-hard 120 --target-control 40 --require-status SELECTED_TRAINING_MANIFEST_READY

.venv/bin/python -m unittest tests.test_corpus_v2_scoring_selection
.venv/bin/python -m unittest tests.test_batched_streaming
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
git ls-files | grep -E '.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan|csv|tsv)$' || true
```

## Non-goals

Do not:

- run GaMS;
- run Piper;
- regenerate text or audio;
- change candidate or holdout corpora;
- change the A100 batch policy;
- use batch size above 1 for scoring;
- use FLEURS or ARTUR failures for selection;
- select holdout rows for training;
- create a training config;
- train prompt columns or adapters;
- issue `TRAINING_ELIGIBLE`;
- evaluate a challenger checkpoint;
- publish text, audio, or model artifacts;
- merge the PR.

## Definition of done

The PR is ready for strategic review only when scoring authorization is
verified, candidate source scoring completes for exactly 415 rows, holdout
scoring completes for exactly 96 rows, metrics are aggregate-only and
privacy-safe, selected training contains exactly 160 candidate-source rows,
selected training contains 120 hard and 40 control rows, no holdout row enters
selected training, diversity constraints and relaxations are recorded, the
selected-training certificate is committed with status
`SELECTED_TRAINING_MANIFEST_READY`, `TRAINING_ELIGIBLE` is not issued, no model
training occurred, no raw predictions/text/IDs/local manifests/monitor CSVs or
audio are committed, all tests and repository checks pass, and the PR remains
unmerged.
