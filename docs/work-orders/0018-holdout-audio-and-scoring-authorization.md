# Work Order 0018: Holdout Audio and Scoring Authorization

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

## Branch and PR

- Branch: `feat/holdout-audio-and-scoring-authorization`
- Commit: `feat: validate holdout audio and authorize corpus-v2 scoring`
- Pull request title: `feat: validate holdout audio and authorize corpus-v2 scoring`

## Goal

Complete the pre-scoring data gate:

1. synthesize the accepted 96-row independent synthetic holdout with Piper;
2. run fail-closed acoustic validation for the holdout;
3. verify candidate-source and holdout partition independence after audio-stage materialization;
4. commit a privacy-safe data certificate authorizing ASR scoring and selected-training construction;
5. explicitly do not authorize model training.

This work stops before ASR scoring.

## Fixed Inputs

Candidate source:

- corpus ID: `sl-corpus-v2-gams-candidate-reservoir-v1`
- accepted text partition SHA256: `b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6`
- accepted review sidecar SHA256: `4dc16336dc9404d48cab196b862e4aa2a4558b20d2728fc954dd7fbb88fed732`
- audio manifest SHA256: `c1d366e1d05b6f728af51b3350556b6d915fabf5a6b584a6aa2f9fdc0df538bc`
- audio certificate SHA256: `25737f59397d5c5acdd99e6af83e1129587199cb1a184eaec43dd27139bb1692`
- rows/audio: 415
- status: `TEXT_ACCEPTED`, `AUDIO_ACCEPTED`

Independent synthetic holdout:

- corpus ID: `sl-corpus-v2-independent-synthetic-holdout-v1`
- accepted text SHA256: `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`
- rows: 96
- decision: `ACCEPT`
- decision ID: `human-holdout-decision-v1`
- review revision: `human-holdout-review-v1`
- starting status: `TEXT_ACCEPTED`

## Runtime

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Required:

- physical GPU selector 1;
- exactly one visible CUDA device;
- logical device `cuda:0`;
- NVIDIA A100-SXM4-80GB;
- external `.venv-piper`;
- Piper engine revision `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`;
- voice `sl_SI-artur-medium`;
- voice revision `217ddc79818708b078d0d14a8fae9608b9d77141`;
- CUDAExecutionProvider required;
- CPUExecutionProvider fallback rejected;
- no Nemotron execution;
- no model training.

## Required Execution

```bash
export CUDA_VISIBLE_DEVICES=1

.venv/bin/python scripts/synthesize_corpus_v2_audio.py \
  --corpus-role synthetic_holdout \
  --stage verify

.venv/bin/python scripts/synthesize_corpus_v2_audio.py \
  --corpus-role synthetic_holdout \
  --stage synthesize

.venv/bin/python scripts/validate_corpus_audio.py \
  --corpus-role synthetic_holdout \
  --require-status AUDIO_ACCEPTED

.venv/bin/python scripts/synthesize_corpus_v2_audio.py \
  --corpus-role synthetic_holdout \
  --stage summarize

.venv/bin/python scripts/authorize_corpus_v2_scoring.py \
  --require-status SCORING_AUTHORIZED
```

Then run repository checks:

```bash
.venv/bin/python -m unittest tests.test_acoustic_quality
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
.venv-piper/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
git ls-files | grep -E '\.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan|csv|tsv)$' || true
```

## Non-goals

Do not:

- run GaMS;
- regenerate candidate or holdout text;
- synthesize candidate audio again unless strictly necessary;
- run Nemotron;
- perform ASR scoring;
- create selected-training data;
- train any model parameter;
- issue `TRAINING_ELIGIBLE`;
- evaluate or promote a checkpoint;
- change the A100 batch-1 evaluation policy;
- use GPUs 0, 2, or 3;
- publish generated text or audio;
- merge the PR.

## Definition of Done

- Holdout text identity is verified.
- All 96 holdout rows are synthesized.
- Holdout audio validates as `AUDIO_ACCEPTED`, or failure is honestly recorded.
- Candidate source audio certificate is verified by hash.
- Candidate/holdout text and audio partition independence is verified.
- Scoring authorization certificate is committed if and only if all required checks pass.
- Certificate status is `SCORING_AUTHORIZED`, not `TRAINING_ELIGIBLE`.
- No ASR scoring or training occurs.
- No raw data or audio is committed.
- Repository checks pass.
- PR remains unmerged.
