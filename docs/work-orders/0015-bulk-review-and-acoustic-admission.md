# Work Order 0015: Bulk Review And Acoustic Admission

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/adr/0006-training-data-admission-policy.md`
- relevant ADRs and data-policy documents

Verify live main, open pull requests, and the working tree before editing.

Expected main commit at issuance:

```text
7713943f094c415eaf3743b95cafea93df7bfffe
```

## Branch And Pull Request

Branch:

```text
feat/bulk-review-and-acoustic-admission
```

Commit:

```text
feat: add bulk corpus review and acoustic admission
```

Pull request title:

```text
feat: add bulk corpus review and acoustic admission
```

Open a draft PR. Do not merge it.

## Goal

Complete two sequential tasks:

1. Add an explicit whole-file human review mode so the human can declare an
   exact corpus file uniformly `ACCEPT`, `REJECT_*`, or
   `REVISE_AND_REREVIEW` without editing one row at a time.
2. Use the already `TEXT_ACCEPTED` 415-row corpus to run Piper synthesis and
   fail-closed acoustic validation.

Do not stop after implementing the bulk-review helper.

## Current Accepted Text Evidence

Corpus:

```text
sl-corpus-v2-gams-candidate-reservoir-v1
```

Expected source reservoir:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/pre-review-candidates.local.jsonl
```

Expected identities:

```text
source reservoir SHA256:
5cb2520c27b3debd18a2f475368c2cdd8601fc5781ec541287092dbcd3ea0fe6

rows:
415

whole-file review decision:
ACCEPT

review revision:
human-review-v1

review sheet SHA256:
b5ed25cb3aa81bb94741a7068f67be7c8a535ab7aab116b3ae80fc2613d749c7

accepted candidate partition SHA256:
b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6

accepted linguistic-review sidecar SHA256:
4dc16336dc9404d48cab196b862e4aa2a4558b20d2728fc954dd7fbb88fed732

post-review validation SHA256:
d15d460c723c171130e7fe17bd6b767d29bf803785667dd2e8e3145e02b2131c
```

Expected text status:

```text
TEXT_ACCEPTED
```

If local artifacts do not match, stop with `EXPERIMENT_INVALID`.

## Part A: Whole-File Review Decision

Extend the review-admission tooling with a corpus-level mode.

Required interface:

```text
.venv/bin/python scripts/admit_reviewed_corpus_v2.py \
  --whole-file-outcome ACCEPT \
  --review-revision human-review-v1 \
  --decision-id human-corpus-decision-v1 \
  --expected-corpus-sha256 5cb2520c27b3debd18a2f475368c2cdd8601fc5781ec541287092dbcd3ea0fe6 \
  --expected-rows 415 \
  --require-status TEXT_ACCEPTED
```

Requirements:

- The mode is mutually exclusive with a row-edited review TSV.
- It requires an explicit human-supplied outcome.
- It requires a non-empty review revision and decision ID.
- It requires an exact expected byte hash and row count.
- It applies the same decision to every row.
- It creates ordinary per-row linguistic-review sidecars locally so the
  existing validator remains authoritative.
- It must never infer `ACCEPT`.
- It must never default to `ACCEPT`.
- It must never rewrite candidate text.
- A wrong hash or row count must fail before producing review decisions.
- It supports `ACCEPT`, every existing `REJECT_*` outcome, and
  `REVISE_AND_REREVIEW`.
- An all-reject decision must not create an accepted subset.
- Public reports record only the decision ID, revision, count, corpus hash,
  outcome, and aggregates, not reviewer identity or raw text.

Update the training-data constitution to record this human-approved rule:

For a bounded corpus that the human judges uniformly good or uniformly bad,
the human may issue one explicit whole-corpus decision bound to the exact
corpus SHA256 and row count. The tooling expands that decision into per-row
review records. This is a genuine human decision, not automatic review. Mixed-
quality corpora still require row-level decisions.

Add tests for bulk accept, bulk rejection, bulk revision, missing decision ID,
blank revision, wrong hash, wrong row count, no implicit default, TSV and whole-
file mutual exclusion, deterministic generated sidecar, and no text mutation.

## Correct Stale Committed Evidence

Regenerate:

```text
docs/data-reports/0002-corpus-v2-linguistic-review-admission.json
docs/data-reports/0002-corpus-v2-linguistic-review-admission.md
```

They must record `TEXT_ACCEPTED`, 415/415 accepted, complete coverage, the new
accepted hashes, all required text checks passed, and that acoustic suitability
was still untested at that point.

## Part B: Piper Synthesis

After whole-file admission reproduces `TEXT_ACCEPTED`, synthesize all 415
accepted candidates.

Use:

```text
export CUDA_VISIBLE_DEVICES=1
```

Required runtime:

- physical GPU selector 1;
- one visible logical GPU, `cuda:0`;
- NVIDIA A100-SXM4-80GB;
- external `.venv-piper`;
- Piper engine revision `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`;
- voice `sl_SI-artur-medium`;
- voice revision `217ddc79818708b078d0d14a8fae9608b9d77141`;
- native rate 22050 Hz;
- final rate 16000 Hz;
- mono signed 16-bit PCM;
- no CPU execution-provider fallback.

Do not alter the historical synthetic-smoke TTS path.

Add a schema-2.0 corpus-v2 bridge that converts accepted text records to the
existing external Piper boundary without weakening the older smoke schema.

## Bounded Piper Concurrency

The current per-row Piper path is serial. Add bounded concurrent synthesis.

Test worker counts:

```text
1, 2, 4, 8
```

Use a deterministic 32-row subset.

For each worker count record:

- successful rows;
- failed rows;
- wall time;
- utterances per minute;
- audio seconds per wall-clock second;
- mean, median and p95 GPU utilization;
- peak GPU memory;
- output-hash parity against worker count 1.

Select the smallest worker count within 5% of the best valid throughput.

A worker count is invalid if any output fails, CPU execution provider is used,
CUDA provider initialization fails, output count differs, output hashes differ
from worker count 1, or GPU or host memory is exhausted.

Do not test worker counts above 8 in this work order.

## Acoustic Validator

Add reusable ownership such as:

```text
slaif_asr/acoustic_quality.py
scripts/validate_corpus_audio.py
configs/data_quality/synthetic_audio_v1.json
```

For every item validate:

- exactly one audio file;
- unique audio path;
- unique audio SHA256;
- expected candidate ID;
- transcript and utterance-family linkage;
- complete Piper and conversion provenance;
- mono;
- 16000 Hz;
- signed 16-bit PCM;
- nonzero frame count;
- configured duration bounds;
- non-silence;
- amplitude above configured minimum;
- clipping fraction below configured maximum;
- no malformed sample data;
- no output overwrite;
- no missing candidate;
- no unexpected candidate;
- no cross-partition variant leakage;
- no CPU TTS fallback.

The configuration must version exact thresholds and report distributions for
duration, peak ratio, RMS level, active-frame fraction, clipping fraction,
leading and trailing silence, audio hashes, voice count, and duration.

Do not claim that waveform checks prove transcript correctness or natural
prosody.

## Status

The acoustic stage may emit:

```text
AUDIO_REJECTED
AUDIO_ACCEPTED
```

It must not emit:

```text
TRAINING_ELIGIBLE
```

Expected status when all 415 files pass:

```text
AUDIO_ACCEPTED
```

The candidate reservoir remains single voice, without an independent synthetic
holdout, without a selected-training partition, without a final training-data
certificate, and unauthorized for model training.

## Local Outputs

Keep all raw outputs ignored:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/audio/
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/audio-manifest.local.jsonl
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/audio-validation.local.json
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/piper-logs/
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/piper-benchmark/
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/gpu-monitor.local.csv
```

Do not commit audio, local manifests, raw logs, or monitoring CSV.

## Privacy-Safe Certificate And Report

Commit:

```text
docs/data-certificates/sl-corpus-v2-gams-candidate-reservoir-v1-audio.json
docs/data-reports/0003-corpus-v2-acoustic-admission.json
docs/data-reports/0003-corpus-v2-acoustic-admission.md
```

The certificate status is `AUDIO_ACCEPTED` or `AUDIO_REJECTED`, never
`TRAINING_ELIGIBLE`.

It must contain corpus identity, accepted text partition hash, review-decision
identity, row and audio counts, audio-manifest hash, engine and voice
revisions, audio format, total and distributional duration, waveform validation
aggregates, duplicate counts, failures by reason, selected worker count,
concurrency benchmark aggregates, GPU utilization aggregates, single-voice
limitation, validator and configuration revisions, and final status.

It must not contain raw text, candidate IDs, audio paths, local paths, or
reviewer identity.

## Required Execution

Run whole-file admission first, then:

```text
export CUDA_VISIBLE_DEVICES=1

.venv/bin/python scripts/synthesize_corpus_v2_audio.py --stage verify
.venv/bin/python scripts/synthesize_corpus_v2_audio.py --stage benchmark-workers
.venv/bin/python scripts/synthesize_corpus_v2_audio.py --stage synthesize
.venv/bin/python scripts/validate_corpus_audio.py --require-status AUDIO_ACCEPTED
.venv/bin/python scripts/synthesize_corpus_v2_audio.py --stage summarize
```

Then run repository tests, Python compilation, repository validation, pip
checks, shell syntax checks, whitespace checks, and tracked artifact inspection.

## Non-Goals

Do not:

- run GaMS again;
- create a synthetic holdout;
- score candidates with Nemotron;
- select hard examples;
- create a selected-training partition;
- train any model parameter;
- issue `TRAINING_ELIGIBLE`;
- evaluate a challenger;
- publish text, audio, or model artifacts;
- use GPUs 0, 2, or 3;
- merge the PR.

## Definition Of Done

The PR is ready for review only when:

- whole-file review works without editing 415 rows;
- it is bound to an exact hash and row count;
- text admission is reproduced as `TEXT_ACCEPTED`;
- stale DRAFT reports are corrected;
- all 415 accepted rows are synthesized or failure is honestly reported;
- Piper used only CUDAExecutionProvider;
- acoustic validation is fail-closed;
- the candidate audio pool is accurately classified;
- concurrency evidence is recorded;
- no raw corpus or audio is committed;
- `TRAINING_ELIGIBLE` is not claimed;
- all repository checks pass;
- the PR remains unmerged.
