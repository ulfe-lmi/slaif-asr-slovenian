# Work Order 0014: Corpus-v2 Linguistic Review Admission

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

Expected main commit at issuance:
`00b59b002edc0fef6401ea8ed2e8665aab70be44`

PR #16 is expected to be merged. Live repository state is authoritative.

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/adr/0006-training-data-admission-policy.md`
- `docs/data-quality-validator.md`
- `docs/work-orders/0013-gams-corpus-v2-candidate-reservoir.md`

Verify live `main`, open PRs, and the working tree before editing.

## Branch and Pull Request

- Branch: `feat/corpus-v2-linguistic-review-admission`
- Commit: `feat: admit linguistically reviewed corpus-v2 candidates`
- Pull request title: `feat: admit linguistically reviewed corpus-v2 candidates`

Open a draft PR. Do not merge it.

## Local Inputs

Expected local source reservoir:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/pre-review-candidates.local.jsonl
```

Expected source reservoir SHA256:

```text
5cb2520c27b3debd18a2f475368c2cdd8601fc5781ec541287092dbcd3ea0fe6
```

Expected human-edited review sheet:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/linguistic-review-sheet.local.tsv
```

Expected review-template SHA256 before human editing:

```text
a22d87aa5e6913d2ceeaa1c61414ab946ef4c5c1bb1f8cc5b8063f984e454d84
```

The edited review TSV is not required to retain the original template hash.

If required local inputs are absent, stop with `ENVIRONMENT_BLOCKED`.

## Goal

Ingest the genuine native-speaker review decisions, produce the accepted
post-review candidate partition, rerun corpus-level text checks, and determine
whether the candidate reservoir reaches `TEXT_ACCEPTED`.

This work order must not emit `TRAINING_ELIGIBLE`.

## Required Implementation

Add a reusable review-admission command such as:

```text
scripts/admit_reviewed_corpus_v2.py
```

It must:

- read the completed TSV safely;
- require exactly one review row per pre-review candidate;
- reject duplicate, unknown, or missing candidate IDs;
- require a supported outcome on every row;
- require non-empty `review_revision` on every row;
- reject malformed boolean or reason-code fields;
- never infer, fabricate, or prefill `ACCEPT`;
- never rewrite or automatically correct generated Slovenian text;
- preserve the complete human decision record in ignored local storage;
- produce an accepted candidate JSONL containing only rows whose outcome is
  `ACCEPT`;
- produce an accepted-only linguistic-review JSONL compatible with
  `scripts/validate_training_corpus.py` only when review metadata is complete;
- rerun the complete text validator on the accepted subset;
- rerun structural concentration and protected-gate checks after filtering;
- write deterministic, atomic outputs;
- keep all candidate text and review details ignored and uncommitted.

Required ignored local outputs:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/
  linguistic-review-decisions.local.jsonl
  accepted-candidates.local.jsonl
  accepted-linguistic-review.local.jsonl
  post-review-validation.local.json
  post-review-review.local.jsonl
  rejected-by-human.local.jsonl
```

## Review Semantics

Allowed outcomes:

```text
ACCEPT
REJECT_GRAMMAR
REJECT_SEMANTICS
REJECT_UNNATURAL
REJECT_TEMPLATE
REJECT_METADATA_LEAK
REJECT_DUPLICATE
REJECT_DOMAIN
REJECT_TRANSCRIPTION
REVISE_AND_REREVIEW
```

`ACCEPT` enters the accepted-candidate subset. `REJECT_*` and
`REVISE_AND_REREVIEW` are excluded and recorded locally.

A revised sentence is a new corpus object requiring fresh generation
provenance, fingerprints, and review; that is out of scope here.

`minimal_pair_approved` must remain false unless the source record actually
declares a minimal-pair family and the reviewer explicitly approved it.

## Protected Indexes

Require valid local hash-only indexes for:

- `fleurs-sl-si-test-full-v2`
- `artur-j-public-gate-v1`

Rebuild them from local manifests if absent and verify their manifest hashes
against committed metadata. Missing indexes result in `DRAFT`; stale or
mismatched indexes result in `TEXT_REJECTED`.

Never expose raw protected references.

## Validation Command

Run the equivalent of:

```bash
.venv/bin/python scripts/validate_training_corpus.py \
  --config configs/data_quality/training_text_v1.json \
  --corpus-id sl-corpus-v2-gams-candidate-reservoir-v1-reviewed \
  --partition synthetic_candidate=runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/accepted-candidates.local.jsonl \
  --linguistic-review runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/accepted-linguistic-review.local.jsonl \
  --protected-index runs/data-quality/protected/fleurs-v2.hash-index.json \
  --protected-index runs/data-quality/protected/artur-j.hash-index.json \
  --output-report runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/post-review-validation.local.json \
  --local-review-output runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/post-review-review.local.jsonl \
  --require-status TEXT_ACCEPTED
```

Do not force `TEXT_ACCEPTED`. If the actual result is `DRAFT` or
`TEXT_REJECTED`, preserve that result, identify exact reasons, and exit
nonzero.

## Privacy-Safe Committed Report

Add:

```text
docs/data-reports/0002-corpus-v2-linguistic-review-admission.md
docs/data-reports/0002-corpus-v2-linguistic-review-admission.json
```

The report may contain only aggregate evidence: source and review-sheet
hashes, counts, review coverage, accepted-partition hash, fingerprint unique
counts, family statistics, protected-index identities, final text status,
validator/config/code revisions, and limitations.

It must not contain generated sentences, candidate IDs, reviewer identity, raw
comments, protected references, or local absolute paths.

If the result is `TEXT_ACCEPTED`, state clearly that text admission has passed,
acoustic suitability remains untested, no synthetic holdout exists, the corpus
is still not `TRAINING_ELIGIBLE`, and TTS and model training remain
unauthorized.

## Non-Goals

Do not:

- generate additional sentences;
- correct or paraphrase reviewed sentences;
- create a synthetic holdout;
- create a selected-training partition;
- run TTS;
- validate audio;
- issue a data acceptance certificate;
- emit `TRAINING_ELIGIBLE`;
- score candidates;
- select hard examples;
- train or evaluate a model;
- commit the review sheet or generated corpus;
- publish data;
- merge the PR.

## Required Verification

Run:

```bash
.venv/bin/python -m unittest tests.test_corpus_v2_generation
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

No GPU is required. Do not run GaMS, Piper, Nemotron, ASR inference, scoring,
or training.

## Definition of Done

The PR is ready for strategic review only when:

- all pre-review rows have exactly one review decision;
- accepted rows are derived without text modification;
- human rejections remain excluded and auditable locally;
- the accepted subset passes fresh structural and protected-gate validation, or
  failure is honestly reported;
- final status is accurately recorded;
- no raw generated text or reviewer identity is committed;
- no TTS, ASR, training, or GPU work occurred;
- all repository checks pass;
- the PR remains unmerged.
