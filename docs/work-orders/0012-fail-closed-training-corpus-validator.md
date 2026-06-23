# Work Order 0012: Implement the Fail-Closed Training-Corpus Validator

Status: ready for execution
Repository: `ulfe-lmi/slaif-asr-slovenian`
Expected base main: `dd105f1b78096c0ca8c184f1e00a4bd21480a509`

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/adr/0003-failure-directed-synthetic-curriculum.md`
- `docs/adr/0006-training-data-admission-policy.md`
- `docs/data-policy.md`
- `docs/testing-strategy.md`
- `docs/training-plan.md`

Verify live `main`, open PR state, and the working tree before editing. Remote
repository state is authoritative.

## Branch And Pull Request

Branch:

```text
feat/fail-closed-corpus-validator
```

Commit:

```text
feat: add fail-closed training-corpus validator
```

Pull request title:

```text
feat: add fail-closed training-corpus validator
```

Open the pull request and leave it unmerged.

## Goal

Implement a reusable, deterministic, privacy-safe text-stage training-data
admission validator that:

- rejects the structural failure modes missed by the Round 1 validator;
- enforces cross-partition content and family independence;
- enforces complete Slovenian linguistic-review coverage;
- rejects retired corpus identities before parsing or processing them;
- checks candidates against local protected-gate hash indexes;
- produces a machine-readable admission report without raw corpus text;
- fails closed when required checks are missing, blocked, unresolved, or not run.

This PR must make equivalent Round 1 failure structures impossible to classify
as `TEXT_ACCEPTED`.

## Scope Boundary

This work order implements text-stage admission tooling only.

It must not:

- generate corpus v2;
- clean, modify, rehabilitate, or rehash the retired corpus;
- issue a real corpus acceptance certificate;
- award `AUDIO_ACCEPTED` or `TRAINING_ELIGIBLE`;
- synthesize audio;
- score candidates with ASR;
- run GaMS, Piper, Nemotron, or an A100;
- train or evaluate a model.

The strongest successful corpus state this validator may emit is
`TEXT_ACCEPTED`. A later data work order must perform acoustic validation and
issue the actual privacy-safe acceptance certificate before `TRAINING_ELIGIBLE`
is possible.

## Required Ownership

Add:

- `slaif_asr/data_quality.py`
- `scripts/validate_training_corpus.py`
- `scripts/build_protected_text_index.py`
- `configs/data_quality/training_text_v1.json`
- `configs/data_quality/retired_corpora.json`
- `tests/test_data_quality.py`
- `tests/fixtures/data_quality/`
- `docs/data-quality-validator.md`

Use the Python standard library and existing project utilities. Do not add an
NLP, embedding, database, or approximate-nearest-neighbour dependency.

Do not replace or silently change historical behavior in:

- `slaif_asr/slovenian_curriculum.py`
- `configs/generation/slovenian_curriculum_round1.json`

The historical validator remains available for exact reproduction of Experiment
0004, but documentation must clearly prohibit it as the admission authority for
new corpora.

## Required Contract

Define a new versioned text record contract with schema version `2.0` and
out-of-band metadata for candidate ID, source ID, source family, template
family, utterance family, phenomena, domain, license, generation provenance,
entities, and minimal pairs.

Require a separate local linguistic-review JSONL keyed by `candidate_id`. Only
`ACCEPT` passes. Missing review, `REVISE_AND_REREVIEW`, blocked, unknown, or
unrun review prevents `TEXT_ACCEPTED`.

Implement the adopted status vocabulary, while this text validator may emit
only:

```text
DRAFT
TEXT_REJECTED
TEXT_ACCEPTED
DIAGNOSTIC_ONLY
RETIRED
```

The CLI must make `TRAINING_ELIGIBLE` impossible.

## Retired Corpus Enforcement

Create `configs/data_quality/retired_corpora.json` containing exactly:

- candidate pool:
  `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17`
- synthetic holdout:
  `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9`
- selected training manifest:
  `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4`

Before parsing an input partition, compute its byte-level SHA256 and return
`RETIRED` on a match.

## Required Fingerprint Suite

Implement deterministic algorithms for:

- surface-normalized fingerprints;
- number-masked fingerprints;
- metadata-token detection;
- entity-masked fingerprints;
- carrier detection and carrier-stripped fingerprints;
- token 2-, 3-, 4-, and 5-gram similarity;
- character 5-gram similarity with a review threshold below the historical
  `0.82` boundary;
- discovered template-family clustering;
- explicit approved minimal-pair handling.

Use deterministic candidate blocking or inverted shingle indexes. Do not use an
unconditional all-pairs fuzzy comparison.

## Cross-Partition Validation

For selected training versus synthetic holdout, require zero overlap in:

- candidate ID;
- source ID;
- source family ID;
- utterance family ID;
- declared template family;
- discovered template family;
- surface hash;
- number-masked hash;
- entity-masked hash;
- carrier-stripped hash.

Also reject selected-training rows absent from the admitted candidate source
pool, source-recording leakage, acoustic/text variants crossing partitions, and
unresolved high-similarity cross-partition pairs.

## Protected Gate Hash Indexes

Add `scripts/build_protected_text_index.py`.

It accepts a local ignored real-gate manifest and committed metadata, verifies
manifest identity, and emits an ignored hash-only index containing gate identity,
manifest hashes, normalizer/fingerprint versions, surface hashes, and
number-masked hashes.

The index must not contain raw references, audio paths, hypotheses, or local
absolute paths.

The corpus validator accepts repeated `--protected-index` arguments. Missing
required protected indexes prevent `TEXT_ACCEPTED`; stale indexes are
`TEXT_REJECTED`.

## CLI

Required interface:

```bash
.venv/bin/python scripts/validate_training_corpus.py \
  --config configs/data_quality/training_text_v1.json \
  --corpus-id sl-corpus-v2-example \
  --partition synthetic_candidate=path/to/candidates.jsonl \
  --partition selected_training=path/to/training.jsonl \
  --partition synthetic_holdout=path/to/holdout.jsonl \
  --linguistic-review path/to/review.jsonl \
  --protected-index path/to/fleurs-v2.hash-index.json \
  --protected-index path/to/artur-j.hash-index.json \
  --output-report runs/data-quality/sl-corpus-v2-example/text-admission-report.json \
  --local-review-output runs/data-quality/sl-corpus-v2-example/review.local.jsonl \
  --require-status TEXT_ACCEPTED
```

The command exits nonzero unless the resulting status satisfies
`--require-status`. The public report must contain no raw utterances, protected
references, local absolute paths, reviewer identity, audio paths, or
hypotheses.

## Tests

Add adversarial tests for:

- same template with different row numbers;
- same template with different names;
- same body with different artificial suffixes;
- same body with different artificial prefixes;
- train/holdout same body with different IDs;
- punctuation and casing variants;
- suspicious inflectional variants;
- legitimate unrelated sentences sharing function words;
- approved minimal-pair family;
- metadata identifier embedded in spoken text;
- acoustic variants of one utterance crossing partitions;
- one source recording split across partitions;
- malformed Slovenian slot insertion with missing or rejecting review;
- similarity just below the old `0.82` threshold;
- missing/stale protected gate index;
- retired artifact hash;
- unresolved fuzzy review pair;
- duplicate or incomplete linguistic review sidecar;
- deterministic output after input reordering.

At least one regression test must demonstrate that the legacy Round 1-style
validator accepts a safe bad fixture while the new validator rejects or blocks
the same failure structure.

## Repository Integration

Update `scripts/check_repository.py` to validate:

- data-quality configuration parses;
- the retired registry contains the exact constitutional hashes;
- committed future `docs/data-certificates/*.json` files contain no raw text,
  local paths, or forbidden keys;
- a certificate claiming `TRAINING_ELIGIBLE` has required high-level sections.

Do not commit a real data certificate in this PR.

## Documentation

Update:

- `README.md`
- `CHANGELOG.md`
- `AGENTS.md`
- `docs/testing-strategy.md`
- `docs/training-plan.md`
- `docs/roadmap.md`
- `docs/project-handoff.md`
- `docs/training-data-constitution.md`

State that the validator enforces text admission but does not prove acoustic
suitability. State that no corpus v2 has yet been accepted and no data
acceptance certificate exists yet.

## Verification

Run:

```bash
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python scripts/validate_training_corpus.py ... --require-status TEXT_ACCEPTED
.venv/bin/python scripts/validate_training_corpus.py ... --require-status TEXT_ACCEPTED
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
git ls-files | grep -E '\\.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan)$' || true
```

When ignored historical Round 1 files or local real-gate manifests are
available, verify the retired identities and build protected indexes. If they
are unavailable, report that integration check as environment-blocked.

No GPU command is required.

## Acceptance Criteria

The PR is ready for strategic review only when:

- the new validator is separate from the historical Round 1 implementation;
- exact retired hashes fail closed;
- a Round 1 analogue is accepted by the legacy validator but blocked or
  rejected by the new validator;
- mandatory structural views are implemented;
- cross-partition family leakage is rejected;
- incomplete linguistic review cannot produce `TEXT_ACCEPTED`;
- protected-gate indexes are hash-only and privacy-safe;
- missing required indexes prevent `TEXT_ACCEPTED`;
- reports contain no raw corpus text or local paths;
- positive and minimal-pair fixtures pass;
- all focused and repository tests pass;
- no corpus, audio, model, checkpoint, private text, or local report is
  committed;
- no real certificate or `TRAINING_ELIGIBLE` claim is produced;
- the PR remains unmerged.
