# Training Corpus Text Validator

`scripts/validate_training_corpus.py` is the reusable fail-closed text-stage
admission validator required by the training-data constitution.

It validates candidate text before TTS, ASR scoring, hard-example selection, or
model training. It does not synthesize audio, run ASR, issue an acoustic
certificate, or produce `TRAINING_ELIGIBLE` status.

## Record Contract

The validator accepts version `2.0` JSONL text records with out-of-band IDs,
source and family metadata, partition role, source type, license, domain,
phenomena, optional entity annotations, optional minimal-pair metadata, and
natural Slovenian `spoken_text`/`target_text`.

The default policy in
[`configs/data_quality/training_text_v1.json`](../configs/data_quality/training_text_v1.json)
requires:

- `language == "sl-SI"`;
- `spoken_text == target_text`;
- globally unique candidate IDs across supplied partitions;
- valid source, source-family, and utterance-family IDs;
- generation provenance for generated text;
- no generation provenance for authentic or real-speech transcript rows;
- complete linguistic review coverage before `TEXT_ACCEPTED`.

## Required Review Sidecar

The validator requires a separate local JSONL review file keyed by
`candidate_id`. Only `ACCEPT` passes. Missing reviews and
`REVISE_AND_REREVIEW` leave the corpus in `DRAFT`; rejection outcomes produce
`TEXT_REJECTED`.

The public report contains aggregate review counts and candidate IDs for
machine triage. It does not contain reviewer identities or raw review comments.

## Fingerprints

The validator computes deterministic structural views:

- surface-normalized hashes using the project Slovenian ASR normalizer;
- number-masked hashes;
- entity-masked hashes from declared entities plus deterministic numeric/date
  masks;
- data-driven carrier detection over prefix, suffix, frame, and n-gram
  concentration;
- carrier-stripped hashes;
- token 2- through 5-gram fuzzy review candidates;
- character 5-gram fuzzy review candidates;
- discovered template-family identifiers.

Exact structural collisions are hard failures except for explicitly approved
same-partition minimal-pair families. High fuzzy similarity is `DRAFT` pending
review, not accepted.

## Protected Gate Indexes

`scripts/build_protected_text_index.py` builds a local ignored hash-only index
from a real-gate manifest and committed metadata:

```bash
.venv/bin/python scripts/build_protected_text_index.py \
  --manifest runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl \
  --metadata docs/evaluation-gates/fleurs-sl-si-test-full-v2.metadata.json \
  --output runs/data-quality/protected/fleurs-v2.hash-index.json
```

The index contains gate identity, manifest hashes, normalizer/fingerprint
versions, surface hashes, and number-masked hashes. It must not contain raw
references, audio paths, hypotheses, or local absolute paths.

The default validator policy requires the canonical FLEURS v2 and ARTUR-J
protected indexes. Missing required indexes leave the corpus in `DRAFT`; stale
indexes are `TEXT_REJECTED`.

## CLI

Example:

```bash
.venv/bin/python scripts/validate_training_corpus.py \
  --config configs/data_quality/training_text_v1.json \
  --corpus-id sl-corpus-v2-example \
  --partition synthetic_candidate=runs/data-quality/example/candidates.jsonl \
  --partition selected_training=runs/data-quality/example/training.jsonl \
  --partition synthetic_holdout=runs/data-quality/example/holdout.jsonl \
  --linguistic-review runs/data-quality/example/review.jsonl \
  --protected-index runs/data-quality/protected/fleurs-v2.hash-index.json \
  --protected-index runs/data-quality/protected/artur-j.hash-index.json \
  --output-report runs/data-quality/example/text-admission-report.json \
  --local-review-output runs/data-quality/example/review.local.jsonl \
  --require-status TEXT_ACCEPTED
```

The command exits nonzero unless the final status equals `--require-status`.
The public report is privacy-safe and contains no raw corpus text, no raw
protected references, no local absolute paths, no generated audio paths, and no
reviewer identity.

## Corpus-v2 Reservoir Workflow

`scripts/generate_gams_corpus_v2.py` and
`scripts/prepare_corpus_v2_review.py` implement the first governed GaMS
candidate-reservoir stage. The generator creates local schema-2.0
`synthetic_candidate` rows, filters malformed and structurally concentrated
output, builds a local review template, and writes a privacy-safe aggregate
report.

The reservoir remains `DRAFT` because no genuine native-speaker linguistic
review sidecar is supplied. It is not `TEXT_ACCEPTED`, cannot become
`TRAINING_ELIGIBLE`, and must not proceed to TTS, ASR scoring, selection, or
training until later work orders complete review, acoustic validation, and data
certification.

`scripts/admit_reviewed_corpus_v2.py` ingests the local TSV review sheet,
preserves the complete human decision record in ignored storage, writes an
accepted-outcome subset, and reruns the text validator on that subset. It does
not infer missing review metadata and does not correct generated text.

Current post-review status: the local edited TSV contains 415 `ACCEPT`
outcomes, but every row lacks the required `review_revision`. The admission
command therefore reports `DRAFT` with reasons `blank_review_revision` and
`missing_linguistic_review`. The reservoir is still not `TEXT_ACCEPTED`;
the accepted-review sidecar is empty because the review metadata is incomplete.
TTS, ASR scoring, selection, and training remain unauthorized.

## Status Boundaries

This validator may emit only:

```text
DRAFT
TEXT_REJECTED
TEXT_ACCEPTED
DIAGNOSTIC_ONLY
RETIRED
```

It cannot emit:

```text
AUDIO_REJECTED
AUDIO_ACCEPTED
TRAINING_ELIGIBLE
```

A later work order must validate audio, issue a privacy-safe data acceptance
certificate, and explicitly authorize training before any corpus can become
`TRAINING_ELIGIBLE`.

## Retired Corpus Registry

[`configs/data_quality/retired_corpora.json`](../configs/data_quality/retired_corpora.json)
contains the three permanently retired Round 1 v1 corpus identities. The
validator hashes every input partition before parsing and returns `RETIRED` on
an exact match.

Hash retirement is not the only safeguard. Structurally similar corpora with new
hashes still go through the fingerprint, concentration, partition, protected
index, and linguistic review checks.
