# Work Order 0017: Corpus-v2 Independent Synthetic Holdout

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

Expected main commit at issuance:
`a73e6f4ff4a1353b6b4a3b6d4c827f89fcb71f7d`

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/data-policy.md`
- `docs/data-quality-validator.md`
- relevant ADRs

Verify live `main`, open pull requests, and the working tree before editing.
Live repository state is authoritative.

## Branch and Pull Request

- Branch: `feat/corpus-v2-independent-holdout`
- Commit: `feat: generate independent corpus-v2 holdout`
- Pull request title: `feat: generate independent corpus-v2 holdout`

Do not merge the pull request.

## Goal

Generate and validate a fixed 96-row synthetic diagnostic holdout that is
independent of the existing 415-row candidate reservoir.

The work must stop at `DRAFT` awaiting one whole-file human decision.

Do not synthesize audio, score ASR, select training rows, issue
`TRAINING_ELIGIBLE`, or train a model.

## Existing Candidate Source

Existing accepted candidate corpus:

```text
sl-corpus-v2-gams-candidate-reservoir-v1
```

Expected accepted text SHA256:

```text
b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6
```

Expected rows: `415`

Existing candidate audio status: `AUDIO_ACCEPTED`

The candidate corpus must remain unchanged.

## Holdout Identity

- Corpus ID: `sl-corpus-v2-independent-synthetic-holdout-v1`
- Partition role: `synthetic_holdout`
- Final fixed rows: `96`
- Requested generation rows: `160`
- Prompt cells: `8`
- Requested rows per cell: `20`
- Selected rows per cell: `12`

The 96-row holdout must be selected before any ASR scoring and without using
model hypotheses, WER, CER, or training behavior.

## Independent Source Strategy

Use separately pinned:

```text
cjvt/GaMS-9B-Instruct
revision: 292744023fa0b7ccc7ae2c3c885a67468e49fa03
```

Use 4-bit NF4, double quantization, BF16 compute, one visible A100, no CPU or
disk offload, and prompt batch size 4.

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Do not expose any candidate-pool sentence, FLEURS reference, ARTUR-J reference,
hypothesis, or per-sample ASR error to the generator.

## Prompt Cells

Use eight holdout prompt cells distinct from the candidate-reservoir prompt
matrix:

1. spontaneous conversational turns;
2. short personal and event narratives;
3. travel and mobility situations;
4. household, food, leisure, and daily planning;
5. education, culture, and public information;
6. weather, environment, and ordinary observations;
7. dates, times, quantities, appointments, and addresses;
8. longer prepared informational speech.

Prompts must require natural standalone Slovenian, correct morphology and
agreement, no numbering or labels, no artificial wrappers or repeated tails, no
corpus identifiers, and no quota filling by repeated sentence frames.

## Required Implementation

Add or update:

- `configs/generation/slovenian_corpus_v2_holdout_v1.json`
- `slaif_asr/corpus_v2_holdout.py`
- `scripts/generate_corpus_v2_holdout.py`
- `tests/test_corpus_v2_holdout.py`
- `docs/data-reports/0004-corpus-v2-independent-holdout.md`
- `docs/data-reports/0004-corpus-v2-independent-holdout.json`

The script must provide stages:

```text
verify
generate
validate
prepare-review
summarize
all
```

The generated local outputs remain ignored under:

```text
runs/data-quality/sl-corpus-v2-independent-synthetic-holdout-v1/
```

## Validation Requirements

Reject and report parser failures, invalid schema, metadata leakage, exact
surface duplicates, number-masked collisions, entity-masked collisions,
prohibited carriers, excessive template concentration, fuzzy duplicates,
protected-gate overlaps, and overlap with the accepted candidate source.

Each prompt cell must supply at least 12 admissible rows. Select the 12 rows
with the smallest `SHA256(candidate_id)` in each cell. Fail rather than
borrowing excess rows from another cell.

Run the existing validator jointly over:

- the 415-row accepted `synthetic_candidate` source;
- the new 96-row `synthetic_holdout`.

Require zero candidate/holdout overlap by IDs, source and utterance families,
declared and discovered template families, and structural fingerprints.

Use the existing FLEURS-v2 and ARTUR-J protected indexes.

The expected final state is `DRAFT` solely because the 96 holdout rows have not
yet received a human whole-file decision. A hard structural or partition
failure is not an acceptable DRAFT result.

## Review Capsule

Create local ignored files:

```text
runs/data-quality/sl-corpus-v2-independent-synthetic-holdout-v1/
  generated-all.local.jsonl
  fixed-holdout.local.jsonl
  rejected.local.jsonl
  validation.local.json
  review-capsule.local.tsv
  review-capsule.local.md
  whole-file-decision-command.local.txt
  gpu-monitor.local.csv
```

The review capsule must include all 96 holdout sentences locally, grouped by
prompt cell. Do not prefill a decision.

Generate the exact future whole-file command using:

```text
--whole-file-outcome <ACCEPT_OR_REJECT>
--review-revision human-holdout-review-v1
--decision-id human-holdout-decision-v1
--expected-corpus-sha256 <ACTUAL_FIXED_HOLDOUT_HASH>
--expected-rows 96
```

Do not execute `ACCEPT` automatically.

## Required Execution

```bash
export CUDA_VISIBLE_DEVICES=1

.venv-gams/bin/python scripts/generate_corpus_v2_holdout.py --stage verify
.venv-gams/bin/python scripts/generate_corpus_v2_holdout.py --stage generate
.venv/bin/python scripts/generate_corpus_v2_holdout.py --stage validate
.venv/bin/python scripts/generate_corpus_v2_holdout.py --stage prepare-review
.venv/bin/python scripts/generate_corpus_v2_holdout.py --stage summarize

.venv/bin/python -m unittest tests.test_corpus_v2_holdout
.venv/bin/python -m unittest tests.test_data_quality
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
.venv-gams/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
```

## Non-goals

Do not modify the existing candidate source, use candidate sentences in holdout
prompts, use FLEURS or ARTUR text, synthesize holdout audio, score any item
with Nemotron, create selected-training data, issue `AUDIO_ACCEPTED` or
`TRAINING_ELIGIBLE` for the holdout, train a model, change the batch-1 A100
evaluation policy, publish generated text, or merge the PR.

## Definition of Done

The PR is ready for strategic review when exactly 96 fixed holdout rows exist,
every prompt cell contributes exactly 12, the holdout is structurally
independent of the candidate source, protected overlap is zero, hard text
checks pass, final status is `DRAFT` only because human review is absent, one
exact-hash whole-file decision command is prepared, no raw generated text is
committed, all tests pass, and the PR remains unmerged.
