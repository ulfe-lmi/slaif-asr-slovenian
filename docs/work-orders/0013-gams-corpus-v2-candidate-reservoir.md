# Work Order 0013: GaMS Corpus-v2 Candidate Reservoir

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

Expected base: `1a1a2546e4a293c041754fd2c0f2269cd968e2f5`

## Governing Instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/training-data-constitution.md`
- `docs/adr/0003-failure-directed-synthetic-curriculum.md`
- `docs/adr/0006-training-data-admission-policy.md`
- `docs/data-quality-validator.md`
- `docs/data-policy.md`
- `docs/training-plan.md`

Verify current remote `main`, open PRs, and the working tree before editing.
Live repository state is authoritative.

## Branch and Pull Request

- Branch: `feat/gams-corpus-v2-candidate-reservoir`
- Commit: `feat: add GaMS corpus-v2 candidate reservoir`
- Pull request title: `feat: add GaMS corpus-v2 candidate reservoir`

Do not merge the pull request.

## Goal

Implement and execute a governed GaMS corpus-v2 candidate-generation stage that:

- generates a diverse surplus of natural Slovenian candidate utterances;
- serializes them using text-record schema 2.0;
- keeps all IDs and generation bookkeeping out of spoken text;
- rejects malformed, duplicated, template-concentrated and protected-gate-overlapping candidates;
- runs the existing fail-closed text validator;
- creates a local native-speaker review pack;
- stops at `DRAFT`, awaiting genuine human linguistic review.

Do not synthesize audio or train a model.

## Scientific Role

This work creates a candidate source reservoir, not training data.

- Corpus ID: `sl-corpus-v2-gams-candidate-reservoir-v1`
- Partition role: `synthetic_candidate`
- Intended status: `DRAFT`
- Target generated rows: 480
- Minimum structurally admissible pre-review rows: 320

A shortfall is preferable to lowering thresholds or admitting poor text.

No selected-training partition and no synthetic holdout are created in this work
order. The holdout must later use a separately governed source strategy.

## Generation Matrix

Use twelve precommitted prompt cells, 40 requested candidates per cell:

1. Everyday conversational Slovenian, short utterances
2. Everyday informational Slovenian, medium utterances
3. Natural questions and requests
4. Practical commands and instructions
5. Public services, institutions and administrative situations
6. Slovenian people, places and inflected named entities
7. Dates, times, quantities, measurements and addresses
8. Case government, agreement and morphology
9. Dual forms, clitics and function words
10. Prepared public speech and interview-style statements
11. Technical, scientific and carefully bounded code-switching
12. Longer natural informational utterances

Each prompt cell must declare domain, register, length target, phenomena,
source family ID, prompt revision, seed sequence, requested rows, and maximum
retries.

Do not put cell IDs, batch numbers, candidate numbers, category labels, or
source-family identifiers into the requested utterance text.

## Execution Requirements

Use only:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Required GaMS settings:

- exactly one visible GPU, logical `cuda:0`;
- `cjvt/GaMS3-12B-Instruct`;
- revision `1d0b27af5748784482600d24779409e7e1dc9adc`;
- 4-bit NF4;
- double quantization enabled;
- BF16 compute;
- no CPU or disk offload;
- maximum GPU memory 76 GiB.

Add prompt batching with default batch size 4 and allowed values 1, 2, 4, and
8. Use explicit decoder-only padding and attention masks. Run a four-prompt
smoke at batch sizes 1 and 4 before the full generation run.

Create ignored local artifacts under:

```text
runs/data-quality/sl-corpus-v2-gams-candidate-reservoir-v1/
```

The local artifacts include raw generation, generated candidate JSONL,
pre-review candidate JSONL, rejected rows, review template, TSV review sheet,
validator report, local review output, and GPU monitoring CSV.

Commit only privacy-safe aggregate Markdown and JSON reports.

## Validation

Build and use protected text indexes for:

- `fleurs-sl-si-test-full-v2`
- `artur-j-public-gate-v1`

Run the existing text validator against the generated synthetic-candidate
partition without a fabricated linguistic-review sidecar. The expected final
status is `DRAFT`, with native-speaker linguistic review outstanding.

The work fails if the corpus reaches `TEXT_ACCEPTED` without genuine review, or
if hard structural defects are still present and the pool is described as ready
for review.

## Non-goals

Do not:

- create a synthetic holdout;
- create selected-training data;
- synthesize Piper audio;
- implement acoustic validation;
- issue a data certificate;
- emit `TRAINING_ELIGIBLE`;
- score candidates with Nemotron;
- select hard examples;
- train a prompt column or adapter;
- evaluate a challenger;
- expose FLEURS or ARTUR hypotheses through GaMS;
- publish generated text, audio, or model artifacts;
- merge the PR.
