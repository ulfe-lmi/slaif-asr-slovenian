# Data Policy

## Purpose

This policy governs real speech, synthetic speech, transcripts, generated text, manifests, metrics, and released dataset artifacts.

`docs/training-data-constitution.md` is the detailed companion policy for
training-data admission. This document continues to govern general privacy,
provenance, licensing, storage, and partition handling; the training-data
constitution adds corpus-quality doctrine, data-status states, admission
stages, certificate requirements, and experiment-interpretation rules. A work
order may make data requirements stricter. A named exception may weaken a
requirement only with explicit human approval, written rationale, and a
narrowed scientific claim.

## Default classification

Treat all speech and transcripts as non-public unless their license and consent status are explicitly recorded.

## Prohibited Git content

Never commit:

- raw training or evaluation audio;
- private transcripts;
- names or identifiers from participant recordings;
- local corpus paths;
- generated TTS audio;
- pseudo-labels tied to private audio;
- access credentials or signed download URLs.

Only small, intentionally public, rights-cleared examples may be committed through an explicit work order.

## Required provenance

Every data source must record:

- source name and version;
- owner or publisher;
- license;
- redistribution status;
- consent/privacy classification;
- acquisition date;
- checksum or stable identifier;
- permitted uses;
- preprocessing and normalization;
- train/development/test role.

Synthetic records additionally require:

- GaMS model and revision;
- generation prompt/template revision;
- generation seed;
- TTS system and revision;
- voice/prosody settings;
- text validation outcome;
- selection reason.

## Partitions

### Controller development

The active loop may inspect references and hypotheses. Metrics from this set are not an unbiased final result.

`artur-controller-dev-v1` is the first governed real-acoustic
controller-development partition. It may be used for aggregate run-control
metrics and future early stopping only when an explicit work order authorizes
that use. Once used for checkpoint selection it is spent development data, not
unbiased acceptance evidence. It must not be used for training, GaMS prompt
content, selected-training construction, hard-example mining from raw
references or hypotheses, public quality claims, or model-release claims.

### Immutable gate

Used after rounds for acceptance. GaMS receives aggregate categories only, not raw reference sentences.

Current immutable real development gates are `fleurs-sl-si-test-full-v2`, the
complete FLEURS Slovenian test split with occurrence-index sample IDs, and the
ARTUR-J public-speech project gate. Neither gate may enter training, candidate
generation, synthetic-data selection, or a GaMS prompt. Only aggregate
categories and metrics may steer later generation when a work order explicitly
permits it. Historical FLEURS v1 evidence is deprecated because it did not
preserve unique audio occurrences.

Immutable gates and final blind tests must not be used for early stopping,
hyperparameter selection, or checkpoint selection.

### Final blind test

Used only for major release decisions. It must not influence training, prompts, selection, or hyperparameters.

### Synthetic holdouts

Hold out templates, lexical families, minimal-pair groups, and TTS conditions where possible.
The current corpus-v2 independent synthetic holdout is diagnostic only. It has
been scored with the untouched base model but must not enter selected training
or be cited as real-speech generalization evidence.

### Multilingual regression

Evaluation-only. It verifies preservation of shared model behavior and must not silently become Slovenian training data.

## Text normalization

A versioned normalization policy must define:

- Unicode normalization;
- whitespace;
- punctuation;
- capitalization;
- numbers, dates, times, units, and abbreviations;
- acceptable variants.

Raw and normalized metrics must be reported separately when normalization materially changes results.

## GaMS leakage controls

- Do not include immutable-gate or final-test text in GaMS prompts.
- Do not include raw FLEURS or ARTUR-J development-gate references in any
  project-generated curriculum prompt.
- Deduplicate generated candidates against public and protected evaluation text.
- Store hashes or similarity indexes needed to audit leakage.
- A candidate too close to protected evaluation text is rejected.
- For the prompt-column active-curriculum experiment, round-2 steering may use
  synthetic candidate-pool references and hypotheses, substitution/deletion/
  insertion clusters, failed phenomenon counts, and aggregate real-gate category
  counts. It must not include raw real-gate references or synthetic-holdout raw
  errors.

## TTS policy

The TTS system's license and output rights must permit the intended training and publication use.

The selected initial TTS path is external Piper `OHF-Voice/piper1-gpl` with the
`sl_SI-artur-medium` voice from `rhasspy/piper-voices`. Piper is
GPL-3.0-or-later and the voice/source licensing metadata is inconsistent across
the voice repository, model card, and ARTUR source record. Until later legal
review, generated audio follows the conservative ARTUR CC BY-SA 4.0 attribution
and publication policy.

Generated Piper audio is local/internal synthetic material in M2. It must not
be committed, published, or labeled as real speech or Apache-2.0 content.

The first adaptation phase uses:

```text
spoken_text == target_text
```

Normalization tasks are added only as separately tagged curricula.

## Retention

Generated candidates that are not selected need not be retained indefinitely. Keep enough metadata to reproduce generation when licensing and model availability permit.

Project-generated curriculum corpora are local synthetic artifacts. Their JSONL
records, audio, manifests, hypotheses, deltas, and checkpoints remain ignored
unless a later publication work order and rights review explicitly authorize
release.

The corpus-v2 candidate source and independent synthetic holdout now have
privacy-safe text and audio admission evidence plus a `SCORING_AUTHORIZED`
certificate. That certificate permits ASR scoring and selected-training
construction only; it does not authorize model training, data publication,
checkpoint promotion, or `TRAINING_ELIGIBLE`.

The Round 1 v1 candidate pool
`0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17`,
synthetic holdout
`ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9`, and
selected training manifest
`92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4`
are permanently retired for future training, steering, model comparison,
early stopping, generator steering, promotion, or public corpus-quality claims.
They may remain in ignored local storage only for audit and safe regression-test
design.

Private real-speech data follows institutional retention and access policy. Public logs must never contain raw private text.

## Publication

A dataset release requires:

- rights and license review;
- privacy review;
- documentation of generation and filtering;
- clear synthetic/real labels;
- removal of private paths and identifiers;
- checksums;
- a data card;
- human approval.

A model release does not imply permission to release its training data.
