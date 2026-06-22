# Data Policy

## Purpose

This policy governs real speech, synthetic speech, transcripts, generated text, manifests, metrics, and released dataset artifacts.

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

### Immutable gate

Used after rounds for acceptance. GaMS receives aggregate categories only, not raw reference sentences.

Current immutable real development gates are the complete FLEURS Slovenian test
split and the ARTUR-J public-speech project gate. Neither gate may enter
training, candidate generation, synthetic-data selection, or a GaMS prompt.
Only aggregate categories and metrics may steer later generation when a work
order explicitly permits it.

### Final blind test

Used only for major release decisions. It must not influence training, prompts, selection, or hyperparameters.

### Synthetic holdouts

Hold out templates, lexical families, minimal-pair groups, and TTS conditions where possible.

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
