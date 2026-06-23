# Testing Strategy

## Principle

Tests are evidence only when they cover the claim being made.

## Test layers

### Repository hygiene

- no forbidden large files;
- no secrets;
- no private/local paths;
- valid configuration syntax;
- documentation links resolve.

### Unit tests

Planned units include:

- text normalization;
- candidate schema validation;
- deduplication;
- manifest construction;
- prompt-index selection;
- trainable-parameter filtering;
- checkpoint-diff verification;
- metric calculation;
- acceptance-rule evaluation.

### Integration tests

Planned integrations include:

- validated candidate text -> Piper synthesis -> 16 kHz WAV -> NeMo manifest;
- checkpoint restoration;
- tokenizer round trip;
- manifest -> NeMo dataloader;
- one forward and RNNT-loss step;
- save and restore challenger checkpoint;
- offline and cache-aware streaming inference;
- adapter extraction and reapplication.

### GPU tests

GPU tests must record environment details. They should be tagged so CPU-only CI can skip them explicitly while a GPU runner executes them separately.

### End-to-end experiment tests

A bounded fixture should prove:

```text
candidate -> synthetic audio fixture -> manifest -> baseline inference
-> selective update -> checkpoint save -> streaming inference -> gate report
```

The fixture proves orchestration, not production accuracy.

## Negative-path requirements

Tests should prove failure for:

- missing or invalid `target_lang`;
- unsupported prompt;
- non-16-kHz or stereo audio when strict mode is used;
- empty transcript;
- tokenizer corruption of Slovenian characters;
- unexpected trainable parameters;
- forbidden checkpoint changes;
- protected-evaluation leakage;
- missing license/provenance fields;
- attempted acceptance with missing required metrics;
- publication without release approval.

## Performance tests

Track:

- real-time factor;
- GPU memory;
- batch throughput;
- latency setting;
- output stability;
- checkpoint load time.

Performance numbers must name hardware and software revisions.

## Reporting vocabulary

Use exactly:

- `PASSED`
- `FAILED`
- `SKIPPED`
- `NOT_RUN`
- `ENVIRONMENT_BLOCKED`
- `OUT_OF_SCOPE`

Do not summarize partial verification as “all tests passed.”

## CI staging

The intended CI progression is:

1. docs and repository hygiene: **implemented by the CPU CI baseline**;
2. CPU unit tests: **implemented by the CPU CI baseline**;
3. optional containerized NeMo smoke tests;
4. scheduled or manually approved GPU verification;
5. release-only full evaluation.

GPU and external-download jobs should be explicit to control cost and supply-chain risk.

The CPU CI baseline runs only tracked-file checks, repository unit tests, Python
compilation, shell syntax checks, and Git whitespace checks. It deliberately
does not install NeMo, download checkpoints or audio, use Hugging Face access,
detect GPUs, or validate model restoration and streaming inference. M1 GPU
evidence remains the RTX 2080 Ti verification recorded by the runtime repair
work order.

M2 Piper verification is a manual GPU evidence path, not CPU CI. It must record
Piper ONNX Runtime provider status, physical GPU 0 selection, GPU 1 non-use,
voice checksum verification, native and resampled WAV validation, provenance
output, NeMo manifest hash, and Nemotron smoke output. It does not prove ASR
quality and must not be represented as training.

The GaMS active-curriculum path adds CPU-testable checks for strict generated
JSON, duplicate and protected-text rejection, deterministic active selection,
promotion and rollback decisions, and corpus-versus-mean metric separation. The
actual GaMS, Piper, Nemotron pre-scoring, prompt-column training, and fixed-gate
evaluation phases remain manual GPU verification until a later self-hosted
runner exists.

The real-gate evaluation suite adds CPU-testable checks for pinned FLEURS and
ARTUR source identities, FLEURS occurrence-index identity planning, repeated
FLEURS source-ID collision handling, ARTUR TRS parsing, deterministic gate
selection, Slovenian normalization, raw/normalized metric separation,
privacy-safe metadata, and leakage guards. The full FLEURS v2 gate verifier
checks row count, unique sample IDs, unique audio paths, source-row index range,
manifest hash, and WAV validity. Full FLEURS and ARTUR-J inference remains
manual GPU evidence.

Training-data admission now has a reusable fail-closed text-stage validator.
Its CPU tests include adversarial fixtures that reproduce the Round 1 v1
failure with safe synthetic strings: metadata identifiers embedded in speech,
same template with different row numbers, same body with different artificial
prefixes or suffixes, train/holdout body overlap with different IDs,
suspicious threshold-boundary pairs, malformed Slovenian slot insertion, and
acoustic variants of one utterance crossing partitions.

The validator does not prove audio quality and cannot emit
`TRAINING_ELIGIBLE`. Future test work must add adversarial fixtures for the
audio-validation and certificate stages before any corpus can be used for
promotion-oriented training.
