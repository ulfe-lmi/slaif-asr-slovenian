# Evaluation Protocol

## Purpose

Evaluation determines whether a challenger improves Slovenian recognition without unacceptable loss of streaming or multilingual behavior.

## Required checkpoint identity

Every result records:

- base model repository and revision;
- base checkpoint checksum;
- parent accepted checkpoint checksum;
- challenger checksum;
- NeMo revision;
- repository commit;
- configuration hash;
- trainable-parameter summary.

## Slovenian metrics

Report:

- raw WER;
- normalized WER;
- CER;
- `č`, `š`, and `ž` error rates;
- deletion and insertion rates;
- proper-name WER;
- number/date/time accuracy;
- punctuation accuracy;
- capitalization accuracy;
- foreign-script leakage;
- silence/noise hallucination rate.

Active-curriculum reports must also distinguish:

- corpus WER from total word edits divided by total reference words;
- corpus CER from total character edits divided by total reference characters;
- mean utterance WER/CER;
- median utterance WER/CER;
- empty-hypothesis counts.

Do not label mean utterance WER as corpus WER.

## Streaming settings

Evaluate every release candidate at the supported context settings:

```text
[56,0]
[56,1]
[56,3]
[56,6]
[56,13]
```

Record the corresponding advertised latency point with the result.

Track:

- final WER/CER;
- first-token latency;
- final-token latency;
- partial-transcript churn;
- disagreement between low-latency and balanced settings;
- real-time factor;
- peak GPU memory.

## Data roles

Separate reports for:

- synthetic candidate pool;
- selected synthetic hard set;
- controller-development real Slovenian;
- immutable real-Slovenian gate;
- final blind test;
- multilingual regression.

Never present a synthetic-only score as real-world Slovenian performance.

## Acceptance comparison

Compare the challenger with its parent accepted checkpoint, not only with the original base.

Initial project guardrails, subject to revision through an ADR:

- targeted hard-set improvement: at least 10% relative;
- immutable Slovenian gate: no more than 0.3 absolute WER regression;
- multilingual macro WER after shared-weight training: no more than 0.5 absolute WER regression;
- no new systematic foreign-script leakage;
- no increased silence hallucination;
- no material low-latency degradation;
- parameter-diff integrity passes.

Statistical uncertainty should be reported using paired bootstrap intervals when the sample size supports it.

## Normalization

The report must state the normalization policy and its version. Raw and normalized results remain distinguishable.

## Blind-test discipline

The final blind test is opened only for a named milestone and human-approved release decision. Once used, its status and resulting potential tuning exposure are recorded.

## Output artifacts

Commit only summary tables and privacy-safe aggregate reports. Per-utterance outputs containing private references remain in controlled storage.
