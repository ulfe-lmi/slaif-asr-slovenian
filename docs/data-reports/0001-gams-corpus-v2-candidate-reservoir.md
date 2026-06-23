# GaMS Corpus-v2 Candidate Reservoir

Status: DRAFT — awaiting native-speaker linguistic review.

This report is privacy-safe. It does not include raw generated sentences, candidate IDs, protected references, local paths, raw GaMS output, audio paths, or hypotheses.

## Identity

- Corpus ID: `sl-corpus-v2-gams-candidate-reservoir-v1`
- Model: `cjvt/GaMS3-12B-Instruct`
- Revision: `1d0b27af5748784482600d24779409e7e1dc9adc`
- Configuration SHA256: `1dca33057076a19447df22493088452fbb81c1c0a1a8ac94e541493fd3933e8c`

## Funnel

- Requested rows: 480
- Raw extracted rows: 476
- Retained pre-review rows: 415
- Minimum structurally admissible target: 320
- Shortfall: 0

## Validation

- Validator status: `DRAFT`
- Decision reasons: `missing_linguistic_review_file`
- Fuzzy review pairs: 0

## Review Pack

- Rows: 415
- Review outcomes prefilled: no
- Human linguistic review is required before any TTS, scoring, selection, or training.

## GPU Measurement

- Monitor samples: 2128
- Mean utilization: 90.195
- Median utilization: 93.0
- P95 utilization: 96.0
- Fraction >=80%: 0.961466
- Peak memory MiB: 11913.0

## Limitations

- Native-speaker review remains outstanding.
- No synthetic holdout exists.
- Acoustic suitability remains untested.
- No data is authorized for training.
