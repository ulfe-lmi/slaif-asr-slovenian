# Corpus-v2 Independent Synthetic Holdout

Status: DRAFT — awaiting whole-file human holdout decision.

This privacy-safe report does not include raw generated sentences, candidate or holdout IDs, protected references, hypotheses, or local paths.

## Identity

- Corpus ID: `sl-corpus-v2-independent-synthetic-holdout-v1`
- Model: `cjvt/GaMS-9B-Instruct`
- Revision: `292744023fa0b7ccc7ae2c3c885a67468e49fa03`
- Configuration SHA256: `1bc1f9bd7c4a7f5490a80d8b3aceb63944738dbce5a23a21a694f61c9ee83a21`
- Fixed holdout SHA256: `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`

## Funnel

- Requested generation rows: 160
- Generated schema rows: 356
- Fixed holdout rows: 96
- Rejected rows: 260
- Rejection counts: `{"deterministic_selection_overflow": 145, "schema_invalid": 3, "surface_duplicate": 15, "token_ngram_concentration": 97}`

## Validation

- Validator status: `DRAFT`
- Decision reasons: `missing_linguistic_review`
- Cross-partition overlap counts: `{}`
- Fuzzy review pairs: 0

## Review Capsule

- Rows: 96
- Review outcome prefilled: no
- Required human decision: ACCEPT or REJECT for the exact fixed-holdout hash.

## A100 Measurement

- Generation wall time seconds: 183.952
- Generated rows per minute: 116.117
- Retained rows per minute: 31.313
- Prompt batch size used: 4
- Monitor samples: 709
- Mean utilization: 88.055
- Median utilization: 88.0
- P95 utilization: 91.0
- Fraction >=80%: 0.994358
- Peak memory MiB: 9465.0

## Limitations

- Human holdout review remains outstanding.
- Acoustic suitability remains untested.
- No ASR scoring, training selection, certificate, or model training occurred.
