# Scale-8000 vs FLEURS/ARTUR-J Vocabulary Coverage

Classification: `VOCAB_COVERAGE_ANALYSIS_COMPLETE_LOW_GATE_COVERAGE`

This Work Order 0048 result is a read-only aggregate corpus diagnostic. It did not train a model, evaluate ASR, generate text, select examples, or create a curriculum. No real-gate reference, vocabulary, missing-word list, local manifest, or local path is committed.

## Input bindings

| Dataset | Role | Rows | Source SHA256 |
|---|---|---:|---|
| scale-8000 | synthetic training text | 64000 | `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd` |
| FLEURS-v2 | immutable real gate | 834 | `8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513` |
| ARTUR-J | immutable real gate | 256 | `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763` |

## Normalization and tokenization

- Normalizer: `sl-asr-normalization-v1`.
- Primary mode: `normalized_word_forms`.
- Token policy: NFC, repository normalization, casefolding, punctuation splitting, at least one Unicode letter, Slovenian diacritics preserved, and numeric-only tokens excluded.
- No lemmatization or stemming.
- P50 and P95 use the nearest-rank method.

## Dataset sizes

| Dataset | Rows | Total word tokens | Unique word forms | Type-token ratio | Mean words/row | P50 words/row | P95 words/row |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale-8000 | 64000 | 504061 | 27694 | 0.054942 | 7.88 | 7 | 14 |
| FLEURS-v2 | 834 | 16254 | 3535 | 0.217485 | 19.49 | 18 | 34 |
| ARTUR-J | 256 | 2289 | 1166 | 0.509393 | 8.94 | 8 | 17 |

## Vocabulary overlap

| Comparison | Shared unique word forms | Coverage of right-side unique vocab | Token-mass coverage of right-side tokens |
|---|---:|---:|---:|
| scale-8000 → FLEURS-v2 | 2144 | 60.65% | 78.80% |
| scale-8000 → ARTUR-J | 849 | 72.81% | 84.40% |
| FLEURS-v2 ↔ ARTUR-J | 320 | 27.44% | 52.51% |

For the first two rows, the right side is the real gate. In the third row, ARTUR-J is the right-side denominator. The three-way intersection contains 312 unique normalized word forms.

## Missing aggregate only

| Gate | Unique forms absent from scale-8000 | Absent-form rate | Gate tokens using absent forms | Absent token-mass rate |
|---|---:|---:|---:|---:|
| FLEURS-v2 | 1391 | 39.35% | 3446 | 21.20% |
| ARTUR-J | 317 | 27.19% | 357 | 15.60% |

No missing form is included in this report.

## Secondary normalized alphanumeric mode

This optional mode includes numeric-only tokens and remains aggregate-only.

| Dataset | Total alphanumeric tokens | Unique alphanumeric forms |
|---|---:|---:|
| scale-8000 | 507961 | 27859 |
| FLEURS-v2 | 16517 | 3602 |
| ARTUR-J | 2289 | 1166 |

| Comparison | Unique-form coverage | Token-mass coverage |
|---|---:|---:|
| scale-8000 → FLEURS-v2 | 60.77% | 78.71% |
| scale-8000 → ARTUR-J | 72.81% | 84.40% |

## Interpretation

- Scale-8000 covers 60.65% of FLEURS-v2 unique normalized forms, which falls below the precommitted 80% warning threshold. It is not broad enough relative to FLEURS-v2 under this narrow surface-form test.
- Scale-8000 covers 72.81% of ARTUR-J unique normalized forms, which falls below the precommitted 80% warning threshold. It is not broad enough relative to ARTUR-J under this narrow surface-form test.
- The uncovered token mass is large enough to remain a plausible lexical limitation, but this vocabulary-only analysis cannot attribute ASR errors.
- These results measure only normalized surface vocabulary. They do not measure meaning, pronunciation, acoustics, or ASR quality, and distinct Slovenian inflections remain distinct forms.
- No corpus change or generation steering is authorized by this result. A safe follow-up would use independently sourced or independently authored Slovenian material to test lexical and morphological breadth, without transferring real-gate words, sentences, or missing-form lists into prompts or selection.

## Safety

- Raw FLEURS/ARTUR-J references committed: no.
- Full or missing real-gate word lists committed: no.
- Local manifests or absolute paths committed: no.
- Used for training selection or curriculum construction: no.
- Used for GaMS prompting or new text generation: no.
- Training or ASR evaluation run: no.
