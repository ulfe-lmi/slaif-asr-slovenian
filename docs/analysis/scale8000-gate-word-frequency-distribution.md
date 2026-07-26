# Scale-8000 vs FLEURS/ARTUR-J Word-Frequency Distribution

Classification: `WORD_FREQUENCY_DISTRIBUTION_ANALYSIS_COMPLETE`

This Work Order 0049 result is a read-only, aggregate-only corpus diagnostic. It separates vocabulary support, shared-form frequency, rank, concentration, long-tail, and finite-sample effects. It is not an equivalence or representativeness claim.

## Input bindings

| Dataset | Role | Rows | Source SHA256 |
|---|---|---:|---|
| scale-8000 | synthetic_training_text | 64000 | `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd` |
| FLEURS-v2 | immutable_real_gate | 834 | `8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513` |
| ARTUR-J | immutable_real_gate | 256 | `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763` |

Every dataset identity, row count, and SHA256 was checked before analysis. A mismatch fails closed as `EXPERIMENT_INVALID` before either report is written.

## Normalization and analysis units

- Normalizer: `sl-asr-normalization-v1`.
- Analysis units: normalized surface-word forms, not lemmas, morphemes, or tokenizer units.
- NFC normalization, casefold/lowercase, punctuation splitting, and preserved Slovenian diacritics.
- Numeric-only tokens are excluded; mixed letter-number tokens are retained.
- No lemmatization or stemming.

## Dataset distribution summaries

| Dataset | Rows | Total word tokens | Unique word forms |
|---|---:|---:|---:|
| scale-8000 | 64000 | 504061 | 27694 |
| FLEURS-v2 | 834 | 16254 | 3535 |
| ARTUR-J | 256 | 2289 | 1166 |

### scale-8000 frequency-of-frequency profile

| Occurrences | Unique forms | Fraction of forms | Tokens contributed | Fraction of tokens |
|---|---:|---:|---:|---:|
| exactly 1 | 11600 | 41.89% | 11600 | 2.30% |
| exactly 2 | 4150 | 14.99% | 8300 | 1.65% |
| 3–5 | 4577 | 16.53% | 17146 | 3.40% |
| 6–10 | 2668 | 9.63% | 20296 | 4.03% |
| 11–50 | 3499 | 12.63% | 77467 | 15.37% |
| more than 50 | 1200 | 4.33% | 369252 | 73.26% |

### FLEURS-v2 frequency-of-frequency profile

| Occurrences | Unique forms | Fraction of forms | Tokens contributed | Fraction of tokens |
|---|---:|---:|---:|---:|
| exactly 1 | 452 | 12.79% | 452 | 2.78% |
| exactly 2 | 1050 | 29.70% | 2100 | 12.92% |
| 3–5 | 1657 | 46.87% | 5365 | 33.01% |
| 6–10 | 245 | 6.93% | 1746 | 10.74% |
| 11–50 | 105 | 2.97% | 2075 | 12.77% |
| more than 50 | 26 | 0.74% | 4516 | 27.78% |

### ARTUR-J frequency-of-frequency profile

| Occurrences | Unique forms | Fraction of forms | Tokens contributed | Fraction of tokens |
|---|---:|---:|---:|---:|
| exactly 1 | 892 | 76.50% | 892 | 38.97% |
| exactly 2 | 140 | 12.01% | 280 | 12.23% |
| 3–5 | 82 | 7.03% | 299 | 13.06% |
| 6–10 | 30 | 2.57% | 218 | 9.52% |
| 11–50 | 19 | 1.63% | 394 | 17.21% |
| more than 50 | 3 | 0.26% | 206 | 9.00% |

### Concentration and entropy

| Dataset | Top-10 mass | Top-50 mass | Top-100 mass | Top-500 mass | Shannon entropy (bits) | Normalized entropy | Simpson concentration | Effective vocabulary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scale-8000 | 24.49% | 39.46% | 45.93% | 62.90% | 10.4187 | 0.7060 | 0.008620 | 1368.77 |
| FLEURS-v2 | 19.68% | 33.07% | 38.37% | 55.11% | 10.2326 | 0.8681 | 0.005387 | 1203.14 |
| ARTUR-J | 18.61% | 35.21% | 44.34% | 70.90% | 9.1886 | 0.9020 | 0.005465 | 583.52 |

Top-k values are aggregate token mass only. Effective k is the available vocabulary when it is smaller than the requested k.

## Pairwise full-union distributions

| Comparison | JSD (base 2) | Hellinger | Total variation | Probability-mass overlap | Support TV component | Shared-frequency TV component |
|---|---:|---:|---:|---:|---:|---:|
| scale-8000 vs FLEURS-v2 | 0.5025 | 0.6797 | 0.6424 | 35.76% | 0.3207 | 0.3217 |
| scale-8000 vs ARTUR-J | 0.5515 | 0.7134 | 0.6816 | 31.84% | 0.3354 | 0.3463 |
| FLEURS-v2 vs ARTUR-J | 0.5807 | 0.7526 | 0.6650 | 33.50% | 0.5264 | 0.1386 |

## Pairwise shared-vocabulary conditional distributions

| Comparison | Shared forms | Shared mass in A | Shared mass in B | Conditional JSD | Conditional Hellinger | Conditional overlap |
|---|---:|---:|---:|---:|---:|---:|
| scale-8000 vs FLEURS-v2 | 2144 | 57.06% | 78.80% | 0.2576 | 0.4445 | 51.60% |
| scale-8000 vs ARTUR-J | 849 | 48.52% | 84.40% | 0.2971 | 0.4823 | 49.14% |
| FLEURS-v2 vs ARTUR-J | 320 | 42.20% | 52.51% | 0.1082 | 0.2811 | 70.29% |

## Shared-form frequency ratios

Absolute log2 ratios use symmetric `min(p_A, p_B)` mass weighting.

| Comparison | Weighted mean | Weighted median | Within factor 2 | Within factor 4 | Within factor 10 |
|---|---:|---:|---:|---:|---:|
| scale-8000 vs FLEURS-v2 | 0.9904 | 0.6182 | 63.87% | 85.40% | 97.22% |
| scale-8000 vs ARTUR-J | 1.0320 | 0.8388 | 63.66% | 88.31% | 95.81% |
| FLEURS-v2 vs ARTUR-J | 0.6558 | 0.4939 | 74.99% | 96.15% | 99.17% |

## Frequency-rank similarity

Spearman ranks use descending counts, average ranks for ties, and codepoint ordering only as an internal deterministic tie-break. No ranked forms are emitted.

| Comparison | All shared eligible | All shared Spearman | Count≥2 eligible | Count≥2 Spearman |
|---|---:|---:|---:|---:|
| scale-8000 vs FLEURS-v2 | 2144 | 0.2816 | 1631 | 0.3076 |
| scale-8000 vs ARTUR-J | 849 | 0.3853 | 236 | 0.5404 |
| FLEURS-v2 vs ARTUR-J | 320 | 0.4888 | 143 | 0.5275 |

## Top-k vocabulary agreement

### scale-8000 vs FLEURS-v2

| Requested k | Effective k in A | Effective k in B | Shared forms | Jaccard | Overlap coefficient |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 10 | 6 | 0.4286 | 0.6000 |
| 50 | 50 | 50 | 29 | 0.4085 | 0.5800 |
| 100 | 100 | 100 | 54 | 0.3699 | 0.5400 |
| 500 | 500 | 500 | 142 | 0.1655 | 0.2840 |
| 1000 | 1000 | 1000 | 231 | 0.1306 | 0.2310 |

### scale-8000 vs ARTUR-J

| Requested k | Effective k in A | Effective k in B | Shared forms | Jaccard | Overlap coefficient |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 10 | 6 | 0.4286 | 0.6000 |
| 50 | 50 | 50 | 27 | 0.3699 | 0.5400 |
| 100 | 100 | 100 | 49 | 0.3245 | 0.4900 |
| 500 | 500 | 500 | 116 | 0.1312 | 0.2320 |
| 1000 | 1000 | 1000 | 231 | 0.1306 | 0.2310 |

### FLEURS-v2 vs ARTUR-J

| Requested k | Effective k in A | Effective k in B | Shared forms | Jaccard | Overlap coefficient |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 10 | 9 | 0.8182 | 0.9000 |
| 50 | 50 | 50 | 27 | 0.3699 | 0.5400 |
| 100 | 100 | 100 | 45 | 0.2903 | 0.4500 |
| 500 | 500 | 500 | 117 | 0.1325 | 0.2340 |
| 1000 | 1000 | 1000 | 181 | 0.0995 | 0.1810 |

## Sampling uncertainty

Bootstrap unit: row/utterance; replicates: 1000; seed: 480049; interval: 2.5th, 50th, and 97.5th percentiles. Rows are sampled independently with replacement while preserving each dataset row count.

### Within-corpus sampling envelopes

| Dataset | Full-union JSD median [95% interval] | Conditional JSD median [95% interval] | Overlap median [95% interval] | Count≥2 Spearman median [95% interval] |
|---|---:|---:|---:|---:|
| scale-8000 | 0.0223 [0.0219, 0.0227] | 0.0108 [0.0105, 0.0110] | 0.9282 [0.9267, 0.9295] | 0.8597 [0.8550, 0.8637] |
| FLEURS-v2 | 0.0924 [0.0795, 0.1074] | 0.0518 [0.0448, 0.0596] | 0.7821 [0.7635, 0.8011] | 0.4342 [0.3450, 0.5197] |
| ARTUR-J | 0.2135 [0.1789, 0.2497] | 0.0574 [0.0464, 0.0702] | 0.6619 [0.6223, 0.6992] | 0.6177 [0.5031, 0.7063] |

### Cross-corpus bootstrap summaries

| Comparison | Full-union JSD median [95% interval] | Conditional JSD median [95% interval] | Overlap median [95% interval] | Count≥2 Spearman median [95% interval] | Envelope label | Excess-divergence ratio |
|---|---:|---:|---:|---:|---|---:|
| scale-8000 vs FLEURS-v2 | 0.5165 [0.5111, 0.5226] | 0.2574 [0.2504, 0.2651] | 0.3479 [0.3407, 0.3546] | 0.3015 [0.2630, 0.3427] | `ABOVE_SAMPLING_ENVELOPE` | 4.81 |
| scale-8000 vs ARTUR-J | 0.5829 [0.5687, 0.5981] | 0.3155 [0.2968, 0.3354] | 0.2954 [0.2797, 0.3091] | 0.4379 [0.3596, 0.5131] | `ABOVE_SAMPLING_ENVELOPE` | 2.33 |
| FLEURS-v2 vs ARTUR-J | 0.6092 [0.5955, 0.6236] | 0.1279 [0.1126, 0.1449] | 0.3083 [0.2919, 0.3237] | 0.5177 [0.4207, 0.6100] | `ABOVE_SAMPLING_ENVELOPE` | 2.44 |

## Pairwise interpretation

### scale-8000 vs FLEURS-v2

1. Shared probability mass: 35.76%.
2. Full-distribution difference: JSD 0.5025, Hellinger 0.6797, and total variation 0.6424.
3. Shared-form frequency difference: conditional JSD 0.2576; 63.87% of symmetric shared mass is within a factor of two.
4. Frequent shared-form ordering: count≥2 Spearman 0.3076 over 1631 eligible forms.
5. Sampling envelope: `ABOVE_SAMPLING_ENVELOPE`; cross-corpus median JSD is 4.81 times the maximum relevant within-corpus JSD p97.5.
6. Total-variation decomposition is larger for different frequencies among shared forms.

### scale-8000 vs ARTUR-J

1. Shared probability mass: 31.84%.
2. Full-distribution difference: JSD 0.5515, Hellinger 0.7134, and total variation 0.6816.
3. Shared-form frequency difference: conditional JSD 0.2971; 63.66% of symmetric shared mass is within a factor of two.
4. Frequent shared-form ordering: count≥2 Spearman 0.5404 over 236 eligible forms.
5. Sampling envelope: `ABOVE_SAMPLING_ENVELOPE`; cross-corpus median JSD is 2.33 times the maximum relevant within-corpus JSD p97.5.
6. Total-variation decomposition is larger for different frequencies among shared forms.

### FLEURS-v2 vs ARTUR-J

1. Shared probability mass: 33.50%.
2. Full-distribution difference: JSD 0.5807, Hellinger 0.7526, and total variation 0.6650.
3. Shared-form frequency difference: conditional JSD 0.1082; 74.99% of symmetric shared mass is within a factor of two.
4. Frequent shared-form ordering: count≥2 Spearman 0.5275 over 143 eligible forms.
5. Sampling envelope: `ABOVE_SAMPLING_ENVELOPE`; cross-corpus median JSD is 2.44 times the maximum relevant within-corpus JSD p97.5.
6. Total-variation decomposition is larger for vocabulary-support mismatch.

These labels refer only to the predeclared bootstrap divergence envelope. Statistical separation is not an equivalence, identity, language-distribution, or representativeness claim. Effect sizes and sampling envelopes remain visible for interpretation.

## Safety

- No raw FLEURS or ARTUR-J references are committed.
- No full, missing, top-k, or ranked real-gate word lists are committed.
- No per-word frequencies, ratios, or bootstrap values are committed.
- No local paths or manifests are committed.
- No gate information was used for prompting, selection, curriculum construction, or training.
- No text generation, corpus modification, model training, or ASR evaluation was run.

## Known limitations

- Surface forms only; no lemmatization.
- Empirical unigram analysis only.
- The small ARTUR-J sample produces wider uncertainty.
- Frequency similarity does not prove acoustic or ASR similarity.
