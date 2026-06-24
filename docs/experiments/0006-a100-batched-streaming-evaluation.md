# Experiment 0006: A100 Batched Streaming Evaluation

Status: **completed in PR; pending strategic review**

This experiment establishes the first valid untouched-base FLEURS-v2 ASR baseline and a parity-proven A100 streaming batch policy. It does not score the corpus-v2 candidate reservoir and does not train a model.

## Input Identity

- Checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- FLEURS-v2 manifest SHA256: `8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513`
- ARTUR-J manifest SHA256: `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763`
- Context: `[56, 3]`
- Target language: `sl-SI`
- Precision: FP32, TF32 disabled

## Official Batch-1 Parity

- Rows: 32
- Exact mismatches: 0
- Normalized mismatches: 0
- Metric differences: 0
- Result: PASSED

## FLEURS-v2 Sweep

| Batch | Bucketed | Status | Exact mismatch | End-to-end RTF | Active RTF | Speedup | Padding ratio | Mean util | P95 util | Peak memory MiB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yes | PASSED | 0 | 0.106087 | 0.098128 | 1.000 | 1.0 | 37.057838 | 40.0 | 5517.0 |
| 2 | yes | PASSED | 15 | 0.058411 | 0.050627 | 1.816 | 1.001578 | 61.108142 | 73.0 | 5525.0 |
| 4 | yes | PASSED | 34 | 0.033355 | 0.025539 | 3.181 | 1.004559 | 56.930881 | 77.0 | 5541.0 |
| 8 | yes | PASSED | 63 | 0.021924 | 0.01423 | 4.839 | 1.010167 | 57.140678 | 92.0 | 5605.0 |
| 16 | yes | PASSED | 89 | 0.018565 | 0.010676 | 5.714 | 1.022207 | 52.468254 | 95.0 | 5705.0 |
| 32 | yes | PASSED | 116 | 0.016029 | 0.008342 | 6.619 | 1.048283 | 47.614319 | 97.0 | 5889.0 |
| 64 | yes | PASSED | 117 | 0.015695 | 0.007899 | 6.759 | 1.103488 | 46.218009 | 98.0 | 6423.0 |
| 128 | yes | PASSED | 119 | 0.015221 | 0.0075 | 6.970 | 1.160807 | 44.77561 | 98.0 | 7935.0 |

## Selected Policy

- Batch size: 1
- Duration bucketing: disabled
- Scientific classification: `A100_BATCHED_STREAMING_NOT_EQUIVALENT`
- Speedup vs batch 1: 1.042892
- Bucketed throughput: 9.426266 audio seconds per wall second
- Unbucketed throughput: 9.830573 audio seconds per wall second

## FLEURS-v2 Untouched Base Metrics

- Raw corpus WER/CER: 62.722 / 19.659
- Normalized corpus WER/CER: 52.703 / 16.423
- Mean utterance WER/CER: 63.832 / 20.392
- Median utterance WER/CER: 62.963 / 17.391
- Empty hypotheses: 1
- Rows: 834
- Audio duration: 8173.14 s
- Wall time: 831.400186 s
- RTF: 0.101723

## ARTUR-J Confirmation

- Selected batch exact mismatches: 0
- Selected batch metric differences: 0

## Notes

- Batch size 1 remains the scientific reference mode.
- The selected A100 policy is not an RTX 2080 Ti policy.
- The corpus-v2 candidate reservoir was not scored.
- Raw references, hypotheses, local manifests, logs, and monitor CSVs remain ignored.
