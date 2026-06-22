# Experiment 0003: Real Slovenian Baseline

Status: **completed for untouched-base development-gate baseline**

## Purpose

This experiment establishes immutable real-speech development gates and the
untouched Nemotron baseline on them. It does not train, fine-tune, publish, or
accept a challenger.

## Gates

| Gate | Identifier | Source | Policy |
|---|---|---|---|
| FLEURS Slovenian full test | `fleurs-sl-si-test-full-v1` | `google/fleurs` `sl_si` `test` at `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | complete split |
| ARTUR-J public speech | `artur-j-public-gate-v1` | CLARIN.SI handles `11356/1772` and `11356/1776` | deterministic 256-segment project gate |

Both gates are immutable development gates, not final blind tests.

## Baseline Model

- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- Context: `[56,3]`
- Target language: `sl-SI`
- Batch size: 1 canonical comparison mode

## Metric Contract

Reports use `sl-asr-normalization-v1` and keep raw and normalized metrics
separate:

- raw corpus WER and CER;
- normalized corpus WER and CER;
- mean utterance WER and CER;
- median utterance WER and CER;
- empty-hypothesis count.

Corpus rates are computed from summed edit counts, not from averaged utterance
percentages.

## Gate Metadata

Committed privacy-safe metadata:

- FLEURS:
  [`docs/evaluation-gates/fleurs-sl-si-test-full-v1.metadata.json`](../evaluation-gates/fleurs-sl-si-test-full-v1.metadata.json)
- ARTUR-J:
  [`docs/evaluation-gates/artur-j-public-gate-v1.metadata.json`](../evaluation-gates/artur-j-public-gate-v1.metadata.json)

Local ignored artifacts:

- manifests and reference sidecars under `runs/evaluation-gates/`;
- audio and ARTUR archives under `runs/evaluation-gates/`;
- raw hypotheses and per-sample outputs under `runs/evaluation-baselines/`.

## Current Result

The untouched base checkpoint was evaluated with physical GPU 0 selected by
`CUDA_VISIBLE_DEVICES=0`. GPU 1 stayed at idle baseline. Metrics are development
gate measurements, not benchmark or production-readiness claims.

| Gate | Raw corpus WER | Normalized corpus WER | Raw corpus CER | Normalized corpus CER | Mean utterance WER | Median utterance WER | Empty hypotheses |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLEURS full test | 62.679 | 52.734 | 19.599 | 16.423 | 53.541 | 52.941 | 0 |
| ARTUR-J gate | 74.585 | 67.453 | 32.234 | 29.016 | 76.555 | 75.000 | 12 |

Runtime:

| Gate | Rows | Audio duration | Wall time | Real-time factor | Observed GPU 0 VRAM |
|---|---:|---:|---:|---:|---:|
| FLEURS full test | 834 | 8173.140 s | 889.083 s | 0.108781 | about 2881 MiB |
| ARTUR-J gate | 256 | 1049.590 s | 167.946 s | 0.160011 | about 2861 MiB |

## Micro-Proof Diagnostic

The prompt-column micro-overfit checkpoint remains unaccepted. The ignored local
artifact and integrity report were available, so it was evaluated as
`unaccepted_micro_overfit_diagnostic`; it is not promoted by this experiment.

| Gate | Base normalized WER | Diagnostic normalized WER | Base normalized CER | Diagnostic normalized CER | Empty hypotheses base/diagnostic |
|---|---:|---:|---:|---:|---:|
| FLEURS full test | 52.734 | 66.961 | 16.423 | 24.916 | 0 / 0 |
| ARTUR-J gate | 67.453 | 76.190 | 29.016 | 32.654 | 12 / 3 |

The earlier public-smoke regression generalized to both full real development
gates in aggregate WER/CER. The diagnostic checkpoint remains unaccepted and is
not a valid parent for later rounds.

## Safety

Local manifests, audio, raw references, raw hypotheses, archives, and per-sample
outputs remain ignored. Committed metadata contains only privacy-safe hashes,
IDs, aggregate statistics, and checksums.
