# Experiment 0004: Slovenian Curriculum Round 1

Status: **completed; challenger rejected**

## Purpose

This experiment tests whether a broader project-generated Slovenian synthetic
curriculum can make the already proven 2048-parameter `sl-SI` prompt-column
adaptation generalize beyond the original eight Piper smoke sentences.

The experiment does not use GaMS, external LLM APIs, external model servers, or
corpus-generation services. The generated text, audio, manifests, hypotheses,
delta, checkpoint, and raw reports remain ignored local artifacts.

## Configuration

- Generation config:
  [`configs/generation/slovenian_curriculum_round1.json`](../../configs/generation/slovenian_curriculum_round1.json)
- Generation specification revision: `sl-curriculum-round1-v1`
- Provenance system: `project-generated`
- Method: `direct-language-generation`
- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Base checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Piper revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Voice revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Hardware: one RTX 2080 Ti selected with `CUDA_VISIBLE_DEVICES=0`
- Precision: FP32
- Effective trainable parameters: 2048

## Corpus Construction

| Corpus | Requested | Valid | SHA256 | Exact duplicates | Near duplicates | Protected overlaps |
|---|---:|---:|---|---:|---:|---:|
| synthetic holdout | 96 | 96 | `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9` | 0 | 0 | 0 |
| candidate pool | 320 | 320 | `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17` | 0 | 0 | 0 |

The holdout was generated and locked before candidate-pool scoring. It was not
used for training, selection, or steering.

## Category Coverage

| Category | Holdout | Candidate pool | Selected training |
|---|---:|---:|---:|
| ordinary/conversational | 16 | 50 | 19 |
| questions/requests | 10 | 35 | 15 |
| commands | 10 | 35 | 19 |
| č/š/ž coverage | 6 | 40 | 18 |
| morphology/inflection | 12 | 40 | 19 |
| dual | 8 | 25 | 15 |
| function words/clitics | 6 | 25 | 15 |
| names/places/institutions | 12 | 25 | 15 |
| dates/numbers/quantities | 8 | 25 | 16 |
| technical/code-switching | 8 | 20 | 9 |

## Synthesis and Selection

| Stage | Result |
|---|---:|
| Holdout synthesized | 96 |
| Candidate pool synthesized | 320 |
| Synthesis failures | 0 |
| Synthesis wall time | 737.488 s |
| Candidate pool scored | 320 |
| Base empty candidate hypotheses | 35 |
| Hard examples selected | 120 |
| General controls selected | 40 |
| Training manifest SHA256 | `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4` |

## Training

| Field | Value |
|---|---:|
| Parent checkpoint | untouched accepted base |
| Optimizer | AdamW |
| Learning rate | 0.01 |
| Weight decay | 0 |
| Batch size | 1 |
| Precision | FP32 |
| Steps | 1800 |
| Early stop | moving-average improvement 0.000943 |
| Initial loss | 98.54013061523438 |
| Final loss | 54.54838562011719 |
| Wall time | 157.697 s |
| Peak VRAM | 5053.0 MiB |

## Parameter Integrity

| Check | Result |
|---|---|
| Effective trainable parameters | 2048 |
| Changed tensors | `prompt_kernel.0.weight` |
| Selected column only | passed |
| Unexpected tensors | none |
| Unexpected changed elements | 0 |
| Encoder/decoder/joint/tokenizer unchanged | passed |
| Restored merged checkpoint | passed |

## Metrics

Normalized metrics use `sl-asr-normalization-v1`. Corpus WER/CER are computed
from summed edits, not averaged utterance percentages.

| Split/gate | Base WER | Challenger WER | Base CER | Challenger CER | Base empty | Challenger empty |
|---|---:|---:|---:|---:|---:|---:|
| selected synthetic training | 89.070 | 51.632 | 62.622 | 22.805 | 35 | 0 |
| fixed synthetic holdout | 77.563 | 76.983 | 39.092 | 37.093 | 20 | 6 |
| FLEURS full test | 52.734 | 70.885 | 16.423 | 33.758 | 0 | 3 |
| ARTUR-J gate | 67.453 | 80.996 | 29.016 | 44.784 | 12 | 24 |

Mean and median utterance metrics are recorded in the ignored local summary and
are intentionally reported separately from corpus WER/CER.

## Decision

`ROUND1_REJECTED`

Reasons:

- synthetic holdout improvement was below the required 15% relative WER/CER;
- FLEURS normalized corpus WER and CER regressed beyond thresholds;
- ARTUR-J normalized corpus WER and CER regressed beyond thresholds;
- empty hypotheses increased on both real gates.

The challenger is not an accepted parent.

## Privacy-Safe Failure Summary

- Synthetic training memorization improved strongly.
- Fixed synthetic holdout did not improve enough to satisfy the promotion gate.
- Real-gate aggregate behavior regressed on both FLEURS and ARTUR-J.
- Raw real references and hypotheses remain local ignored artifacts.
- Synthetic holdout raw errors were not used for steering.

## Limitations

- The generated candidate pool is local and unpublished.
- The corpus is a first bounded project-generated round, not a production
  curriculum.
- The prompt-column surface can memorize selected synthetic examples but this
  round did not generalize to fixed synthetic or real gates.
- No model quality, release, or benchmark claim is made.

## Next Recommendation

Do not generate Round 2 from this rejected challenger. Use the aggregate failure
evidence to design the next controlled work order, likely comparing whether the
same data requires a broader prompt-kernel adaptation surface while keeping real
gates protected.
