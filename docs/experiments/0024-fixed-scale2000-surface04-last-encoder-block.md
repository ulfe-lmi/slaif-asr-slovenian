# Experiment 0024: Fixed Scale-2000 Surface04 Last Encoder Block

Classification: `SURFACE04_MATCHES_PR36_WITH_ACCEPTABLE_TRADEOFF`

This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.

## Result

- Surface: `SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK` (`encoder.layers.23`).
- Trainable parameters: 49,585,184 total; 14,940,160 decoder, 9,455,648 joint, and 25,189,376 final encoder block.
- Training stopped after round 6 (12,000 optimizer steps and 96,000 exposures): `three_rounds_without_new_raw_best`.
- ARTUR controller-dev selected round 3 at 53.182 WER / 19.037 CER / 0 empty hypotheses.
- Selected checkpoint SHA256: `56399d4d34c782430203be44071bf2c6fd432e2d228726e3587d83eed4ec412f`.
- Training hardware: NVIDIA GeForce RTX 2080 Ti, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM 6788.154/8450.000 MiB.
- Surface04 did not beat PR #36 cleanly, but it is a credible real-gate tradeoff candidate under the one-sided non-regression tolerance. Surface05 is justified as the next controlled diagnostic, subject to a separate work order and review.

## ARTUR Controller-Dev Curve

| Round | Step | Exposures | Train loss | Synthetic anchor | Synthetic scale | ARTUR-dev WER | CER | Empty | Delete | Insert | Substitute | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | None | 48.29162 | 47.956822 | 66.467 | 27.409 | 13 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 1 | 2000 | 16000 | 16.423968 | 6.340908 | 7.015988 | 59.205 | 21.837 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 2 | 4000 | 32000 | 4.550514 | 9.071614 | 9.416921 | 54.72 | 18.819 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 3 | 6000 | 48000 | 3.90842 | 9.389366 | 10.22622 | 53.182 | 19.037 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 4 | 8000 | 64000 | 5.446432 | 9.914442 | 10.301498 | 53.353 | 17.76 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 5 | 10000 | 80000 | 3.718203 | 9.395036 | 9.923884 | 53.439 | 19.537 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 6 | 12000 | 96000 | 2.940405 | 9.090633 | 9.64436 | 53.353 | 18.246 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |

`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.

## Post-Selection Directional Metrics

| Split | Base | Scale-2000 joint-adapter | PR #36 round20 | PR #39 round6 | PR #42 S6TTS hardvoice | SURFACE_04 selected |
|---|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 55.435 / 20.073 / 0 | 34.317 / 13.765 / 0 | 44.565 / 16.428 / 0 | 38.043 / 16.008 / 0 | 41.460 / 14.522 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 27.407 / 7.597 / 0 | 14.752 / 4.682 / 0 | 18.711 / 6.196 / 0 | 17.081 / 6.939 / 0 | 16.071 / 4.962 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 51.589 / 16.238 / 0 | 46.195 / 15.604 / 0 | 48.023 / 15.946 / 0 | 48.162 / 16.725 / 0 | 46.292 / 14.792 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 60.114 / 20.630 / 0 | 56.793 / 20.177 / 0 | 57.274 / 20.375 / 0 | 54.871 / 20.318 / 0 | 55.920 / 18.535 / 0 |

Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.

## Parameter Integrity

| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |
|---|---|---|---|---|---|
| decoder | trainable | true | `dc97e339f59a7b0a84e59329d677f97ad306276c969d01087f7c95783dfdae18` | `d48b5c0a7fcf18efa4bf02d1a45ef0930dd7aaeb93f405c6c56aa36f193815c1` | bitwise aggregate fingerprint |
| joint | trainable | true | `22aeb19b5252507e4fc76ef2c27d7e273a9cc237104e27e9f473b3051973339e` | `13f10de99338ff4f279b6d3b546963eef2e16ccbe6a00c0ab31a32bc1c4e1fd0` | bitwise aggregate fingerprint |
| final_encoder_block | trainable | true | `f025471e7a42e2fcab698eae237d4dbf5bb9fdbcff40202512d970000be41fc2` | `8b64a76bb710c1151431bfc785ef9088c78aae6732ad377d2560e8e32b6d60d7` | bitwise aggregate fingerprint |
| lower_encoder_and_frontend | frozen | false | `0bc5aab4f5a0ef1c3970261dd823ae5f5a589e254cb04b5ef396fdeb7be1bb0b` | `0bc5aab4f5a0ef1c3970261dd823ae5f5a589e254cb04b5ef396fdeb7be1bb0b` | bitwise aggregate fingerprint |
| prompt_path | frozen | false | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | bitwise aggregate fingerprint |

## Boundaries

- `accepted_parent` remains `none`.
- The result is diagnostic, noncanonical, and promotion-ineligible.
- No real speech was used for training; ARTUR controller-dev was aggregate run-control only.
- No checkpoint, audio, prediction, raw reference/hypothesis, or local manifest is committed.
- No `TRAINING_ELIGIBLE` status or model publication is issued.
