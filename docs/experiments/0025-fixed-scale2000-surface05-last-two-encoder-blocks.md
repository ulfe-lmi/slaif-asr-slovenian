# Experiment 0025: Fixed Scale-2000 Surface05 Last Two Encoder Blocks

Classification: `SURFACE05_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF`

This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.

## Result

- Surface: `SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS` (`encoder.layers.22, encoder.layers.23`).
- Trainable parameters: 74,774,560 total; 14,940,160 decoder, 9,455,648 joint, and 50,378,752 final-two encoder blocks.
- Training stopped after round 6 (12,000 optimizer steps and 96,000 exposures): `three_rounds_without_new_raw_best`.
- ARTUR controller-dev selected round 3 at 50.235 WER / 16.853 CER / 0 empty hypotheses.
- Selected checkpoint SHA256: `d82c19ca5769e8460cd88cd1dae008c63de9e6d77b36dcac1a6fe917dffcac81`.
- Training hardware: NVIDIA GeForce RTX 2080 Ti, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM 7129.691/8472.000 MiB.
- Surface05 matches the best-known envelope with an acceptable tradeoff; Surface06 remains subject to separate strategic review.

## ARTUR Controller-Dev Curve

| Round | Step | Exposures | Train loss | Synthetic anchor | Synthetic scale | ARTUR-dev WER | CER | Empty | Delete | Insert | Substitute | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | NOT_APPLICABLE | 48.29162 | 47.956822 | 66.467 | 27.409 | 13 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 1 | 2000 | 16000 | 15.452589 | 5.246239 | 6.448827 | 56.514 | 21.307 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 2 | 4000 | 32000 | 4.045623 | 8.251555 | 9.033674 | 52.841 | 18.95 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 3 | 6000 | 48000 | 3.446447 | 7.994952 | 9.156891 | 50.235 | 16.853 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 4 | 8000 | 64000 | 4.691922 | 7.775399 | 9.088835 | 51.516 | 17.216 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 5 | 10000 | 80000 | 3.121785 | 8.017862 | 9.205 | 52.285 | 17.542 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 6 | 12000 | 96000 | 2.491925 | 7.603163 | 9.475243 | 50.961 | 17.665 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |

`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.

## Post-Selection Directional Metrics

| Split | Base | Scale-2000 joint-adapter | PR #36 round20 | PR #39 round6 | PR #43 Surface04 | Surface05 selected |
|---|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 55.435 / 20.073 / 0 | 34.317 / 13.765 / 0 | 44.565 / 16.428 / 0 | 41.460 / 14.522 / 0 | 39.130 / 13.485 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 27.407 / 7.597 / 0 | 14.752 / 4.682 / 0 | 18.711 / 6.196 / 0 | 16.071 / 4.962 / 0 | 13.509 / 4.177 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 51.589 / 16.238 / 0 | 46.195 / 15.604 / 0 | 48.023 / 15.946 / 0 | 46.292 / 14.792 / 0 | 46.564 / 14.950 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 60.114 / 20.630 / 0 | 56.793 / 20.177 / 0 | 57.274 / 20.375 / 0 | 55.920 / 18.535 / 0 | 53.473 / 17.473 / 0 |

Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.

## Best-Known Real-Gate Envelope

| Metric | Best prior value | Prior source | Surface05 value | Within tolerance |
|---|---:|---|---:|---|
| fleurs_v2 WER | 46.195 | PR #36 | 46.564 | true |
| fleurs_v2 CER | 14.792 | Surface04 | 14.950 | true |
| artur_j WER | 55.920 | Surface04 | 53.473 | true |
| artur_j CER | 18.535 | Surface04 | 17.473 | true |

## Parameter Integrity

| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |
|---|---|---|---|---|---|
| decoder | trainable | true | `dc97e339f59a7b0a84e59329d677f97ad306276c969d01087f7c95783dfdae18` | `a58d6bd31bc59081076470133880a516a335239b9483e668ef9eea85af51f708` | bitwise aggregate fingerprint |
| joint | trainable | true | `22aeb19b5252507e4fc76ef2c27d7e273a9cc237104e27e9f473b3051973339e` | `6d2c9621d6fa553befca0241cac5c7990bb3f1fbcf665ba200b1d930f954db87` | bitwise aggregate fingerprint |
| encoder_final_two_blocks | trainable | true | `ec2a7b895cff5bd6295b1eecf2ce1d0af8aeaa0cf6c8084a9088c2910180ec5f` | `55c70c9f19ca74250e40644f98ce83bfdbfb932b8161c48926a797b354de94d7` | bitwise aggregate fingerprint |
| encoder_lower_frozen | frozen | false | `d698f26efb655bc2c2a2fd26ee03019fcc56f9435112ce26de44470e1bf4226d` | `d698f26efb655bc2c2a2fd26ee03019fcc56f9435112ce26de44470e1bf4226d` | bitwise aggregate fingerprint |
| frontend_subsampling_preprocessor | frozen | false | `437d4c7d9773ffd178e6e4832b87225f8e1c23f180b00aa1bd3fbee208f7dbad` | `437d4c7d9773ffd178e6e4832b87225f8e1c23f180b00aa1bd3fbee208f7dbad` | bitwise aggregate fingerprint |
| tokenizer | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| prompt_path | frozen | false | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | bitwise aggregate fingerprint |
| adapters | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| temporary_lm_heads | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |

## Boundaries

- `accepted_parent` remains `none`.
- The result is diagnostic, noncanonical, and promotion-ineligible.
- No real speech was used for training; ARTUR controller-dev was aggregate run-control only.
- No checkpoint, audio, prediction, raw reference/hypothesis, or local manifest is committed.
- No `TRAINING_ELIGIBLE` status or model publication is issued.
- Surface06 is not authorized by this experiment.
