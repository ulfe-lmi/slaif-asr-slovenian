# Experiment 0026: Fixed Scale-2000 Surface06 Last Four Encoder Blocks

Classification: `SURFACE06_NEW_BEST_DIRECTIONAL_CANDIDATE`

This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.

## Result

- Surface: `SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS` (`encoder.layers.20, encoder.layers.21, encoder.layers.22, encoder.layers.23`).
- Trainable parameters: 125,153,312 total; 14,940,160 decoder, 9,455,648 joint, and 100,757,504 final-four encoder blocks.
- Training stopped after round 8 (16,000 optimizer steps and 128,000 exposures): `three_rounds_without_new_raw_best`.
- ARTUR controller-dev selected round 5 at 46.818 WER / 16.309 CER / 0 empty hypotheses.
- Selected checkpoint SHA256: `e82bd833fcbb622c73b7acb8f295e18d32fd605fcd49915286fd84ebee19cdf1`.
- Training hardware: NVIDIA GeForce RTX 2080 Ti, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM 7812.766/8514.000 MiB.
- Surface06 is a new best directional candidate; Surface07 remains subject to separate strategic review and work-order approval.

## ARTUR Controller-Dev Curve

| Round | Step | Exposures | Train loss | Synthetic anchor | Synthetic scale | ARTUR-dev WER | CER | Empty | Delete | Insert | Substitute | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | NOT_APPLICABLE | 48.29162 | 47.956822 | 66.467 | 27.409 | 13 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 1 | 2000 | 16000 | 14.583634 | 4.543805 | 5.716893 | 56.301 | 20.284 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 2 | 4000 | 32000 | 3.731066 | 6.33274 | 7.934966 | 51.986 | 17.651 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 3 | 6000 | 48000 | 3.156642 | 7.070135 | 8.634986 | 48.74 | 16.584 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 4 | 8000 | 64000 | 4.108035 | 8.435026 | 8.933748 | 47.843 | 16.548 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 5 | 10000 | 80000 | 2.655207 | 6.852179 | 8.432213 | 46.818 | 16.309 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 6 | 12000 | 96000 | 2.166629 | 7.212985 | 8.182497 | 47.416 | 16.722 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 7 | 14000 | 112000 | 2.448938 | 8.205366 | 8.915628 | 47.202 | 16.751 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 8 | 16000 | 128000 | 2.119793 | 8.851494 | 9.033285 | 48.996 | 17.963 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |

`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.

## Post-Selection Directional Metrics

| Split | Base | Scale-2000 joint-adapter | PR #36 round20 | PR #39 round6 | PR #43 Surface04 | PR #44 Surface05 | Surface06 selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 55.435 / 20.073 / 0 | 34.317 / 13.765 / 0 | 44.565 / 16.428 / 0 | 41.460 / 14.522 / 0 | 39.130 / 13.485 / 0 | 34.161 / 9.952 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 27.407 / 7.597 / 0 | 14.752 / 4.682 / 0 | 18.711 / 6.196 / 0 | 16.071 / 4.962 / 0 | 13.509 / 4.177 / 0 | 9.783 / 2.761 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 51.589 / 16.238 / 0 | 46.195 / 15.604 / 0 | 48.023 / 15.946 / 0 | 46.292 / 14.792 / 0 | 46.564 / 14.950 / 0 | 44.506 / 13.528 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 60.114 / 20.630 / 0 | 56.793 / 20.177 / 0 | 57.274 / 20.375 / 0 | 55.920 / 18.535 / 0 | 53.473 / 17.473 / 0 | 50.590 / 15.803 / 0 |

Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.

## Best-Known Real-Gate Envelope

| Metric | Best prior value | Prior source | Surface06 value | Within tolerance |
|---|---:|---|---:|---|
| fleurs_v2 WER | 46.195 | PR #36 | 44.506 | true |
| fleurs_v2 CER | 14.792 | Surface04 | 13.528 | true |
| artur_j WER | 53.473 | Surface05 | 50.590 | true |
| artur_j CER | 17.473 | Surface05 | 15.803 | true |

## Parameter Integrity

| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |
|---|---|---|---|---|---|
| decoder | trainable | true | `dc97e339f59a7b0a84e59329d677f97ad306276c969d01087f7c95783dfdae18` | `c2fb04ede53120f145f1d9b4793d0b1057f9d9fbb9319ace13f2e12afbb1ee03` | bitwise aggregate fingerprint |
| joint | trainable | true | `22aeb19b5252507e4fc76ef2c27d7e273a9cc237104e27e9f473b3051973339e` | `4f934eac45fece5d575ca82c4118c93e11ec495c0f7903f8e0a382df0192d29d` | bitwise aggregate fingerprint |
| encoder_final_four_blocks | trainable | true | `fab38193b43b79d088040c1821fd4727eef12e3f9fcd0393beda1423e728d86f` | `35b6639b9c97dfe7dcae0cc50f596548fdcd0de978483689b4000e2c6faa314c` | bitwise aggregate fingerprint |
| encoder_lower_frozen | frozen | false | `4892fbc11a2e4267e84fc638c51fe65acfc8500511ab03aae72f9d84b755ae70` | `4892fbc11a2e4267e84fc638c51fe65acfc8500511ab03aae72f9d84b755ae70` | bitwise aggregate fingerprint |
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
- Surface07 is not authorized by this experiment.
