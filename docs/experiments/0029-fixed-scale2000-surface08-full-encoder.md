# Experiment 0029: Fixed Scale-2000 Surface08 Full Encoder

Classification: `SURFACE08_NEW_BEST_DIRECTIONAL_CANDIDATE`

This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.

## Result

- Surface: `SURFACE_08_FULL_ENCODER` (`encoder.layers.0` through `encoder.layers.23`).
- Fusion bridge: `prompt_kernel`, proven as the post-concatenation 1152 -> 2048 -> 1024 projection with no learnable prompt table or embedding.
- Trainable parameters: 633,400,352 total; 14,940,160 decoder, 9,455,648 joint, 604,545,024 encoder layers, and 4,459,520 prompt kernel.
- Training stopped after round 9 (18,000 optimizer steps and 144,000 exposures): `three_rounds_without_new_raw_best`.
- ARTUR controller-dev selected round 6 at 40.965 WER / 14.321 CER / 0 empty hypotheses.
- Selected checkpoint SHA256: `608b42182954aed0fdd89ab06e4402f95d522d84e123c05c1d78686cdd606d6f`.
- Training hardware: NVIDIA GeForce RTX 3090, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM 18882.195/19598.000 MiB.
- Surface08 is a new best directional candidate; it still remains a one-time synthetic-only boundary diagnostic.

## Fusion Bridge Discovery

| Candidate module | Included | Reason | Trainable params | Safety note |
|---|---:|---|---:|---|
| `prompt_kernel` | true | post-concat acoustic/prompt projection selected as the sole fusion bridge | 4459520 | Selected post-concat bridge; prompt identity is a non-parameter one-hot mapping. |
| `prompt_kernel.0` | false | nested bridge component or non-selected candidate | 0 | Not independently selected; nested component or non-bridge candidate. |
| `prompt_kernel.1` | false | nested bridge component or non-selected candidate | 0 | Not independently selected; nested component or non-bridge candidate. |
| `prompt_kernel.2` | false | nested bridge component or non-selected candidate | 0 | Not independently selected; nested component or non-bridge candidate. |

## ARTUR Controller-Dev Curve

| Round | Step | Exposures | Train loss | Synthetic anchor | Synthetic scale | ARTUR-dev WER | CER | Empty | Delete | Insert | Substitute | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | NOT_APPLICABLE | 48.291606 | 47.956826 | 66.467 | 27.409 | 13 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 1 | 2000 | 16000 | 13.824143 | 4.65875 | 5.096832 | 49.509 | 18.391 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 2 | 4000 | 32000 | 7.046622 | 6.917809 | 6.931447 | 42.76 | 14.234 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 3 | 6000 | 48000 | 6.433516 | 8.123126 | 7.400624 | 41.777 | 16.098 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 4 | 8000 | 64000 | 6.407375 | 9.136706 | 8.674229 | 43.742 | 16.04 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 5 | 10000 | 80000 | 4.549959 | 6.582034 | 7.574535 | 44.212 | 15.75 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 6 | 12000 | 96000 | 5.277265 | 7.940504 | 7.506954 | 40.965 | 14.321 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 7 | 14000 | 112000 | 5.303324 | 10.311305 | 8.540923 | 43.956 | 15.649 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 8 | 16000 | 128000 | 4.564557 | 8.878751 | 7.28603 | 42.418 | 15.366 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 9 | 18000 | 144000 | 4.696569 | 9.409743 | 7.595072 | 42.076 | 14.466 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |

`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.

## Post-Selection Directional Metrics

| Split | Base | PR #36 round20 | Surface06 | Surface07 | Surface08 selected |
|---|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 34.317 / 13.765 / 0 | 34.161 / 9.952 / 0 | 23.137 / 7.429 / 0 | 20.497 / 6.112 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 14.752 / 4.682 / 0 | 9.783 / 2.761 / 0 | 7.842 / 2.145 / 0 | 5.202 / 1.850 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 46.195 / 15.604 / 0 | 44.506 / 13.528 / 0 | 42.084 / 12.985 / 0 | 41.878 / 13.186 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 56.793 / 20.177 / 0 | 50.590 / 15.803 / 0 | 47.357 / 14.805 / 0 | 41.765 / 13.553 / 0 |

Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.

## Best-Known Real-Gate Envelope

| Metric | Best prior value | Prior source | Surface08 value | Within tolerance |
|---|---:|---|---:|---|
| fleurs_v2 WER | 42.084 | Surface07 | 41.878 | true |
| fleurs_v2 CER | 12.985 | Surface07 | 13.186 | true |
| artur_j WER | 47.357 | Surface07 | 41.765 | true |
| artur_j CER | 14.805 | Surface07 | 13.553 | true |

## Parameter Integrity

| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |
|---|---|---|---|---|---|
| decoder | trainable | true | `dc97e339f59a7b0a84e59329d677f97ad306276c969d01087f7c95783dfdae18` | `573d34c0da3b96bba50ca53f22402f7114da2fc30da3834d4e205d6ce65110d6` | bitwise aggregate fingerprint |
| joint | trainable | true | `22aeb19b5252507e4fc76ef2c27d7e273a9cc237104e27e9f473b3051973339e` | `0d9074e455b429f55c1634c7ace4f8a13ee687416818223663a46d366059c471` | bitwise aggregate fingerprint |
| encoder_all_layers | trainable | true | `3ddd30c6bf5275b34fa5b09ab56009e3ec7415b7a75a93401c6c21f11dd6c2c3` | `c15ea88df5489f2603edaf332983b42732b55817c8e05c0b71e31bf7464bb993` | bitwise aggregate fingerprint |
| fusion_bridge_candidate | trainable | true | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | `51c4d4e188037ebfb7c2790992d9d775aa0b344e88f2d688b8a74af89004ac85` | bitwise aggregate fingerprint |
| frontend_subsampling_preprocessor | frozen | false | `437d4c7d9773ffd178e6e4832b87225f8e1c23f180b00aa1bd3fbee208f7dbad` | `437d4c7d9773ffd178e6e4832b87225f8e1c23f180b00aa1bd3fbee208f7dbad` | bitwise aggregate fingerprint |
| tokenizer | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| other_prompt_or_fusion_modules | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| adapters | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| temporary_lm_heads | frozen | false | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | bitwise aggregate fingerprint |
| language_id_mapping | frozen | false | `184fa057c488254765e999bef52efe35de0a9d895a39f0314771a92d03e6d080` | `184fa057c488254765e999bef52efe35de0a9d895a39f0314771a92d03e6d080` | protected non-tensor configuration fingerprint |
| prompt_labels_tables_embeddings | frozen | false | `eb2ef47e7546b37bc254099f6ecf2a0998746102cd39005a7ba998b8fc7f1ce6` | `eb2ef47e7546b37bc254099f6ecf2a0998746102cd39005a7ba998b8fc7f1ce6` | protected non-tensor configuration fingerprint |
| target_lang_machinery | frozen | false | `dd2df2aaeebc2e242c686c114c6314be137fdb714c4321115eb3fc3ae6e7d995` | `dd2df2aaeebc2e242c686c114c6314be137fdb714c4321115eb3fc3ae6e7d995` | protected non-tensor configuration fingerprint |
| tokenizer | frozen | false | `62185153a1b898baabde31320b235e9a9e53dddcd5c527d8d37465542ef80ec9` | `62185153a1b898baabde31320b235e9a9e53dddcd5c527d8d37465542ef80ec9` | protected non-tensor configuration fingerprint |

## Boundaries

- `accepted_parent` remains `none`.
- The result is diagnostic, noncanonical, and promotion-ineligible.
- No real speech was used for training; ARTUR controller-dev was aggregate run-control only.
- No checkpoint, audio, prediction, raw reference/hypothesis, or local manifest is committed.
- No `TRAINING_ELIGIBLE` status or model publication is issued.
- This Work Order 0043 Surface08 run is a one-time boundary exception; Surface09 and full-model training remain prohibited.
