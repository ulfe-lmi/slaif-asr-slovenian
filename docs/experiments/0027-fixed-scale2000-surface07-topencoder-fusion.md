# Experiment 0027: Fixed Scale-2000 Surface07 Top Encoder Plus Fusion

Classification: `SURFACE07_NEW_BEST_DIRECTIONAL_CANDIDATE`

This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.

## Result

- Surface: `SURFACE_07_TOP_ENCODER_PLUS_PROMPT_ACOUSTIC_FUSION` (`encoder.layers.20, encoder.layers.21, encoder.layers.22, encoder.layers.23`).
- Fusion bridge: `prompt_kernel`, proven as the post-concatenation 1152 -> 2048 -> 1024 projection with no learnable prompt table or embedding.
- Trainable parameters: 129,612,832 total; 14,940,160 decoder, 9,455,648 joint, 100,757,504 final-four encoder blocks, and 4,459,520 fusion bridge.
- Training stopped after round 16 (32,000 optimizer steps and 256,000 exposures): `three_rounds_without_new_raw_best`.
- ARTUR controller-dev selected round 13 at 43.443 WER / 15.097 CER / 0 empty hypotheses.
- Selected checkpoint SHA256: `349d06dd517b6e99b71a74f15a04d6020afe56223ef946014d3bdca1440706b0`.
- Training hardware: NVIDIA GeForce RTX 2080 Ti, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM 7865.097/8514.000 MiB.
- Surface07 is a new best directional candidate; the next step is strategic review and canonical evaluation of named challengers, not full-encoder expansion.

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
| 0 | 0 | 0 | NOT_APPLICABLE | 48.29162 | 47.956822 | 66.467 | 27.409 | 13 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 1 | 2000 | 16000 | 14.379448 | 4.381477 | 5.586967 | 56.514 | 20.763 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 2 | 4000 | 32000 | 3.659821 | 6.779726 | 7.799963 | 48.612 | 15.823 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 3 | 6000 | 48000 | 3.118396 | 6.58783 | 8.095223 | 47.971 | 15.975 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 4 | 8000 | 64000 | 3.981157 | 7.69125 | 8.23539 | 49.124 | 16.265 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 5 | 10000 | 80000 | 2.626903 | 5.632031 | 7.808436 | 48.74 | 16.824 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 6 | 12000 | 96000 | 2.164183 | 6.745709 | 8.095491 | 47.202 | 16.809 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 7 | 14000 | 112000 | 2.473542 | 7.79435 | 9.098131 | 47.159 | 17.143 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 8 | 16000 | 128000 | 2.120294 | 7.813046 | 9.344894 | 47.971 | 17.041 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 9 | 18000 | 144000 | 2.637427 | 5.885981 | 7.935727 | 48.825 | 17.179 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 10 | 20000 | 160000 | 2.964163 | 3.528053 | 5.469396 | 45.75 | 16.265 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 11 | 22000 | 176000 | 2.705713 | 3.36435 | 4.535124 | 45.963 | 15.823 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 12 | 24000 | 192000 | 2.578898 | 3.080787 | 4.326448 | 44.938 | 15.562 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 13 | 26000 | 208000 | 3.810158 | 3.391797 | 4.3642 | 43.443 | 15.097 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | True |
| 14 | 28000 | 224000 | 2.519843 | 2.424959 | 3.803295 | 44.938 | 15.99 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 15 | 30000 | 240000 | 2.396915 | 2.020056 | 3.683216 | 46.049 | 15.997 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |
| 16 | 32000 | 256000 | 2.253964 | 2.238718 | 3.792823 | 44.853 | 16.49 | 0 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | False |

`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.

## Post-Selection Directional Metrics

| Split | Base | PR #36 round20 | Surface04 | Surface05 | Surface06 | Surface07 selected |
|---|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 34.317 / 13.765 / 0 | 41.460 / 14.522 / 0 | 39.130 / 13.485 / 0 | 34.161 / 9.952 / 0 | 23.137 / 7.429 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 14.752 / 4.682 / 0 | 16.071 / 4.962 / 0 | 13.509 / 4.177 / 0 | 9.783 / 2.761 / 0 | 7.842 / 2.145 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 46.195 / 15.604 / 0 | 46.292 / 14.792 / 0 | 46.564 / 14.950 / 0 | 44.506 / 13.528 / 0 | 42.084 / 12.985 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 56.793 / 20.177 / 0 | 55.920 / 18.535 / 0 | 53.473 / 17.473 / 0 | 50.590 / 15.803 / 0 | 47.357 / 14.805 / 0 |

Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.

## Best-Known Real-Gate Envelope

| Metric | Best prior value | Prior source | Surface07 value | Within tolerance |
|---|---:|---|---:|---|
| fleurs_v2 WER | 44.506 | Surface06 | 42.084 | true |
| fleurs_v2 CER | 13.528 | Surface06 | 12.985 | true |
| artur_j WER | 50.590 | Surface06 | 47.357 | true |
| artur_j CER | 15.803 | Surface06 | 14.805 | true |

## Parameter Integrity

| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |
|---|---|---|---|---|---|
| decoder | trainable | true | `dc97e339f59a7b0a84e59329d677f97ad306276c969d01087f7c95783dfdae18` | `2852e44a8549cc18f57d9e71cc1c512a3ff7ce58a98eb560e32504505f70093f` | bitwise aggregate fingerprint |
| joint | trainable | true | `22aeb19b5252507e4fc76ef2c27d7e273a9cc237104e27e9f473b3051973339e` | `b04bae7225b9f2cadf9e2535e162dd89f5ab9fb29627268166ae88c2189d0cec` | bitwise aggregate fingerprint |
| encoder_final_four_blocks | trainable | true | `fab38193b43b79d088040c1821fd4727eef12e3f9fcd0393beda1423e728d86f` | `96bb3d537fc0a11a20bacb7f42783e4f73ecd95a63069d4f645651ad2f271511` | bitwise aggregate fingerprint |
| encoder_lower_frozen | frozen | false | `4892fbc11a2e4267e84fc638c51fe65acfc8500511ab03aae72f9d84b755ae70` | `4892fbc11a2e4267e84fc638c51fe65acfc8500511ab03aae72f9d84b755ae70` | bitwise aggregate fingerprint |
| fusion_bridge_candidate | trainable | true | `5090603a4277d89b5785541a54e259a79cda1c51ed89d48257eeedbb3307d0c9` | `e8abf9d4e8ccc3e1966c1e1658b6ae2f730bb44af26332d3e87a8032e8362d7f` | bitwise aggregate fingerprint |
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
- Surface08 and full-encoder training remain prohibited.
