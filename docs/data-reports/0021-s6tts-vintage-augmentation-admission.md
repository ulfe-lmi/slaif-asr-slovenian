# S6TTS Vintage Augmentation Admission

Classification: `S6TTS_AUGMENTED_VIEW_AUDIO_ACCEPTED`

This report admits one internal diagnostic S6TTS transcript-preserving augmented synthetic bank. It does not authorize model training, public audio release, checkpoint acceptance, or `TRAINING_ELIGIBLE` status.

## Identity

- Corpus: `sl-corpus-v4-gams-16000-training-v1`
- Source clean view: `sl-corpus-v4-s6tts-clean-view-v1`
- Augmented view: `sl-corpus-v4-s6tts-augmented-view-v1`
- Fixed text SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- TTS engine revision: `6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c`
- Augmentation config SHA256: `b686f549c6873b4a919acfe47712a2ff4dcbb0baa898a67bb8604c15eb42faee`

## Counts

- Semantic rows: 16000
- Source clean files: 16000
- Profiles per row: 11
- Expected augmented files: 176000
- Actual augmented files: 176000
- Augmentation failures: 0
- Duplicate paths: 0

## Duplicate Audio Hashes

- Duplicate audio hashes: 19
- Explained duplicate audio hashes: 19
- Unexplained duplicate audio hashes: 0

| Audio SHA256 | Explanation | Profile IDs | Rows | Text SHA256 values |
|---|---|---|---:|---|
| `0f675e5a4bb9d62e036a743df8cd393060b05e58a7590069e4431e7362b9f6ae` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 9325, 9325 | `7a24af8c23e01381110d129c0dd8cbae67c9d1b3bf562329a6db13f624e5a7cc, 7a24af8c23e01381110d129c0dd8cbae67c9d1b3bf562329a6db13f624e5a7cc` |
| `131fcb2dcfc505a1cfe70495a99e7db9e3452f9d5243aef4312a5d5fd8d25b60` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 3271, 3271 | `175ec93274aed83a0262f2ee79681d8550394797ea2c9aba02cd60ad36e2fc52, 175ec93274aed83a0262f2ee79681d8550394797ea2c9aba02cd60ad36e2fc52` |
| `16ab78a5d8f58ba7caa48f001397ea8a4b70af7fcc4f0d4c2cf4a8d33a272d21` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 10896, 10896 | `8372ceeaa6d30135f4f1ddcb5645e214bd08be2b312d4bb1029aeaee316d68c3, 8372ceeaa6d30135f4f1ddcb5645e214bd08be2b312d4bb1029aeaee316d68c3` |
| `1a7d04c3bee94e89aa9555514000a0f5f12053b19eb30bf169f8bc27919e4091` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 4853, 4853 | `4ffecaea4c85383b78d6e62345322a5341f757f3b6f194f465530964ad44c1c5, 4ffecaea4c85383b78d6e62345322a5341f757f3b6f194f465530964ad44c1c5` |
| `1d278043c867d298f68dc35af18492194091bf54d1c42879a35df51d10b15167` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 15253, 15253 | `448338ff77f5653019b4cc841c53e322aee39524edc8e956669be2b5d745b3da, 448338ff77f5653019b4cc841c53e322aee39524edc8e956669be2b5d745b3da` |
| `30443d2ab8d6699cab99e73d6a39fb5539ddb8c75d44e8456f59afdc4caf9241` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 8327, 8327 | `ca8c3a3c88579573db3ae0d768de32b57f5dcf26e7c3189af3465e2c5080d337, ca8c3a3c88579573db3ae0d768de32b57f5dcf26e7c3189af3465e2c5080d337` |
| `33bf2de241d59e691764cf0c0e21227b582cdced81450de60372069a71cd4d52` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 7207, 7207 | `4f4ee188d3d0bca7634f46345f526f7c5b01d1eb4277f8a306a218e6c2952b71, 4f4ee188d3d0bca7634f46345f526f7c5b01d1eb4277f8a306a218e6c2952b71` |
| `34bcc982b8e591b2bb7a8580fa91e29860740432688ae06e14bd1885e30ae7bd` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 11502, 11502 | `f4ea5ca014d7492d72838119b8703175348974b77ebc2d4b4d1959847debc5a9, f4ea5ca014d7492d72838119b8703175348974b77ebc2d4b4d1959847debc5a9` |
| `46f46cdae3f88c1b6d7e916e94d7dc746b3fd7fc09f9a0f6a324a7938400eae2` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 2394, 2394 | `73eb49cf51d1c1431f707a070153988308b40bf028400da916f72f35f0fe6346, 73eb49cf51d1c1431f707a070153988308b40bf028400da916f72f35f0fe6346` |
| `5465d6205b203eb1957820758877d845c05f0264e9390e72345cac08cb8094db` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 4849, 4849 | `536d45173268ba6cd7c1a8e28b1981b2dfeab446e2c6b8639175ba277cf4f200, 536d45173268ba6cd7c1a8e28b1981b2dfeab446e2c6b8639175ba277cf4f200` |
| `69e405e7886e9fd091ab75d3b7854aec853c1664f28af7c6f8e9ccfcd40829ba` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 14990, 14990 | `fa992b4f4ee5694ba9b5454ede641fb2232782e15045621b656d9a7eced96611, fa992b4f4ee5694ba9b5454ede641fb2232782e15045621b656d9a7eced96611` |
| `8a6ef6efe5180127d12ca047bb849929834b865a4fa9bbc0738a9f56f78050aa` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 12236, 12236 | `8738fdd847d4cbb9b11dbf2bd1a2351c411bb3b84e995445aa7c4f20e3607dd2, 8738fdd847d4cbb9b11dbf2bd1a2351c411bb3b84e995445aa7c4f20e3607dd2` |
| `a64bba83a8ae30e9b847ae043b859fd18dd2248f44bc97a90f6ec01cc780910d` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 11947, 11947 | `d19ada6eb7d603abefa11877266fcb971e1095ed3c2a1488fac89521ee536b3c, d19ada6eb7d603abefa11877266fcb971e1095ed3c2a1488fac89521ee536b3c` |
| `bb03e69e2c0962728e287988a44980ad44ae24956699883d24ab215062da65f2` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 15642, 15642 | `a3d2fb91299e985db31c95f7aa103a4442f9ea87618a1bfd9808d4b1a752c7e9, a3d2fb91299e985db31c95f7aa103a4442f9ea87618a1bfd9808d4b1a752c7e9` |
| `d3d92cf9e2992a8ad45cf6f44e269f377f8c0cd5e706b9e9c229933a63b82ffa` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 7670, 7670 | `6a5245fc40ea8e77eb1b11b452899bfa181ad9e6c74c0e39a5bcbbf8a85d8d41, 6a5245fc40ea8e77eb1b11b452899bfa181ad9e6c74c0e39a5bcbbf8a85d8d41` |
| `e0b1253c26ae39151f9f469a7581d4f3167f0b57b3edc0c7f69082fdb6faa493` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 10478, 10478 | `3b32bb6d80d19d83c0b0bd616d96b55eea663dc4bae755dda87615a7164f7411, 3b32bb6d80d19d83c0b0bd616d96b55eea663dc4bae755dda87615a7164f7411` |
| `e54f038a6322a2052d768a9d55e178f27bee68e20450d0b73e371a6fef28dc14` | `deterministic_augmentation_profile_equivalence` | `coupled_speed_pitch_resampling, tempo_preserving_pitch` | 4604, 4604 | `6d0f5d43dbc7be493b8b30c20857bd79120ad123d80e1a7a2570bb489aeecf9b, 6d0f5d43dbc7be493b8b30c20857bd79120ad123d80e1a7a2570bb489aeecf9b` |
| `f48ee6ec1f66287c4ebe0f5ca3e0d20b6187be42915e4440cb49a81e579c785a` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 1301, 1301 | `4413efc2febeb1b366591c711935cec0a549b410ad966544bffe2b8d76206145, 4413efc2febeb1b366591c711935cec0a549b410ad966544bffe2b8d76206145` |
| `fee46318ad2bf91a4d75c8c14b280e209e07a9175bc085b7636bfb0ac0d3db2f` | `deterministic_augmentation_profile_equivalence` | `microphone_channel_filtering, mild_pitch_formant_vtlp_proxy` | 13100, 13100 | `ed24482fb6eb3a7ed25bb731863cf9e1bdf29911b2592a14b46cee38eb800a67, ed24482fb6eb3a7ed25bb731863cf9e1bdf29911b2592a14b46cee38eb800a67` |

## Audio

- Format: mono signed 16-bit PCM WAV at 16000 Hz
- Total duration seconds: 756423.611653
- Duration distribution: `{"max": 25.439812, "mean": 4.297861, "min": 1.060188, "p50": 3.930438, "p95": 7.718063}`
- Peak distribution: `{"max": 0.97998, "mean": 0.670026, "min": 0.158112, "p50": 0.717041, "p95": 0.848145}`

## Safety

- Generated audio committed: no
- Local manifests committed: no
- Raw text committed: no
- Local absolute paths committed: no
- Model training run: no
- Accepted parent: none
- TRAINING_ELIGIBLE issued: no

## Limitations

- S6TTS-generated and S6TTS-augmented audio remain internal diagnostic synthetic material only.
- This PR admits an audio bank but does not define a training sampling schedule.
