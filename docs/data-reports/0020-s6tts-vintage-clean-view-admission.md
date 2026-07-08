# S6TTS Vintage Clean-View Admission

Classification: `S6TTS_REJECTED_SYNTHESIS_QUALITY`

This report records an internal diagnostic S6TTS clean synthetic voice-view admission attempt for the fixed scale-2000 text corpus. The fixed S6TTS revision generated all expected WAV files, but the view is not admitted because duplicate audio hashes remain. It does not authorize model training, public audio release, checkpoint acceptance, or `TRAINING_ELIGIBLE` status.

## Identity

- Corpus: `sl-corpus-v4-gams-16000-training-v1`
- View: `sl-corpus-v4-s6tts-clean-view-v1`
- Fixed text SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- TTS engine: `s6tts` at `6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c`
- Voice label: `s6tts-sl-si-s6-vintage`
- Revision note: the work order's original S6TTS pin was updated after the
  human lead authorized rerunning with the upstream missing-diphone bug fix.

## Counts

- Semantic rows: 16000
- Expected clean files: 16000
- Actual clean files: 16000
- Duplicate paths: 0
- Duplicate audio hashes: 3
- Synthesis failures: 0

## Duplicate Audio Hashes

- Duplicate groups: 3
- Extra duplicate files: 3

| Audio SHA256 | Rows | Text SHA256 values | Duration seconds | Frames |
|---|---:|---|---:|---:|
| `0ae7560cbdbd3ecb3750d57010277ec855a95b595147ec728e59c2e73123cd1b` | 6084, 6192 | `08b47f889caf3aca73de3638e16ae3e3ff6cd73bb8733759b954c26ca403dbf6, 63d9ea984897262248c42e46f19d6d5f063f4f9ec14278ef072d6815c8543678` | 2.94525 | 47124 |
| `6c3cf12f9fa7dfe7491a11cd4064778d1b9f7cf9bf8ed2234380208d00487f83` | 6182, 6210 | `722a9adb205e3be477f890149c5f924bb8dd0526f67d9856d7d5ec12f8e48617, ea2e61052017d5edd45ef960b929f70c7e4a920311cab1156c33ef21d127b4e7` | 3.314 | 53024 |
| `df35ad734e2607cc80c3b0d7c5eb1f5435fbf4441417c6cd06a964ee7c4b6d26` | 6010, 6014 | `09b234521a70c4c03fafccafe611b62ee467911b22918f11cf7c14a3f833ae02, fefd988717d8c24ce07f769fe21f5382b7a4cad1eaf68883fa3b0157fbe23c6c` | 2.464375 | 39430 |

Raw text for these duplicate cases is retained only in ignored local debugging files and is not committed.

## Audio

- Format: mono signed 16-bit PCM WAV at 16000 Hz
- Total duration seconds: 68193.891926
- Duration distribution: `{"max": 23.555375, "mean": 4.262118, "min": 1.161812, "p50": 3.896438, "p95": 7.662312}`
- Peak distribution: `{"max": 0.733093, "mean": 0.701291, "min": 0.197784, "p50": 0.719879, "p95": 0.724609}`

## Safety

- Generated audio committed: no
- Local manifests committed: no
- Raw text committed: no
- Public release authorized: no
- Accepted parent: none
- TRAINING_ELIGIBLE issued: no
