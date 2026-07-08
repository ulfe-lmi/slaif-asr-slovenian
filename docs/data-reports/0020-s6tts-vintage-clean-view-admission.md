# S6TTS Vintage Clean-View Admission

Classification: `S6TTS_REJECTED_SYNTHESIS_QUALITY`

This report records a failed internal diagnostic S6TTS clean synthetic voice-view admission attempt for the fixed scale-2000 text corpus. It does not authorize model training, public audio release, checkpoint acceptance, or `TRAINING_ELIGIBLE` status.

## Identity

- Corpus: `sl-corpus-v4-gams-16000-training-v1`
- View: `sl-corpus-v4-s6tts-clean-view-v1`
- Fixed text SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- TTS engine: `s6tts` at `b0c7d3fe7e7b0a06e05bf50e61f774a9daa5e8b6`
- Voice label: `s6tts-sl-si-s6-vintage`

## Counts

- Semantic rows: 16000
- Expected clean files: 16000
- Actual clean files: 14528
- Duplicate paths: 0
- Duplicate audio hashes: 3
- Synthesis failures: 1472

## Audio

- Format: mono signed 16-bit PCM WAV at 16000 Hz
- Total duration seconds: 60774.060757
- Duration distribution: `{"max": 23.555375, "mean": 4.183237, "min": 1.161812, "p50": 3.84375, "p95": 7.444375}`
- Peak distribution: `{"max": 0.733093, "mean": 0.700989, "min": 0.197784, "p50": 0.719879, "p95": 0.724609}`

## Representative Failure Reproduction

- S6TTS revision: `b0c7d3fe7e7b0a06e05bf50e61f774a9daa5e8b6`
- Row index: 1
- Safe key: `s6tts-scale2000-00001`
- Text SHA256: `ef112110a8a2dbee814ae2f9452928f3934cbac326d6f62b6f5ac8eb4381d3c5`
- Text length: 21 characters / 21 UTF-8 bytes
- Command template: `s6cli --ini data/sl-si-s6/sint.ini --text <RAW_TEXT> -o row_00001.fail.wav`
- Observed result: exit status 1, no stdout/stderr, no WAV
- Standalone reproduction bundle: retained only in ignored local run storage
- Raw text committed: no

## Safety

- Generated audio committed: no
- Local manifests committed: no
- Raw text committed: no
- Public release authorized: no
- Accepted parent: none
- TRAINING_ELIGIBLE issued: no
