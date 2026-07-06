# Scale-8000 Dual-GPU Generation

Status: `TEXT_ACCEPTED`

This report is privacy-safe scale-8000 generation evidence. It contains no raw generated text, audio paths, hypotheses, model artifacts, or monitor CSV data.

## Canonical Pass

- `.venv/bin/python -m unittest discover -s tests`: `passed`, Ran 298 tests in 99.204s; OK
- `.venv/bin/python -m py_compile $(git ls-files '*.py')`: `passed`
- `.venv/bin/python scripts/check_repository.py`: `passed`, Repository validation passed for 318 tracked files.
- `.venv/bin/python -m pip check`: `passed`, No broken requirements found.
- `.venv-gams/bin/python -m pip check`: `passed`, No broken requirements found.
- `.venv-piper/bin/python -m pip check`: `passed`, No broken requirements found.
- `.venv-supertonic/bin/python -m pip check`: `passed`, No broken requirements found.
- `bash -n scripts/*.sh`: `passed`
- `git diff --check`: `passed`
- Note: The first wrapper invocation of py_compile embedded newline-separated file names in a shell string and returned exit 126; the required py_compile command was rerun directly and passed.
- Note: Scale-8000 text generation proceeded after the run root was moved to external data storage with sufficient free space.

## Corpus

- Corpus ID: `sl-corpus-v5-scale8000-training-v1`
- Parent scale-2000 corpus: `sl-corpus-v4-gams-16000-training-v1`
- Parent scale-2000 SHA256: `dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14`
- Semantic rows planned: `64000`
- Clean files/views planned: `576000`
- Augmented files/views planned: `704000`
- Total views/exposures planned: `1280000`

## Inclusion

- Policy: `prefix`
- Evidence: The 16,000 inherited scale-2000 rows remain byte-for-byte unchanged and appear before all newly generated scale-8000 rows in the combined local corpus.

## Dual-GPU Plan

- `gpu0`: physical GPU `0`, `CUDA_VISIBLE_DEVICES=0`, tasks `600`, requested rows `36000`
- `gpu1`: physical GPU `1`, `CUDA_VISIBLE_DEVICES=1`, tasks `600`, requested rows `36000`

## Storage Preflight

- Available bytes: `205947699200`
- Projected new bytes: `247023026364`
- Required free bytes with safety margin: `308778782955`
- Sufficient: `False`

## Clean TTS Evidence

- Clean-audio status: `CLEAN_AUDIO_GENERATED`
- Planned clean files/views: `576000`
- Generated clean files/views: `576000`
- Piper rows: `64000`
- Piper manifest SHA256: `290c733207ee08a58d6a707b241b65ed25121d19da11472caed99f562c0fde9f`
- Piper duration seconds: `201544.979654`
- Supertonic rows: `512000`
- Supertonic manifest SHA256: `c6b3152ee2d8925dfbeb4c7acd2c6f6656f8aa26a32f65570f425a3c5004d5a4`
- Supertonic native manifest SHA256: `b61ac8d7ce223f2c9eca0a09b719627016b2de68b4fd759fdf1cb21bffdbea3b`
- Supertonic duration seconds: `2083074.411514`
- Augmented-audio status: `NOT_IMPLEMENTED_IN_THIS_RUNNER`
- Generated augmented files/views: `0`

## Decision

Scale-8000 text is accepted and the nine clean synthetic voice realizations have been generated. The current runner does not generate the eleven augmentation views, so the full 1,280,000-view scale-8000 dataset remains incomplete.
