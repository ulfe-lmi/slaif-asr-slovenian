# Scale-8000 Dual-GPU Generation

Status: `ENVIRONMENT_BLOCKED`

This report is privacy-safe planning and preflight evidence. It contains no raw generated text, audio paths, hypotheses, model artifacts, or monitor CSV data.

## Canonical Pass

- `.venv/bin/python -m unittest discover -s tests`: passed, 298 tests.
- `.venv/bin/python -m py_compile $(git ls-files '*.py')`: passed after a direct rerun.
- `.venv/bin/python scripts/check_repository.py`: passed.
- `.venv/bin/python -m pip check`: passed.
- `.venv-gams/bin/python -m pip check`: passed.
- `.venv-piper/bin/python -m pip check`: passed.
- `.venv-supertonic/bin/python -m pip check`: passed.
- `bash -n scripts/*.sh`: passed.
- `git diff --check`: passed.

The first py-compile wrapper invocation was invalid because newline-separated
file names were embedded in a shell string. The required command was rerun
directly and passed. No canonical result blocked generation.

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

- Available bytes: `211107639296`
- Projected new bytes: `247023026364`
- Required free bytes with safety margin: `308778782955`
- Sufficient: `False`

## Decision

Generation must not begin while storage preflight is insufficient. This is an environment blocker, not a corpus acceptance decision.
