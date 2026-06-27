# Scale-8000 Dual-GPU Generation

Status: `TEXT_ACCEPTED`

This report is privacy-safe scale-8000 text-generation evidence. It contains no raw generated text, candidate IDs, audio paths, hypotheses, model artifacts, or monitor CSV data.

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
- Inherited semantic rows: `16000`
- Newly selected semantic rows: `48000`
- Combined semantic rows: `64000`
- Combined text SHA256: `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd`
- New-addition text SHA256: `88a9cd19b726cf0fe44fd9f0d2e19d69eba392cc1d65439d9d5f3c0ca53f9887`

## Inclusion

- Policy: `prefix`
- Evidence: the 16,000 inherited scale-2000 rows remain byte-for-byte unchanged and appear before all newly generated scale-8000 rows in the combined local corpus.
- Per-cell combined count: `1600`
- Per-cell inherited/new count: `400` / `1200`

## Dual-GPU Generation

- `gpu0`: physical GPU `0`, `CUDA_VISIBLE_DEVICES=0`, logical device `cuda:0`, `NVIDIA A100-SXM4-80GB`, completed attempts `1142`
- `gpu1`: physical GPU `1`, `CUDA_VISIBLE_DEVICES=1`, logical device `cuda:0`, `NVIDIA A100-SXM4-80GB`, completed attempts `1142`
- Initial attempts: `1200`
- Refill attempts: `1084`
- Total attempts: `2284`
- Initial requested rows: `72000`
- Refill requested rows: `65040`
- Total requested rows: `137040`
- Parsed generated rows: `158573`
- Structurally admissible new rows: `65278`
- Selected new rows: `48000`

## Rejections

- `surface_duplicate`: `89016`
- `number_masked_collision`: `3367`
- `schema_invalid`: `639`
- `metadata_leak`: `271`
- `holdout_surface_overlap`: `2`
- Total rejected rows: `93295`

## Scale Plan

- Clean files/views planned: `576000`
- Augmented files/views planned: `704000`
- Total views/exposures planned: `1280000`
- Batch-8 optimizer steps if later authorized: `160000`
- Exposure multiplier versus the 160-item reference: `8000x`

The `8000x` figure refers to deterministic exposure count, not independent linguistic information.

## Review Capsule

A local ignored review capsule was created with:

- fixed combined rows: `64000`
- generated rows: `158573`
- new-addition rows: `48000`
- rejected rows: `93295`
- review TSV lines including header: `64001`

Required whole-file decision:

```text
ACCEPT or REJECT sl-corpus-v5-scale8000-training-v1 e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd 64000
```

## Storage Preflight

- Runtime storage: external data runs root
- Available bytes: `3286672367616`
- Projected new bytes: `247023026364`
- Required free bytes with safety margin: `308778782955`
- Sufficient: `true`

## Boundary

No audio synthesis, acoustic validation, training, or ASR evaluation has been run for scale-8000. This corpus is not `TEXT_ACCEPTED` and no `TRAINING_ELIGIBLE` status exists.

## Whole-File Decision

- Outcome: `ACCEPT`
- Decision ID: `human-scale8000-decision-v1`
- Review revision: `human-scale8000-review-v1`
- Bound SHA256: `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd`
- Bound rows: `64000`
