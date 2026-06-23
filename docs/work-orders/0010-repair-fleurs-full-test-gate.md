# Work Order 0010: Repair FLEURS Full-Test Gate Identity

Status: ready for execution

## Objective

Repair the FLEURS Slovenian full-test development gate identity defect by
creating a new immutable gate, `fleurs-sl-si-test-full-v2`, while preserving
the historical v1 metadata and reports for auditability.

## Problem

The v1 FLEURS builder used the upstream FLEURS source `id` as both `sample_id`
and WAV filename stem. That source field is not unique per audio occurrence, so
repeated source IDs produced duplicate manifest identities, duplicate audio
paths, and silent WAV overwrites. The resulting 834 manifest rows represented
only 347 unique sample identities.

## Required Repair

- Preserve `fleurs-sl-si-test-full-v1` as deprecated historical evidence.
- Add `fleurs-sl-si-test-full-v2` with:
  - dataset `google/fleurs`;
  - configuration `sl_si`;
  - split `test`;
  - revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
  - license CC BY 4.0;
  - construction algorithm `fleurs-sl-si-test-full-v2`.
- Derive occurrence identity from deterministic source-row enumeration:

```text
sample_id = f"fleurs-sl-si-test-occ-{source_row_index:05d}"
```

- Use the same identity for each WAV filename.
- Retain upstream `source_id` only as provenance; it is allowed to repeat.
- Fail closed before writing audio if planned row indexes, sample IDs, or
  relative audio paths collide.
- Commit privacy-safe v2 metadata with hashes, durations, gender counts, source
  row indexes, and source IDs only.
- Add a reusable real-gate verifier.
- Update documentation so canonical FLEURS references point to v2 and
  historical v1 metrics are marked deprecated.

## Verification

Run:

```text
.venv/bin/python -m unittest discover -s tests -p 'test_real_eval.py'
.venv/bin/python scripts/build_real_evaluation_gates.py --gate fleurs
.venv/bin/python scripts/verify_real_evaluation_gate.py \
  --manifest runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl \
  --metadata docs/evaluation-gates/fleurs-sl-si-test-full-v2.metadata.json \
  --expected-gate-id fleurs-sl-si-test-full-v2 \
  --expected-rows 834
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
```

No generated audio, local manifests, reference sidecars, checkpoints, or model
artifacts may be committed.

## Non-Goals

- Do not rerun ASR baseline metrics.
- Do not train or fine-tune a model.
- Do not alter ARTUR-J construction.
- Do not generate synthetic data.
- Do not run GaMS, Piper, or Nemotron.
- Do not use GPU time.
