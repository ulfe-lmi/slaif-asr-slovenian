# Work Order 0026: GaMS 16000 Scale-2000 Text-Only Diagnostic

Status: in progress on branch `exp/gams-16000-scale2000-text-only`

This work order is a single `DIAGNOSTIC_ONLY` PR for a direct semantic-text
scale ablation. It expands the accepted scale-200 corpus from 1,600 semantic
training items to a strictly nested 16,000-item corpus while holding the
voice, augmentation, model, training, and directional evaluation protocols
fixed.

The scientific change relative to Experiment 0013 is only:

```text
semantic training items: 1,600 -> 16,000
```

All downstream acoustic and model settings remain unchanged:

- nine clean voice sources;
- eleven transcript-preserving augmentation profiles;
- 20 exposure rounds;
- batch size 8 training;
- frozen-base Slovenian RNNT joint adapter with bottleneck 32;
- AdamW learning rate 0.001;
- FP32 with TF32 disabled;
- batch-32 directional evaluation only.

The 16,000-row corpus must be a strict superset of:

```text
sl-corpus-v3-gams-1600-training-v1
SHA256 9a23df00734193eca0a52bf9b3dae385ff6087d0282529f3f4cb1a28bbf6dccf
rows 1600
```

No inherited row may be rewritten, renumbered, regenerated, or reselected.

Required stop point after text generation:

```text
ACCEPT or REJECT
sl-corpus-v4-gams-16000-training-v1
<COMBINED_SHA256>
16000
```

The experiment may continue only after an exact whole-file decision bound to
the combined corpus SHA256 and row count.

Retry policy update:

- The human lead explicitly overrode the original finite retry budgets on
  2026-06-26.
- Generation now retries deficient cells until the combined corpus is
  structurally valid.
- The override does not weaken validators, reuse rejected text, expose
  protected text, or infer human acceptance.
- Each refill round remains bounded and resumable so progress and evidence stay
  auditable.

Boundaries:

- `TRAINING_ELIGIBLE` is not issued;
- accepted parent remains `none`;
- generated text, audio, predictions, model artifacts, adapters, checkpoints,
  local manifests, progress logs, and monitor CSVs remain ignored;
- batch-1 canonical evaluation is not run;
- the 2,000x figure refers to deterministic exposure count, not independent
  linguistic information.
