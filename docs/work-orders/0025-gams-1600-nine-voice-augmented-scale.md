# Work Order 0025: GaMS 1600 Nine-Voice Augmented Scale

Status: in progress on branch `exp/gams-1600-nine-voice-augmented-scale`

This work order is a single `DIAGNOSTIC_ONLY` PR for a 200x effective synthetic
exposure diagnostic. It covers GaMS text generation, exact whole-file human
text admission, Piper and Supertonic synthesis, transcript-preserving
augmentation, acoustic validation, one fixed frozen-base Slovenian joint-adapter
training arm, and fast batch-32 directional evaluation.

The first implementation phase stops at the mandatory human checkpoint:

```text
ACCEPT or REJECT sl-corpus-v3-gams-1600-training-v1 <SHA256> 1600
```

No TTS, augmentation, training, or evaluation may run until the exact fixed
1,600-row text file has a whole-file human decision bound to its SHA256 and row
count.

Generation-budget update:

- The first bounded GaMS pass produced enough total admissible rows but left
  prompt cell `cell33` below the required 40 fixed rows.
- The human operator directed increasing the generation budget until a full
  fixed set is available.
- A later validator pass found that the set was still structurally too
  concentrated around common openings, so targeted diversity retries were added
  for the affected prompt cells with explicit prompt guidance to avoid those
  openings.
- Validation thresholds, protected gates, partition checks, and the 40-per-cell
  final selection rule are unchanged.

Required boundaries:

- status remains `DIAGNOSTIC_ONLY`;
- `TRAINING_ELIGIBLE` is not issued;
- accepted parent remains `none`;
- no raw generated text, audio, predictions, model, adapter, checkpoint, local
  manifests, progress logs, or monitor CSVs are committed;
- the 200x claim refers to deterministic exposure count, not independent
  linguistic information.
