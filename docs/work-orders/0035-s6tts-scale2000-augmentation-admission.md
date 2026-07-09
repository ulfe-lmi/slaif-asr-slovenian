# Work Order 0035: S6TTS Transcript-Preserving Augmentation Admission

Status: `DIAGNOSTIC_ONLY`

This work order admits transcript-preserving augmented views derived from the
already admitted S6TTS clean view for the fixed scale-2000 text corpus. It
performs no model training and issues no `TRAINING_ELIGIBLE` status.

## Scope

- Source clean view: `sl-corpus-v4-s6tts-clean-view-v1`
- Augmented view: `sl-corpus-v4-s6tts-augmented-view-v1`
- Semantic rows: 16000
- Augmentation profiles per row: 11
- Expected augmented files: 176000

## Boundaries

Generated S6TTS audio and augmented audio remain ignored local synthetic
diagnostic artifacts. Public audio release, checkpoint acceptance, model
publication, and training eligibility are out of scope.

No real speech, immutable gate material, controller-development material, or
final blind-test material may be used for augmentation construction.

## Evidence

The PR commits only privacy-safe aggregate certificate and report evidence.
Local manifests, generated WAVs, raw text, local absolute paths, logs, CSV/TSV
monitor outputs, model artifacts, checkpoints, and predictions remain outside
Git.
