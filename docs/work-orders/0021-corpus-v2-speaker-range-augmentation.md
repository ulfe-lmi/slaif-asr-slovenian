# Work Order 0021: Corpus-v2 Speaker-range Augmentation Diagnostic

Status: completed in PR; pending strategic review

Repository: `ulfe-lmi/slaif-asr-slovenian`

This work order authorizes a bounded `DIAGNOSTIC_ONLY` experiment that tests
one scientific variable relative to Experiment 0008's `a100_batched` arm:
deterministic speaker-range resampling of the selected-training waveforms.

The experiment keeps fixed the selected-training membership, transcripts,
utterance/source families, base checkpoint, 2,048-value `sl-SI` prompt-column
trainable surface, optimizer, learning rate, batch size, epoch count, sample
exposures, batch-order algorithm, seed, precision, evaluation manifests,
evaluation batch size, normalization, and decision thresholds.

The augmentation policy is
`configs/augmentation/corpus_v2_speaker_range_resampling_v1.json`. It defines
five proxy profiles: `child_like_proxy`, `high_voice_proxy`, `clean`,
`low_voice_proxy`, and `elder_like_proxy`. The non-clean profiles use
`scipy.signal.resample_poly` with exact rational factors 4/5, 9/10, 11/10, and
6/5. These are acoustic proxies only; they are not represented as actual age,
gender, or independent-speaker identities.

Before training, the PR must commit the diagnostic certificate
`docs/data-certificates/sl-corpus-v2-speaker-range-diagnostic-v1.json` with
status `DIAGNOSTIC_ONLY`. No `TRAINING_ELIGIBLE` status is issued. No resulting
checkpoint may become an accepted parent.

The required training arm is `speaker_range_augmented_batch8`: batch size 8,
12 epochs, 1,920 sample exposures, 240 optimizer steps, AdamW, learning rate
0.01, FP32, TF32 disabled, SpecAugment disabled, and only the Slovenian
prompt-column delta trainable.

The arm is evaluated on the original clean audio for selected synthetic
training, independent synthetic holdout, FLEURS-v2, and ARTUR-J using the
canonical batch-size-1 evaluation policy. Raw audio, generated variants,
references, hypotheses, local manifests, model deltas, checkpoints, and monitor
CSVs remain ignored local artifacts.

Outcome: the diagnostic completed and was classified as
`SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED`. Synthetic-holdout gain remained,
but the real-regression burden increased relative to the clean batch-8 arm. No
checkpoint is accepted.
