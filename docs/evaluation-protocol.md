# Evaluation Protocol

## Purpose

Evaluation determines whether a challenger improves Slovenian, and eventually
Slovenian-English, streaming recognition under validation-only real-speech
gates. Multilingual regression evidence is secondary unless a work order makes
it an explicit acceptance condition.

## Required checkpoint identity

Every result records:

- base model repository and revision;
- base checkpoint checksum;
- parent accepted checkpoint checksum;
- challenger checksum;
- NeMo revision;
- repository commit;
- configuration hash;
- trainable-parameter summary.

## Slovenian metrics

Report:

- raw WER;
- normalized WER;
- CER;
- `č`, `š`, and `ž` error rates;
- deletion and insertion rates;
- proper-name WER;
- number/date/time accuracy;
- punctuation accuracy;
- capitalization accuracy;
- foreign-script leakage;
- silence/noise hallucination rate.

Active-curriculum reports must also distinguish:

- corpus WER from total word edits divided by total reference words;
- corpus CER from total character edits divided by total reference characters;
- mean utterance WER/CER;
- median utterance WER/CER;
- empty-hypothesis counts.

Do not label mean utterance WER as corpus WER.

The committed Slovenian normalizer for development-gate reporting is
`sl-asr-normalization-v1`. It applies NFC normalization, Slovenian-aware
lowercasing, apostrophe normalization, whitespace normalization, punctuation
removal, and hyphen-to-space handling while preserving Slovenian letters
including `č`, `š`, and `ž`. It does not expand numbers with an LLM.

## Streaming settings

Evaluate every release candidate at the supported context settings:

```text
[56,0]
[56,1]
[56,3]
[56,6]
[56,13]
```

Record the corresponding advertised latency point with the result.

Track:

- final WER/CER;
- first-token latency;
- final-token latency;
- partial-transcript churn;
- disagreement between low-latency and balanced settings;
- real-time factor;
- peak GPU memory.

Current A100 development-gate execution uses the measured policy in
[`configs/evaluation/a100_streaming_batch_policy.json`](../configs/evaluation/a100_streaming_batch_policy.json):
batch size 1, no duration bucketing, FP32 cache-aware inference, and TF32
disabled. Batch size 1 remains the scientific reference mode. Experiment 0006
tested duration-bucketed batch sizes 1 through 128 on FLEURS-v2; all sizes
above 1 changed at least some transcripts and are not eligible canonical
comparison modes.

## Data roles

Separate reports for:

- synthetic candidate pool;
- selected synthetic hard set;
- controller-development real Slovenian;
- immutable real-Slovenian gate;
- final blind test;
- multilingual regression.

Never present a synthetic-only score as real-world Slovenian performance.

Current immutable Slovenian development gates:

- `fleurs-sl-si-test-full-v2`: complete official `google/fleurs` Slovenian
  `sl_si` test split at revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, with occurrence-unique
  sample IDs and WAV filenames derived from source-row indexes;
- `artur-j-public-gate-v1`: deterministic 256-utterance project gate from
  ARTUR-J `Artur-J-Splosni` standardized orthographic transcripts and public
  audio.

These are not final blind tests. Raw references and hypotheses remain local
ignored artifacts. Future challengers must evaluate both gates before any
accepted-parent decision.

Real Slovenian speech is validation-only. Gate audio and references must not be
used for model training, synthetic prompt construction, selected-training
membership, early stopping, hyperparameter tuning, per-sample steering, or
adapter-surface selection. Aggregate real-gate metrics may compare completed
challengers and determine whether a completed challenger is worth the next
governed step.

`artur-controller-dev-v1` is a separate controller-development partition under
ADR 0008. It may support aggregate per-round run-control and future
early-stopping rules only when a work order explicitly authorizes that use. It
is not an immutable gate and cannot support checkpoint acceptance, public
quality claims, or release claims.

The untouched-base FLEURS-v2 baseline is recorded in
[`docs/experiments/0006-a100-batched-streaming-evaluation.md`](experiments/0006-a100-batched-streaming-evaluation.md).
Historical FLEURS-v1 numbers in Experiment 0003 remain deprecated audit
evidence only.

Historical `fleurs-sl-si-test-full-v1` evidence is deprecated. It used the
non-unique FLEURS source ID for manifest sample IDs and WAV filenames, causing
path overwrites; its 834 manifest rows represented only 347 unique sample
identities. v1 FLEURS metrics must not be used as complete-split quality
evidence. ARTUR-J measurements are unaffected.

The first project-generated Round 1 challenger evaluated both gates and was
rejected: FLEURS and ARTUR-J normalized WER/CER regressed and empty hypotheses
increased. This reinforces that selected-synthetic improvement is diagnostic
only.

The first Slovenian residual-adapter proof reused the same Round 1 corpus and
gates. Rank 16 and rank 64 adapters both improved the fixed synthetic holdout
but regressed FLEURS and ARTUR-J normalized WER/CER beyond promotion
thresholds. This is classified as synthetic-only evidence, not real-speech
generalization.

Synthetic holdout scores are diagnostic even when the holdout is properly
constructed. They must not be described as real-generalization evidence, and
they cannot make a challenger an accepted parent without real-gate
non-regression and the precommitted promotion rules. The Round 1 v1 synthetic
corpora are additionally retired by the training-data constitution and must not
be reused for future promotion-oriented experiments.

The corpus-v2 candidate source and independent synthetic holdout have reached
text and audio admission, and a privacy-safe `SCORING_AUTHORIZED` certificate
permitted ASR scoring plus selected-training construction. Those steps now
produce aggregate-only scoring evidence and a selected-training manifest with
`SELECTED_TRAINING_MANIFEST_READY` status. This does not permit model training
or `TRAINING_ELIGIBLE`, and the synthetic holdout remains diagnostic rather
than real-speech evidence.

Work Order 0020 used that manifest once under a named `DIAGNOSTIC_ONLY`
exception. The corpus-v2 prompt-column arms were evaluated with batch size 1,
no duration bucketing, and the fixed FLEURS-v2 and ARTUR-J gates. The outcome
was synthetic-only, no checkpoint was accepted, and the result is not a public
quality or real-speaker generalization claim.

Work Order 0021 kept the same evaluation policy and tested only deterministic
speaker-range resampling of training audio. It did not augment evaluation
audio. The outcome was `SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED`; no
checkpoint was accepted and the result is not evidence of real-speaker,
multi-speaker, child, elderly, or gender coverage.

Work Order 0022 kept the same evaluation policy and tested only one
frozen-base RNNT joint-hidden adapter on the original clean Piper training
audio. Evaluation used batch size 1, no duration bucketing, and the fixed
FLEURS-v2 and ARTUR-J gates. The outcome was
`SL_JOINT_ADAPTER_SYNTHETIC_ONLY`; no adapter or checkpoint was accepted.

Work Order 0023 kept the same batch-size-1 evaluation policy and tested the
same frozen-base joint-adapter surface trained on Supertonic 3 preset
multi-voice synthetic audio. The outcome was
`SUPERTONIC3_MULTIVOICE_MITIGATES_PIPER_REGRESSION`: synthetic diagnostics
improved and the real-regression burden fell, but real-gate regression was not
eliminated. No adapter or checkpoint was accepted.

## Acceptance comparison

Compare the challenger with its parent accepted checkpoint, not only with the original base.

Initial project guardrails, subject to revision through an ADR:

- targeted hard-set improvement: at least 10% relative;
- normalized FLEURS corpus WER: no more than 1.0 absolute-point regression;
- normalized ARTUR-J corpus WER: no more than 1.0 absolute-point regression;
- normalized FLEURS corpus CER: no more than 1.5 absolute-point regression;
- normalized ARTUR-J corpus CER: no more than 1.5 absolute-point regression;
- empty-hypothesis count must not increase on either real gate;
- multilingual macro WER after shared-weight training: no more than 0.5 absolute WER regression;
- no new systematic foreign-script leakage;
- no increased silence hallucination;
- no material low-latency degradation;
- parameter-diff integrity passes.

Statistical uncertainty should be reported using paired bootstrap intervals when the sample size supports it.

A claim of real-speech improvement requires material improvement on at least
one real gate and non-regression on the other. Improvement only on synthetic
splits is classified as synthetic-only and cannot accept a parent checkpoint.
The fact that a challenger was trained on synthetic audio is not itself a
rejection reason; validation-only real-gate behavior is decisive.

Batch-32 directional evaluation may be used for faster development decisions
when explicitly configured as noncanonical and promotion-ineligible. It must
not replace canonical batch-1 evidence for acceptance, release, or public
quality claims.

## Normalization

The report must state the normalization policy and its version. Raw and normalized results remain distinguishable.

## Blind-test discipline

The final blind test is opened only for a named milestone and human-approved release decision. Once used, its status and resulting potential tuning exposure are recorded.

## Output artifacts

Commit only summary tables and privacy-safe aggregate reports. Per-utterance outputs containing private references remain in controlled storage.
