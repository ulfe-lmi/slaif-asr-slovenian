# ADR 0009: Fixed Scale-2000 Trainable-Surface Sweep

## Status

Accepted as the historical authorization for the bounded diagnostic program
defined by Work Orders 0037, 0038, 0039, 0040, 0043, 0045, 0046, and 0047.
Every named run is complete. This ADR is not continuing authorization for
Surface08, scale-8000 training, Surface09, or full-model training.

## Decision

Authorize a fixed-data trainable-surface sweep on original scale-2000 augmented
data. Each experiment changes one named trainable surface while preserving the
corpus, exposure schedule, validation protocol, target language, streaming
context, and reporting contract.

Phase 1 authorizes only
`SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK`: the RNNT decoder, RNNT joint,
and exactly the final encoder block. ARTUR controller-dev may provide aggregate
run-control under ADR 0008. FLEURS-v2 and ARTUR-J remain post-selection
directional gates and cannot select a checkpoint.

## Phase 2 / Work Order 0038

Authorize
`SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS`: the RNNT decoder, RNNT
joint, and exactly the final two encoder blocks. In the pinned live model these
must resolve to `encoder.layers.22` and `encoder.layers.23`; execution fails
closed if that identity cannot be proved.

Phase 2 permits:

- training the decoder and joint;
- training exactly the final two encoder blocks at the lower encoder learning
  rate;
- using only the original scale-2000 augmented corpus v4 and its fixed
  exposure schedule;
- using ARTUR controller-dev aggregate run-control under ADR 0008.

Phase 2 forbids:

- full encoder training or encoder blocks below the final two;
- frontend, subsampling, or preprocessor training;
- tokenizer or prompt labels, tables, embeddings, or fusion-path changes;
- S6TTS, scale-8000, database-extension, or real-speech training data;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- Surface06, checkpoint acceptance, model publication, `TRAINING_ELIGIBLE`, or
  an `accepted_parent` change.

## Phase 3 / Work Order 0039

Authorize
`SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS`: the RNNT decoder,
RNNT joint, and exactly the final four encoder blocks. In the pinned live model
these must resolve to `encoder.layers.20` through `encoder.layers.23`;
execution fails closed if that identity cannot be proved.

Phase 3 permits:

- training the decoder and joint;
- training exactly the final four encoder blocks at `1.0e-5`;
- using only the original scale-2000 augmented corpus v4 and its fixed
  exposure schedule;
- using ARTUR controller-dev aggregate run-control under ADR 0008.

Phase 3 forbids:

- full encoder training or encoder blocks below the final four;
- frontend, subsampling, or preprocessor training;
- tokenizer or prompt labels, tables, embeddings, or fusion-path changes;
- S6TTS, scale-8000, database-extension, or real-speech training data;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- Surface07 or any fusion-combined experiment;
- checkpoint acceptance, model publication, `TRAINING_ELIGIBLE`, or an
  `accepted_parent` change.

## Phase 4 / Work Order 0040

Authorize `SURFACE_07_TOP_ENCODER_PLUS_PROMPT_ACOUSTIC_FUSION`: the RNNT
decoder, RNNT joint, exactly the final four encoder blocks, and exactly one
separable prompt/acoustic fusion bridge. The pinned model exposes that bridge
as `prompt_kernel`, a post-concatenation `1152 -> 2048 -> 1024` MLP over 1024
encoder features and a 128-way one-hot prompt. The one-hot prompt dictionary is
configuration, not a learnable embedding or table. Execution fails closed if
this structure or ownership cannot be proved from the live model.

Phase 4 permits:

- training the decoder and joint;
- training exactly `encoder.layers.20` through `encoder.layers.23` at
  `1.0e-5`;
- training exactly `prompt_kernel` at `5.0e-5`;
- using only the original scale-2000 augmented corpus v4 and its fixed
  exposure schedule;
- using ARTUR controller-dev aggregate run-control under ADR 0008.

Phase 4 forbids:

- full encoder training or encoder blocks below the final four;
- frontend, subsampling, or preprocessor training;
- tokenizer, prompt labels, prompt tables, prompt embeddings, language-ID
  mappings, or target-language machinery changes;
- training any prompt/fusion module other than the proven `prompt_kernel`;
- S6TTS, scale-8000, database-extension, or real-speech training data;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- Surface08, Surface09, checkpoint acceptance, model publication,
  `TRAINING_ELIGIBLE`, or an `accepted_parent` change.

## Phase 5 / Work Order 0043

Authorize exactly one `SURFACE_08_FULL_ENCODER` boundary diagnostic. This is not
general authorization for full-encoder training, a release or promotion path,
or authorization for full-model training.

Phase 5 permits:

- training the decoder and joint;
- training exactly all 24 encoder layers at `5.0e-6`;
- training the previously proven separable `prompt_kernel` bridge at `5.0e-5`;
- using only the original scale-2000 augmented corpus v4 and its fixed
  exposure schedule;
- using ARTUR controller-dev aggregate run-control under ADR 0008;
- using exactly one visible RTX 3090, or another already-authorized single GPU
  only if the RTX 3090 is unavailable.

Phase 5 forbids:

- preprocessor, frontend, or subsampling training outside `encoder.layers`;
- tokenizer, prompt labels, prompt tables, prompt embeddings, prompt identity,
  language-ID mappings, or target-language machinery changes;
- training any prompt/fusion module other than the proven `prompt_kernel`;
- full-model training or Surface09;
- S6TTS, scale-8000, database-extension, or real-speech training data;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- checkpoint acceptance, model publication, `TRAINING_ELIGIBLE`, or an
  `accepted_parent` change.

## Reason

The strongest clean WER/CER reduction came from original scale-2000 augmented
audio with decoder+joint RNNT training. Later data mixtures introduced
tradeoffs. The next controlled question is which model surface should move,
not which data variant should be added.

## Permitted In Phase 1

- Train `model.decoder`.
- Train `model.joint`.
- Train exactly the final encoder block, discovered as `encoder.layers.23` in
  the pinned live model.
- Use the fixed scale-2000 exposure schedule without replacement or expansion.
- Use ARTUR controller-dev aggregate metrics for run-control under ADR 0008.

## Forbidden

- Full encoder training or any lower encoder block.
- Subsampling, frontend, or preprocessor training.
- Tokenizer or prompt labels, tables, or embeddings changes.
- S6TTS, scale-8000, database-extension, or real-speech training data.
- FLEURS-v2 or ARTUR-J checkpoint selection.
- Text-only objectives, temporary LM heads, or adapters.
- Checkpoint acceptance, model release, `TRAINING_ELIGIBLE`, or an
  `accepted_parent` change.

## Consequences

These are diagnostic exceptions to the default synthetic-only encoder freeze,
not general authorization to train encoder parameters. Phase 5 permits one
named Surface08 run only. Surface09, full-model training, and any later
full-encoder run require separate governance.

## Phase 6 / Work Order 0045

Work Order 0045 authorizes exactly one data-axis follow-up using the already
proved Surface08 parameter boundary:

- start from the untouched Nemotron base;
- train decoder, joint, all 24 encoder layers, and the proven separable
  `prompt_kernel`;
- use only the admitted scale-8000 clean synthetic pool;
- create transcript-preserving augmented waveform views in memory with the
  deterministic PR #50 OTF implementation;
- keep the fixed 320,000-exposure hard cap and use ARTUR controller-dev only
  for aggregate run-control under ADR 0008.

Phase 6 does not change the fixed-data surface-sweep evidence. It is one
explicitly human-authorized comparison of the linguistic-data axis after the
surface ladder selected Surface08.

Phase 6 forbids:

- scale-2000, S6TTS, database-extension, or real speech as training sources;
- pre-rendered augmented WAVs;
- initialization from Surface08 or any other adapted checkpoint;
- preprocessor, frontend, subsampling, tokenizer, prompt-identity,
  language-ID, or target-language machinery changes;
- Surface09 or full-model training;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- checkpoint acceptance, `TRAINING_ELIGIBLE`, model publication, or accepted
  parent changes.

This is not general authorization for scale-8000 full-encoder training. Any
follow-up schedule, surface, initialization, or data change requires separate
human review and authorization.

Phase 6 completed as Experiment 0031 with classification
`SCALE8000_OTF_SURFACE08_NEW_BEST_DIRECTIONAL`. Its selected checkpoint scored
39.353/12.002/0 on FLEURS-v2 and 39.974/12.251/0 on ARTUR-J. Standard
deterministic OTF is the selected balanced augmentation control. The one-run
Phase 6 authorization is consumed.

## Phase 7 / Work Order 0046

Work Order 0046 authorizes exactly one matched augmentation-intensity
diagnostic against Experiment 0031:

- retain `SURFACE_08_FULL_ENCODER`, including decoder, joint, all 24 encoder
  layers, and the proven separable `prompt_kernel`;
- start again from the untouched Nemotron base;
- use only the admitted scale-8000 clean synthetic pool;
- retain FP32, TF32-disabled, effective-batch-8 training and the 144,000
  virtual-exposure cap;
- replace the standard single-profile OTF policy with deterministic StrongAug
  v1 chains, using a 70/30 standard/strong mixture in rounds 1-4 and a 25/75
  mixture in rounds 5-9;
- use ARTUR controller-dev only for aggregate run-control under ADR 0008.

StrongAug v1 permits only in-memory transcript-preserving procedural
degradations. Chains contain at most three bounded operations, procedural noise
never falls below 8 dB SNR, and no augmented WAV is rendered.

Phase 7 forbids:

- changing the model surface, base checkpoint, scale-8000 text, clean audio
  reservoir, effective batch, optimizer, or learning-rate groups;
- scale-2000, S6TTS, database-extension, or real speech as training sources;
- external uncertified noise data or pre-rendered augmented WAVs;
- initialization from Experiment 0031 or any adapted checkpoint;
- preprocessor, frontend, subsampling, tokenizer, prompt-identity,
  language-ID, or target-language machinery changes;
- Surface09 or full-model training;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- checkpoint acceptance, `TRAINING_ELIGIBLE`, model publication, or accepted
  parent changes.

This is a single augmentation-policy comparison, not general StrongAug or
scale-8000 full-encoder authorization.

Phase 7 completed as Experiment 0032 with classification
`STRONGAUG_V1_REAL_GATE_REGRESSION`. Its selected checkpoint scored
39.904/12.343/0 on FLEURS-v2 and 39.406/12.357/0 on ARTUR-J. StrongAug v1 is
negative domain-tradeoff evidence and is not selected. The one-run Phase 7
authorization is consumed.

## Phase 8 / Work Order 0047

Work Order 0047 authorizes exactly one matched ParametricVoiceAug v1
diagnostic against Experiment 0031:

- retain `SURFACE_08_FULL_ENCODER`, including decoder, joint, all 24 encoder
  layers, and the proven separable `prompt_kernel`;
- start again from the untouched Nemotron base;
- use only the admitted scale-8000 clean synthetic pool;
- retain FP32, TF32-disabled, effective-batch-8 training and the 144,000
  virtual-exposure cap;
- use deterministic, language-agnostic, in-memory speech-shape transforms with
  key `scale8000-otf-parametric-voiceaug-v1`;
- use a 50/50 standard/parametric mixture in rounds 1-4 and a 10/90 mixture in
  rounds 5-9;
- use ARTUR controller-dev only for aggregate run-control under ADR 0008.

ParametricVoiceAug v1 may perturb F0, tempo, spectral envelope, aperiodicity,
channel, companding, room response, procedural noise, and dynamics. It may not
use target identities, Slovenian-specific speech models, downloaded voice
models, voice conversion, voice cloning, or pre-rendered augmented WAVs.

Phase 8 forbids:

- changing the model surface, base checkpoint, NeMo revision, scale-8000 text,
  clean audio reservoir, effective batch, optimizer, or learning-rate groups;
- scale-2000, S6TTS, scale-32000, database-extension, or real speech as
  training sources;
- initialization from Experiment 0031, Experiment 0032, or any adapted
  checkpoint;
- preprocessor, frontend, subsampling, tokenizer, prompt-identity,
  language-ID, or target-language machinery changes;
- target-speaker conversion or external person, celebrity, character, or
  community voice models;
- Surface09 or full-model training;
- FLEURS-v2 or ARTUR-J checkpoint selection;
- checkpoint acceptance, `TRAINING_ELIGIBLE`, model publication, or accepted
  parent changes.

This is a single augmentation-family comparison. StrongAug v1 remains a
negative diagnostic comparator and is not the default ParametricVoiceAug
recipe.

Phase 8 completed as Experiment 0033 with classification
`PARAMETRIC_VOICEAUG_V1_REAL_GATE_REGRESSION`. Its selected checkpoint scored
40.371/12.186/0 on FLEURS-v2 and 39.318/12.017/0 on ARTUR-J. The selected
checkpoint SHA256 remains
`ffe421a671036b0b5bc8a75315c519c50a28da0be17c93e3caca503ce77dd9cd`.
ParametricVoiceAug v1 is negative domain-tradeoff evidence and is not selected.
The one-run Phase 8 authorization is consumed.

## Completed Program Disposition

- Selected balanced augmentation control: standard deterministic OTF from
  Experiment 0031.
- Broad augmentation sweep: closed.
- Surface08: fixed only as the model surface for a future explicitly governed
  data-axis comparison.
- Surface09: not authorized.
- Full-model training: not authorized.
- Accepted checkpoint: none.
- `TRAINING_ELIGIBLE`: not issued.
- Model publication: not authorized.
