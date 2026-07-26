# Work Order 0047: Surface08 Scale-8000 OTF ParametricVoiceAug v1

## Goal

Run one matched Surface08 scale-8000 diagnostic that changes only the
deterministic on-the-fly augmentation family relative to Experiment 0031.

The hypothesis is that bounded, language-agnostic speech-shape changes may
reduce synthetic-to-real speaker, excitation, room, and channel mismatch more
effectively than generic StrongAug intensity.

## Fixed Control

- untouched `nvidia/nemotron-3.5-asr-streaming-0.6b` base;
- Surface08: decoder, joint, all encoder layers, and `prompt_kernel`;
- `sl-corpus-v5-scale8000-training-v1`, 64,000 semantic rows and 576,000 clean
  synthetic files;
- 144,000 virtual exposures, effective batch 8, FP32, TF32 disabled;
- ARTUR controller-dev aggregate checkpoint selection only;
- post-selection directional batch-32 Piper, Supertonic, FLEURS-v2, and
  ARTUR-J evaluation.

Experiment 0031 standard OTF is the critical comparator. Experiment 0032
StrongAug v1 is a negative diagnostic comparator only.

## ParametricVoiceAug v1

The augmentation key is
`scale8000-otf-parametric-voiceaug-v1`. Deterministic content keys select
bounded two- or three-operation chains from:

- duration-preserving pitch perturbation;
- tempo perturbation;
- STFT log-spectral-envelope formant proxy warping;
- controlled high-frequency aperiodicity/noisiness;
- channel and telephone filtering;
- companding and moderate codec proxies;
- short-to-moderate synthetic room response;
- procedural noise;
- gain, compression, limiting, and rare soft clipping.

The implementation uses no target identity, Slovenian-specific speech model,
downloaded voice model, voice conversion, or voice cloning.

Rounds 1-4 use 50% standard OTF and 50% ParametricVoiceAug. Rounds 5-9 use 10%
standard OTF and 90% ParametricVoiceAug. Selection and parameters are stable
across process restarts, worker counts, worker ordering, and queue timing.

## Safety And Scope

- Augmented waveforms exist in memory only during training.
- Local audit WAVs, checkpoints, logs, manifests, and absolute paths remain
  ignored and uncommitted.
- No real speech, S6TTS, scale-2000, scale-32000, or database-extension rows
  enter training.
- FLEURS-v2 and ARTUR-J do not select checkpoints.
- No checkpoint acceptance, `TRAINING_ELIGIBLE`, model publication, or
  `accepted_parent` change is authorized.
- The execution agent must not merge its own PR.

## Required Evidence

Before training, record dependency provenance, deterministic replay,
waveform-safety audit, transcript preservation, and loader throughput. During
training, record per-round controller metrics and loader telemetry. Evaluate
only the ARTUR controller-dev selected checkpoint and commit only aggregate,
privacy-safe evidence.
