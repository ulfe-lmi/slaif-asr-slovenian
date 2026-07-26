# Work Order 0046: Surface08 Scale-8000 OTF StrongAug v1

## Goal

Measure whether stronger deterministic transcript-preserving in-memory
augmentation improves real-speech transfer over Experiment 0031 while holding
the Surface08 model, scale-8000 clean reservoir, untouched base, optimizer,
effective batch, run-control policy, evaluation suite, and 144,000-exposure
budget fixed.

## Authorized Change

Use `scale8000-otf-strongaug-v1` with this curriculum:

- rounds 1-4: 70% standard OTF and 30% StrongAug v1;
- rounds 5-9: 25% standard OTF and 75% StrongAug v1.

Strong chains contain two or three bounded procedural operations. Noise remains
at or above 8 dB SNR. Augmented waveforms exist only in worker memory.

## Fixed Controls

- untouched Nemotron base;
- `SURFACE_08_FULL_ENCODER`;
- `sl-corpus-v5-scale8000-training-v1`;
- FP32 with TF32 disabled;
- physical microbatch 2, accumulation 4, effective batch 8;
- at most nine rounds and 144,000 virtual exposures;
- ARTUR controller-dev-only checkpoint selection;
- post-selection directional Piper, Supertonic, FLEURS-v2, and ARTUR-J.

## Non-Goals

No real-speech training, S6TTS, scale-2000 training, new text, external noise
corpus, pre-rendered augmentation, model-surface change, prior-checkpoint
initialization, immutable-gate selection, checkpoint acceptance,
`TRAINING_ELIGIBLE`, or publication is authorized.
