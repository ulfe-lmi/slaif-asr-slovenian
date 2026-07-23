# Work Order 0039: Fixed-Data Surface06 Last Four Encoder Blocks

## Scope

Run one fixed-data boundary diagnostic:
`SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS`.

The only scientific change from Surface05 is the trainable encoder depth and
its predeclared learning rate. Train decoder, joint, and exactly
`encoder.layers.20` through `encoder.layers.23` from the untouched Nemotron
base. Use the original scale-2000 augmented corpus v4, fixed exposure schedule,
ARTUR controller-dev run-control, and post-selection directional suite.

## Controls

- Audio-conditioned RNNT loss only.
- Decoder and joint learning rate: `5.0e-4`.
- Final-four encoder learning rate: `1.0e-5`.
- Effective batch size: 8.
- FP32 with TF32 disabled.
- Maximum 20 rounds and 40,000 optimizer steps.
- ARTUR controller-dev alone selects the checkpoint.

## Boundaries

No S6TTS, scale-8000, new text, real-speech training, immutable-gate
selection, fusion change, Surface07, checkpoint acceptance, model publication,
`TRAINING_ELIGIBLE`, or `accepted_parent` change is authorized. Runtime
checkpoints, predictions, audio, manifests, logs, and local paths remain
ignored and uncommitted.
