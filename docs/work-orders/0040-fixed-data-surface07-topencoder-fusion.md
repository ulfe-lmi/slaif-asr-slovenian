# Work Order 0040: Fixed-Data Surface07 Top Encoder Plus Fusion

## Scope

Run one fixed-data fusion-bottleneck diagnostic:
`SURFACE_07_TOP_ENCODER_PLUS_PROMPT_ACOUSTIC_FUSION`.

Train decoder, joint, exactly `encoder.layers.20` through
`encoder.layers.23`, and exactly one proven separable prompt/acoustic fusion
bridge from the untouched Nemotron base. The pinned live model identifies that
bridge as `prompt_kernel`, a post-concatenation `1152 -> 2048 -> 1024` MLP.
It contains no prompt embedding, table, label, tokenizer, or language-ID
mapping parameter.

## Controls

- Original scale-2000 augmented corpus v4 and fixed exposure schedule only.
- Audio-conditioned RNNT loss only.
- Decoder and joint learning rate: `5.0e-4`.
- Final-four encoder learning rate: `1.0e-5`.
- Fusion bridge learning rate: `5.0e-5`.
- Effective batch size: 8.
- FP32 with TF32 disabled.
- Maximum 20 rounds and 40,000 optimizer steps.
- ARTUR controller-dev alone selects the checkpoint.
- Post-selection directional batch-32 evaluation remains noncanonical.

## Fail-Closed Discovery

Training may begin only if live model inspection proves:

- the final encoder blocks are `encoder.layers.20` through
  `encoder.layers.23`;
- `prompt_kernel` is the sole post-concatenation prompt/acoustic projection;
- its only parameters are the two linear layers and their biases;
- prompt identity remains a non-parameter one-hot mapping; and
- no protected prompt, language-ID, tokenizer, or target-language parameter
  overlaps the bridge.

Otherwise the result is `BLOCKED_FUSION_BRIDGE_UNRESOLVED`.

## Boundaries

No S6TTS, scale-8000, new text, real-speech training, immutable-gate
selection, lower encoder block, frontend, tokenizer, prompt-identity,
Surface08, full-encoder, checkpoint acceptance, model publication,
`TRAINING_ELIGIBLE`, or `accepted_parent` change is authorized. Runtime
checkpoints, predictions, audio, manifests, logs, and local paths remain
ignored and uncommitted.
