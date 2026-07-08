# Work Order 0032: Scale-2000 Decoder+Joint RNNT with ARTUR Controller-Dev Early Stopping

Status: in progress

Branch: `exp/scale2000-decoder-joint-rnnt-artur-earlystop`

This work order reruns the scale-2000 decoder+joint RNNT diagnostic from the
untouched Nemotron base while retaining per-round ignored checkpoints and using
`artur-controller-dev-v1` for aggregate run-control under ADR 0008.

## Governed Use

`artur-controller-dev-v1` may be used only for aggregate per-round WER, CER,
empty hypothesis counts, edit rates, and the predeclared earliest-within-
tolerance checkpoint selection rule. It must not be used for training,
gradient updates, synthetic prompt construction, hard-example mining from raw
references or hypotheses, immutable-gate acceptance, or model release claims.

Immutable FLEURS-v2, immutable ARTUR-J, and any final blind test remain
unavailable for early stopping, checkpoint selection, hyperparameter
selection, prompt construction, or training.

## Fixed Training Protocol

- Base: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Training source: scale-2000 augmented corpus v4
- Semantic rows: 16000
- Exposure records: 320000
- Objective: audio-conditioned RNNT loss only
- Trainable surface: `model.decoder` and `model.joint` only
- Frozen surfaces: encoder, preprocessor, prompt pathway, tokenizer, adapters
- Optimizer: AdamW
- Learning rate: 0.001
- Effective batch size: 8
- Max rounds: 20
- Max optimizer steps: 40000
- Precision: FP32
- TF32: disabled

## Run-Control Rule

Evaluate `artur-controller-dev-v1` after each completed exposure round using
batch size 1, no duration bucketing, FP32, TF32 disabled, target `sl-SI`, and
attention context `[56, 3]`.

The selected round is the earliest available checkpoint within 0.50 absolute
WER points and 0.25 absolute CER points of the best controller-dev checkpoint,
with empty hypotheses no greater than the untouched-base empty count.

## Safety

No checkpoint, prediction, raw transcript, raw audio, local manifest, CSV/TSV
monitor output, or local absolute path may be committed. `accepted_parent`
remains `none`; `TRAINING_ELIGIBLE` must not be issued.
