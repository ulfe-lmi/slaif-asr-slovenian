# Work Order 0029: Text-Only Slovenian Decoder-LM Adaptation

Status: implemented as `DIAGNOSTIC_ONLY`.

This work order tests whether the accepted scale-8000 Slovenian GaMS text can
adapt the RNNT prediction-decoder language side without audio training.

The intended trainable surface is one decoder-side residual bottleneck adapter
named `sl-si-decoder-lm-adapter-v1` plus a temporary next-token LM head used
only for text training. The encoder, prompt kernel, RNNT decoder base, RNNT
joint base, tokenizer, and all other pretrained tensors remain frozen.

The data source is only `sl-corpus-v5-scale8000-training-v1`:

- rows: 64,000
- text SHA256: `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd`
- decision: `ACCEPT`
- decision ID: `human-scale8000-decision-v1`

No TTS, synthetic audio training, augmentation, fake audio, dummy encoder-state
training, real-gate transcript use, tokenizer replacement, model publication,
or `TRAINING_ELIGIBLE` decision is authorized.

ASR evaluation is directional batch-32 evidence only. It is not canonical
batch-1 evidence and cannot promote a model.
