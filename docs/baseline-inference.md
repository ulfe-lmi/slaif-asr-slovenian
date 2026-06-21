# Baseline Runtime Contract and Inference

This document defines the M1 baseline path for the SLAIF Slovenian adaptation of NVIDIA Nemotron 3.5 ASR Streaming.

It does not fine-tune, publish, or evaluate Slovenian quality. It downloads the official checkpoint into ignored local storage, records the loaded runtime contract, audits Slovenian tokenizer round trips, and runs cache-aware streaming inference with `target_lang=sl-SI`.

## Verified upstream interfaces

Verified on 2026-06-21:

- Official model repository: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- Pinned model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Pinned checkpoint file: `nemotron-3.5-asr-streaming-0.6b.nemo`
- Checkpoint SHA256: `79766c070eed987b43ee595fff7bd21fe49aae6ee26e881f51b86d8e662e713d`
- NeMo source: [`NVIDIA-NeMo/NeMo`](https://github.com/NVIDIA-NeMo/NeMo)
- Pinned NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`

The official model card documents cache-aware streaming through NeMo, prompt conditioning through `target_lang`, and the five released latency settings:

```text
[56,0], [56,1], [56,3], [56,6], [56,13]
```

The latest NeMo release tag inspected for this PR, `v2.7.3`, does not expose `target_lang` or `strip_lang_tags` in the cache-aware streaming example. The implementation therefore pins the later verified NeMo commit above instead of using an unpinned `main` branch.

## Local artifact policy

The following paths are ignored and must remain local:

```text
.external/NeMo/
models/checkpoints/
runs/contracts/
runs/tokenizer-audits/
runs/inference/
```

Do not commit downloaded `.nemo` files, audio files, per-utterance private outputs, or local absolute paths.

## Runtime setup

Use a disposable GPU environment. The target smoke-test hardware is an NVIDIA A100 with the CUDA/PyTorch stack supplied by the pinned NeMo 26.06 runtime or by installing the pinned NeMo source checkout.

```bash
scripts/setup_runtime_env.sh
source .venv/bin/activate
```

The setup script checks out NeMo at:

```text
8044a3924bfcfe8ef71d792bb73bf274fe853575
```

Generated runtime contracts record exact Python, PyTorch, CUDA, GPU, and NeMo versions for each run.

## Download checkpoint

Dry-run the official URL and metadata:

```bash
python3 scripts/download_nemotron_checkpoint.py --dry-run
```

Download and verify the checkpoint:

```bash
python3 scripts/download_nemotron_checkpoint.py
```

The checkpoint is written under `models/checkpoints/`, and a `.sha256` sidecar is written next to it. Both are ignored by Git.

## Inspect runtime contract

```bash
python3 scripts/inspect_runtime_contract.py \
  --checkpoint models/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output runs/contracts/nemotron-3.5-asr-streaming-0.6b.json
```

The JSON contract includes loaded class, parameter count, encoder shape, tokenizer vocabulary size, sample rate, `sl-SI` and `sl` prompt indices when introspectable, prompt-related parameter shapes, streaming contexts, checkpoint identity, and environment details.

## Audit Slovenian tokenizer behavior

```bash
python3 scripts/audit_slovenian_tokenizer.py \
  --checkpoint models/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output runs/tokenizer-audits/sl-si.json
```

The audit records token IDs, decoded text, and an exact round-trip pass/fail result for representative Slovenian text containing `č`, `š`, `ž`, uppercase letters, punctuation, dates, decimal commas, euro signs, and degree symbols.

## Prepare audio

Input audio for the wrappers must be mono 16 kHz WAV. Convert local audio with:

```bash
ffmpeg -i input-audio-file -ac 1 -ar 16000 -sample_fmt s16 private-slovenian-16k.wav
```

Keep converted audio outside Git-tracked paths or under ignored local paths.

## Single-file streaming inference

Run one context:

```bash
scripts/infer_sl_si_56_13.sh \
  --audio-file private-slovenian-16k.wav \
  --cuda 0
```

For single-file mode, the local wrapper writes a temporary one-entry manifest under `runs/inference/` and invokes the upstream manifest path. This avoids requiring a private reference transcript and keeps all generated input/output metadata in ignored storage.

Run all five contexts:

```bash
python3 scripts/run_streaming_inference.py \
  --audio-file private-slovenian-16k.wav \
  --all-contexts \
  --cuda 0
```

The wrapper passes:

```text
target_lang=sl-SI
strip_lang_tags=true
```

to the pinned NeMo cache-aware streaming script.

## Manifest streaming inference

Manifest entries follow [`docs/examples/inference-manifest.schema.json`](examples/inference-manifest.schema.json). A text-only example is provided in [`docs/examples/slovenian-inference-manifest.example.jsonl`](examples/slovenian-inference-manifest.example.jsonl).

Run a manifest at all five settings:

```bash
python3 scripts/run_streaming_inference.py \
  --manifest path/to/private-manifest.jsonl \
  --all-contexts \
  --batch-size 1 \
  --cuda 0
```

NeMo writes manifest outputs under `runs/inference/`, which is ignored. Treat per-utterance outputs as private when the input audio or reference text is private.

## Known limitations

- No checkpoint is committed.
- No audio, dataset, or benchmark output is committed.
- No fine-tuning, GaMS integration, TTS integration, active learning, model publication, or release step is implemented.
- CPU-only tests validate serialization, tokenizer-audit handling, config syntax, and wrapper syntax. They do not prove checkpoint loading or ASR quality.
- GPU smoke inference is required before any runtime-quality claim and must report hardware, CUDA, PyTorch, NeMo revision, checkpoint revision, context setting, command, result, and peak memory when available.
