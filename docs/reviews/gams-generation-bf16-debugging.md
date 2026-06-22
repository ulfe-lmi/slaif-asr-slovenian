# GaMS Generation BF16 Debugging Report

Date: 2026-06-22

## Scope

This report records the local diagnosis of unusable GaMS3 text generation on
the A100 runtime. The task did not train Nemotron, run Piper, publish generated
text, or accept any challenger checkpoint.

## Environment

- Host class: NVIDIA A100-SXM4-80GB runtime
- Authorized model device for the diagnosis: one visible GPU selected with
  `CUDA_VISIBLE_DEVICES=1`
- GaMS environment: repository-local `.venv-gams`
- Primary model: `cjvt/GaMS3-12B-Instruct`
- Primary revision: `1d0b27af5748784482600d24779409e7e1dc9adc`
- Transformers version after repair: `4.55.2`
- PyTorch version: `2.7.1+cu126`

## Symptom

The command-line GaMS probe produced multilingual garbage such as unrelated
tokens from many languages and scripts. A related diagnostic also produced only
padding tokens.

Transformers also printed warnings that `temperature`, `top_p`, and `top_k`
were invalid for greedy generation. Those warnings were real but were not the
primary cause of the unusable text. They come from the upstream
`generation_config.json`, which contains sampling-only fields while
`do_sample` defaults to false.

## Root Cause

The bad generation path had three concrete problems:

1. The committed GaMS config and generator forced 4-bit NF4 with FP16 compute.
   The GaMS3 model config declares BF16, and local A100 diagnostics showed that
   FP16 compute can collapse generation.
2. The generator did not pass an `attention_mask`, which is unreliable for
   Gemma-family tokenizers when padding and end-of-turn tokens are involved.
3. The generator used `tokenizer.eos_token_id` as `pad_token_id` even though the
   tokenizer and model config declare `pad_token_id=0`.

The decisive comparison used the same model, revision, prompt, GPU, tokenizer,
and Transformers version:

| Mode | Result | Peak VRAM |
|---|---|---:|
| Full BF16 | coherent Slovenian output | 22598.4 MiB |
| 4-bit NF4 with BF16 compute | coherent Slovenian output | 11221.0 MiB |
| 4-bit NF4 with FP16 compute | repeated `<pad>` tokens | 11221.0 MiB |

## Repairs

- Pin GaMS generator dependencies to the model-compatible Transformers stack:
  `transformers==4.55.2` and `huggingface-hub==0.36.2`.
- Change GaMS 4-bit compute dtype from FP16 to BF16.
- Pass `torch_dtype` consistently when loading the quantized model.
- Pass an explicit `attention_mask` to `generate`.
- Use the configured tokenizer/model pad token, falling back only when absent.
- Sanitize inherited sampling-only generation fields before explicit
  generation kwargs are applied.
- Make the `.venv-gams` setup verification respect the operator-selected
  single visible GPU instead of hard-coding physical GPU 0.
- Add a direct CLI probe that prints raw generated text to stdout and metadata
  to stderr or a JSON file.

## Working Commands

4-bit BF16 compute is the preferred memory-efficient A100 path:

```bash
TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
.venv-gams/bin/python scripts/test_gams_cli.py \
  --bnb-compute-dtype bfloat16 \
  --max-memory-gib 76 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  "Napiši deset naravnih slovenskih stavkov, vsak v svoji vrstici."
```

Full BF16 is the highest-stability diagnostic path on an A100:

```bash
TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
.venv-gams/bin/python scripts/test_gams_cli.py \
  --bf16 \
  --max-memory-gib 76 \
  --max-new-tokens 256 \
  "Napiši deset naravnih slovenskih stavkov, vsak v svoji vrstici."
```

## Remaining Limits

- This report proves only that GaMS3 generation can be made coherent on the
  local A100 runtime.
- It does not validate the generated sentences as a curriculum.
- It does not run Piper synthesis, Nemotron pre-scoring, prompt-column
  training, or real-gate evaluation.
- Generated text remains a local artifact until a later work order authorizes
  a curriculum run and records privacy-safe aggregate evidence.
