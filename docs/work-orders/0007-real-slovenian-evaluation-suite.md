# Work Order 0007: Real Slovenian Evaluation Suite

## Goal

Create and execute the first immutable, non-synthetic Slovenian ASR evaluation
suite without training or fine-tuning the model.

## Required branch and metadata

- Branch: `feat/real-slovenian-evaluation-suite`
- Commit: `feat: add real Slovenian evaluation suite`
- Pull-request title: `feat: add real Slovenian evaluation suite`
- Do not include tool branding in Git or GitHub metadata.

## Inputs

- Current main includes `09edbb3e67e130609317fb70dbfdb63262b3bb73`.
- Accepted model remains the untouched `nvidia/nemotron-3.5-asr-streaming-0.6b`
  checkpoint at revision `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`.
- Checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.

## Required gates

1. Complete official FLEURS Slovenian test split:
   - `google/fleurs`
   - configuration `sl_si`
   - split `test`
   - revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
   - license CC BY 4.0
   - gate identifier `fleurs-sl-si-test-full-v1`
2. Deterministic ARTUR-J public-speech gate:
   - transcription handle `11356/1772`
   - audio handle `11356/1776`
   - license CC BY-SA 4.0
   - use only `Artur-J-Splosni` standardized orthographic `std`
   - exclude `Artur-B-Studio`, `Artur-B-Izloceno`, `Artur-N`, `Artur-P`, and
     `pog`
   - target 256 utterances, maximum four per source recording
   - gate identifier `artur-j-public-gate-v1`

## Evaluation policy

- Gates are immutable development gates, not final blind tests.
- Neither gate may enter training.
- Raw gate references must not be sent to GaMS or any synthetic-data generator.
- Commit only privacy-safe metadata, IDs, hashes, aggregate statistics, and
  checksums.
- Keep audio, archives, local manifests, raw references, raw hypotheses, and
  per-sample outputs ignored.

## Baseline evaluation

Evaluate only the untouched accepted base checkpoint with:

- `target_lang=sl-SI`
- `att_context_size=[56,3]`
- batch size 1
- physical GPU 0 selected with `CUDA_VISIBLE_DEVICES=0`

Report raw and normalized corpus WER/CER, mean and median utterance WER/CER,
empty-hypothesis counts, wall time, real-time factor, peak VRAM where measured,
and GPU 1 non-use.

## Non-goals

- no training or fine-tuning;
- no GaMS or Piper execution;
- no synthetic corpus generation;
- no model acceptance, publication, or release;
- no GPU 1 or A100 use;
- no CI redesign, service API, or database.
