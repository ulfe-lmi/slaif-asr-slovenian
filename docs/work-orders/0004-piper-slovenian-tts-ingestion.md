# Work Order 0004: Piper Slovenian TTS Ingestion

## Goal

Begin M2 with a real GPU-validated Piper-to-Nemotron vertical slice:

```text
validated Slovenian text
    -> Piper sl_SI-artur-medium on GPU 0
    -> native 22,050 Hz WAV
    -> deterministic mono 16 kHz PCM WAV
    -> provenance record
    -> NeMo ASR manifest
    -> Nemotron transcription on GPU 0
    -> machine-readable smoke results
```

This work order does not train Nemotron.

## Required branch and metadata

- Branch: `feat/m2-piper-tts-ingestion`
- Commit: `feat: add Piper Slovenian TTS ingestion`
- Pull-request title: `feat: add Piper Slovenian TTS ingestion`
- Do not include agent or tool branding in Git or GitHub metadata.

## Inputs

Piper:

- Repository: `https://github.com/OHF-Voice/piper1-gpl`
- Revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- License: GPL-3.0-or-later

Voice:

- Repository: `rhasspy/piper-voices`
- Revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Voice: `sl_SI-artur-medium`
- Native sample rate: 22,050 Hz

## Required implementation

- Add ADR 0004 for the GPL external-executable boundary.
- Add third-party license and attribution documentation.
- Create `.venv-piper` setup for pinned Piper plus ONNX Runtime GPU.
- Keep Piper source, voice artifacts, generated WAVs, logs, and manifests ignored.
- Download the pinned voice files and verify exact byte sizes and SHA256 values.
- Add a minimal versioned synthetic-smoke candidate JSONL schema and fixture.
- Render candidates with `python -m piper --cuda --debug` through argv only.
- Fail if CUDA provider creation fails or Piper falls back to CPU.
- Convert native 22,050 Hz output to mono 16 kHz signed 16-bit PCM WAV with
  deterministic parameters.
- Write ignored provenance JSONL and deterministic NeMo manifest JSONL.
- Run Nemotron smoke transcription on the generated manifest at `[56,3]`.
- Verify physical GPU 0 is used and GPU 1 remains unused.

## Non-goals

- no Nemotron training or fine-tuning;
- no GaMS generation;
- no large corpus generation;
- no active learning;
- no public synthetic-audio release;
- no model or dataset publication;
- no Piper source or binary committed;
- no GPU 1 use;
- no A100 use;
- no service API, UI, or database.

## Acceptance criteria

- Piper source and voice revisions are pinned.
- License records and attribution are complete.
- `.venv-piper` builds reproducibly.
- ONNX Runtime `CUDAExecutionProvider` is available.
- Real Piper uses GPU 0 without CPU fallback.
- The selected voice downloads and verifies by SHA256.
- All smoke candidates render successfully.
- Native 22,050 Hz files and final mono 16 kHz PCM files validate.
- Provenance and NeMo manifests are complete and deterministic.
- Nemotron transcribes the generated batch on GPU 0.
- GPU 1 remains unused.
- No model, voice, audio, result, private path, or secret is committed.
- M2 remains in progress.
