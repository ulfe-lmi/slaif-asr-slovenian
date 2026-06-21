# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project has completed M1 runtime contract and one-RTX-2080-Ti baseline smoke verification.
- M2 data and TTS ingestion is in progress through the Piper Slovenian TTS
  vertical slice.
- The repository has a CPU-only GitHub Actions baseline for tracked-file hygiene,
  unit tests, Python compilation, and shell syntax. This CI does not install
  NeMo, download checkpoints or audio, use GPUs, or prove model restoration.
- Executable baseline helpers exist for official-checkpoint download, runtime inspection, tokenizer audit, and forced `sl-SI` cache-aware streaming inference.
- No model weights, datasets, or benchmark results are part of the repository.
- Selected first base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Selected framework: NVIDIA NeMo.
- Slovenian locale/prompt: `sl-SI`.
- Planned active loop: GaMS -> Slovenian TTS -> current-model failure selection -> bounded training -> acceptance/rollback.
- Selected initial TTS engine: external `OHF-Voice/piper1-gpl`.
- Selected initial TTS voice: `rhasspy/piper-voices` `sl_SI-artur-medium`.
- GitHub is for method and evidence; Hugging Face will be used for model artifacts.
- Pinned model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`.
- Pinned NeMo revision for the baseline interface: `8044a3924bfcfe8ef71d792bb73bf274fe853575`.
- Correct checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
- Current development hardware: 48 GB class RAM, 2 x RTX 2080 Ti with 11 GB each, one GPU used per process.
- Default runtime selection: `CUDA_VISIBLE_DEVICES=0`; the second RTX 2080 Ti remains unused unless a later work order explicitly permits it.
- M1 and M2 use one RTX 2080 Ti. The first prompt-specific M3 proof should attempt one RTX 2080 Ti before requesting stronger hardware.
- A100 is not a default prerequisite and becomes mandatory only through a later work order backed by measured memory, throughput, or authoritative benchmarking requirements.

## Non-negotiable rules

- Do not fork NeMo.
- Do not commit model/data artifacts.
- Do not expose private speech or transcripts.
- Do not replace the tokenizer without an ADR.
- Do not escalate trainable scope silently.
- Do not make performance claims before a committed evaluation protocol is executed.
- Do not publish or merge without human approval.

## Current runtime commands

See:

[`baseline-inference.md`](baseline-inference.md)

The baseline commands require the repository-local `.venv` and a disposable GPU environment before checkpoint loading and streaming inference can be represented as passed.

## Current CPU validation

The durable pull-request baseline is:

```text
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile <all tracked Python files>
.venv/bin/python scripts/check_repository.py
bash -n scripts/*.sh
git diff --check
```

GPU verification remains separate manual or future self-hosted evidence. The M1
GPU evidence comes from the RTX 2080 Ti verification work order and should not
be inferred from CPU CI.

## Current M2 TTS validation

The Piper TTS slice uses a separate local `.venv-piper` environment and ignored
voice/audio storage. The executable path is:

```text
scripts/setup_piper_tts_env.sh
.venv/bin/python scripts/download_piper_sl_voice.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/render_piper_candidates.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_streaming_inference.py --manifest runs/tts/piper/nemo-manifest.jsonl --context '[56,3]' --batch-size 1 --cuda 0 --output-dir runs/tts/piper/asr-smoke
```

The rendered smoke audio, provenance, manifest, ASR logs, and result files remain
ignored local evidence. This proves a real TTS-to-ASR vertical slice only; it is
not a benchmark and does not start training.

## Next recommended task

After the Piper TTS ingestion PR is reviewed and merged, continue M2 with
manifest validation, leakage controls, and larger synthetic-data governance as
separate bounded work orders. Do not start training before an approved M3 work
order.

## Do not do next

- Do not implement training before the runtime contract is verified.
- Do not add GaMS orchestration yet.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.

## Strategic questions after the next PR

- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
