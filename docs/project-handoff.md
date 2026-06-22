# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project has completed M1 runtime contract and one-RTX-2080-Ti baseline smoke verification.
- M2 Piper Slovenian TTS vertical slice is complete, while scalable generated
  data governance remains pending.
- M3 prompt-column micro-proof is complete for one tiny synthetic experiment.
  The result supports the prompt-column mechanism on synthetic smoke data but
  does not establish an accepted release parent.
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
- First M3 trainable surface: one additive `sl-SI` prompt-column delta with
  2048 effective trainable scalars, later merged into only the selected first
  prompt-projection column.
- M3 prompt-column micro-result: Phase A supported, Phase B executed, synthetic
  training WER improved from 92.5 to 38.333 and empty synthetic-training
  hypotheses dropped from 3 to 0. Synthetic holdout WER was unchanged at 87.5.
  Public FLEURS smoke WER regressed from 75.0 to 85.0.

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

Continue M2b with manifest validation, leakage controls, protected-evaluation
deduplication, and larger synthetic-data governance as separate bounded work
orders. Do not treat the prompt-column micro-checkpoint as an accepted parent or
publishable artifact.

## Do not do next

- Do not add GaMS orchestration yet.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.
- Do not escalate beyond the prompt column without a new work order and real
  evidence.

## Strategic questions after the next PR

- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
