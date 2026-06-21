# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project has completed M1 runtime contract and one-RTX-2080-Ti baseline smoke verification.
- Executable baseline helpers exist for official-checkpoint download, runtime inspection, tokenizer audit, and forced `sl-SI` cache-aware streaming inference.
- No model weights, datasets, or benchmark results are part of the repository.
- Selected first base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Selected framework: NVIDIA NeMo.
- Slovenian locale/prompt: `sl-SI`.
- Planned active loop: GaMS -> Slovenian TTS -> current-model failure selection -> bounded training -> acceptance/rollback.
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

## Next recommended task

After the M1 repair-and-verification PR is reviewed and merged, execute the next approved work order for manifest and audio validation. Do not start training before that next approved work order.

## Do not do next

- Do not implement training before the runtime contract is verified.
- Do not add GaMS/TTS orchestration yet.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.

## Strategic questions after the next PR

- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
