# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project is at M1 runtime contract and baseline inference tooling.
- Executable baseline helpers exist for official-checkpoint download, runtime inspection, tokenizer audit, and forced `sl-SI` cache-aware streaming inference.
- No model weights, datasets, or benchmark results are part of the repository.
- Selected first base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Selected framework: NVIDIA NeMo.
- Slovenian locale/prompt: `sl-SI`.
- Planned active loop: GaMS -> Slovenian TTS -> current-model failure selection -> bounded training -> acceptance/rollback.
- GitHub is for method and evidence; Hugging Face will be used for model artifacts.
- Pinned model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`.
- Pinned NeMo revision for the baseline interface: `8044a3924bfcfe8ef71d792bb73bf274fe853575`.

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

The baseline commands require a disposable GPU environment before checkpoint loading and streaming inference can be represented as passed.

## Next recommended task

After the runtime baseline PR is reviewed and merged, execute the next approved work order for manifest and audio validation. Do not start training before runtime contract evidence exists.

## Do not do next

- Do not implement training before the runtime contract is verified.
- Do not add GaMS/TTS orchestration yet.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.

## Strategic questions after the next PR

- Does the generated runtime contract expose the expected `sl-SI` prompt index in the actual GPU environment?
- Does the tokenizer audit pass exact Slovenian round trips with the downloaded checkpoint?
- Do all supported streaming contexts run successfully on A100?
- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
