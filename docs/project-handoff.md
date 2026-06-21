# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project is at M0 strategic scaffold.
- No executable code has been accepted.
- No model weights, datasets, or benchmark results are part of the repository.
- Selected first base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Selected framework: NVIDIA NeMo.
- Slovenian locale/prompt: `sl-SI`.
- Planned active loop: GaMS -> Slovenian TTS -> current-model failure selection -> bounded training -> acceptance/rollback.
- GitHub is for method and evidence; Hugging Face will be used for model artifacts.

## Non-negotiable rules

- Do not fork NeMo.
- Do not commit model/data artifacts.
- Do not expose private speech or transcripts.
- Do not replace the tokenizer without an ADR.
- Do not escalate trainable scope silently.
- Do not make performance claims before a committed evaluation protocol is executed.
- Do not publish or merge without human approval.

## Next recommended task

Execute:

[`work-orders/0001-runtime-contract-and-baseline-inference.md`](work-orders/0001-runtime-contract-and-baseline-inference.md)

The task should produce one PR with pinned setup, checkpoint inspection, Slovenian tokenizer audit, and baseline cache-aware streaming inference.

## Do not do next

- Do not implement training before the runtime contract is verified.
- Do not add GaMS/TTS orchestration yet.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.

## Strategic questions after the next PR

- Does the loaded checkpoint expose the expected `sl-SI` prompt?
- Does the tokenizer preserve Slovenian characters and desired transcript style?
- Are all supported streaming contexts reproducible?
- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
