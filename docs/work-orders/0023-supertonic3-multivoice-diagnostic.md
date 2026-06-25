# Work Order 0023: Supertonic 3 Multi-voice Diagnostic

Status: in progress

## Scope

Integrate pinned Supertonic 3 synthesis as a second local Slovenian TTS backend,
create a governed multi-voice synthetic audio bank from the existing corpus-v2
selected-training and holdout text partitions, and run one fixed frozen-base
RNNT joint-adapter diagnostic.

## Required sequence

1. Commit pinned Supertonic 3 integration, provenance, synthesis tooling,
   acoustic validation support, live-progress integration, and CPU tests.
2. Generate and validate local Supertonic audio, then commit only privacy-safe
   aggregate evidence and `DIAGNOSTIC_ONLY` authorization.
3. Run one fixed joint-adapter experiment and commit only aggregate evidence.

## Boundaries

- Do not regenerate or alter text.
- Do not run GaMS or Piper.
- Do not mix Piper and Supertonic audio in training.
- Do not train on held-out Supertonic styles `M5` or `F5`.
- Do not train prompt-column, encoder, decoder, prompt-kernel, tokenizer, or
  RNNT joint base parameters.
- Do not issue `TRAINING_ELIGIBLE`.
- Do not accept or publish an adapter or checkpoint.
- Do not commit generated audio, raw manifests, predictions, progress logs,
  monitor CSVs, model weights, or adapter weights.
- Do not merge the pull request.

## Fixed design

- Supertonic package: `supertonic==1.3.1`.
- Supertonic execution override: the human explicitly changed the original
  CPU-only TTS policy on 2026-06-25 and required Supertonic synthesis to use
  GPU execution. The integration therefore requires `onnxruntime-gpu`, physical
  GPU selector `1`, and `CUDAExecutionProvider` as the primary ONNX Runtime
  provider for Supertonic synthesis.
- Supertonic model: `Supertone/supertonic-3`.
- Model revision: `724fb5abbf5502583fb520898d45929e62f02c0b`.
- Training styles: `M1`, `M2`, `M3`, `M4`, `F1`, `F2`, `F3`, `F4`.
- Held-out styles: `M5`, `F5`.
- Training rows: 160 selected-training texts.
- Holdout rows: 96 independent synthetic holdout texts.
- Training arm: `supertonic3_multivoice_joint_adapter_dim32`.
- Batch size: 8.
- Epochs: 12.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Status: `DIAGNOSTIC_ONLY`.

## Acceptance

The PR is ready for strategic review only when pinned Supertonic assets are
verified, exactly 1280 training and 192 held-out final WAVs are generated and
validated locally, authorization predates Nemotron loading, one fixed
joint-adapter arm completes, both synthetic engines and both real gates are
evaluated, one scientific classification is issued, `accepted_parent` remains
`none`, required checks pass, and the PR remains unmerged.
