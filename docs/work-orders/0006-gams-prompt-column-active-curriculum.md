# Work Order 0006: GaMS-Directed Prompt-Column Active Curriculum

## Goal

Test whether the proven 2,048-parameter Slovenian prompt-column adaptation can
generalize when driven by two bounded GaMS -> Piper -> Nemotron active-learning
rounds.

This work order does not broaden the trainable surface. It trains only the
`sl-SI` prompt-column delta and preserves the frozen encoder, decoder, joint,
tokenizer, prompt kernel outside the selected column, and all non-Slovenian
prompt parameters.

## Required branch and metadata

- Branch: `feat/prompt-column-active-curriculum`
- Commit: `feat: add GaMS-directed prompt-column curriculum`
- Pull-request title: `feat: add GaMS-directed prompt-column curriculum`
- Do not include tool branding in Git or GitHub metadata.

## Inputs

- Current main includes `70c2d93bb511064fa09754397cc845ea491be2ed`.
- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Base revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`
- Checkpoint SHA256:
  `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
- NeMo revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Piper revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Piper voice revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- GaMS3 revision: `1d0b27af5748784482600d24779409e7e1dc9adc`
- GaMS-9B fallback revision: `292744023fa0b7ccc7ae2c3c885a67468e49fa03`

## Hardware and environments

- Use physical GPU 0 only with `CUDA_VISIBLE_DEVICES=0`.
- GPU 1 remains unused.
- Use `.venv` for Nemotron, `.venv-piper` for Piper, and `.venv-gams` for GaMS.
- Run GaMS, Piper, and Nemotron sequentially so model memory is released between
  phases.

## Protocol

1. Make the Nemotron training runtime reproducible with pinned CUDA 12.6
   PyTorch, Numba, llvmlite, and NVCC/NVVM.
2. Build a fixed 64-utterance FLEURS Slovenian real gate from the pinned dataset
   revision. Real gate audio and manifests remain ignored.
3. Generate a fixed 64-item synthetic holdout before training. It is never used
   for training or later steering.
4. Generate 128 round-1 candidates with strict JSON GaMS output.
5. Synthesize valid candidates through Piper on GPU 0.
6. Pre-score candidates with the untouched Nemotron checkpoint at `[56,3]`.
7. Select 48 hard examples and 16 seeded general controls.
8. Train only the `sl-SI` prompt-column delta for at most 1500 FP32 steps.
9. Merge, restore, and run parameter-integrity checks.
10. Promote or roll back round 1 using fixed synthetic and real gates.
11. Build a round-2 brief from synthetic candidate-pool failures without real
    gate text.
12. Generate, synthesize, pre-score, select, train, and evaluate round 2.
13. Conclude with exactly one of:
    `PROMPT_COLUMN_GENERALIZATION_SUPPORTED`,
    `PROMPT_COLUMN_SYNTHETIC_ONLY`,
    `PROMPT_COLUMN_SCALING_NOT_SUPPORTED`, or `EXPERIMENT_INVALID`.

## Promotion gates

Round promotion requires:

- state-dictionary integrity passes;
- synthetic holdout corpus WER improves by at least 15% relative, or synthetic
  holdout corpus CER improves by at least 15% relative;
- real-gate corpus WER does not regress by more than 1.0 absolute point;
- real-gate corpus CER does not regress by more than 1.5 absolute points;
- real-gate empty-hypothesis count does not increase;
- no execution or data-integrity failure occurs.

Training-set improvement alone never permits promotion.

## Non-goals

- no prompt-kernel-wide, decoder, joint, encoder, tokenizer, LoRA, or real-speech
  training;
- no synthetic holdout steering;
- no real-gate text exposure to GaMS;
- no more than two active rounds;
- no GPU 1 or A100 use;
- no model, delta, generated data, audio, checkpoint, or dataset publication;
- no service API, database, or CI redesign.

## Required evidence

- local unit and repository checks;
- `.venv`, `.venv-piper`, and `.venv-gams` `pip check`;
- Nemotron RNNT CUDA loss smoke;
- GaMS generation metadata and validity rate;
- Piper synthesis metadata;
- Nemotron pre-scoring, training, evaluation, and integrity reports;
- GPU 0 peak memory and GPU 1 idle evidence for every model phase;
- aggregate corpus and mean utterance WER/CER kept separate.
