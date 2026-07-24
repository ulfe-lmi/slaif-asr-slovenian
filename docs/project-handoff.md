# Project Handoff

## Current truth

- Intended repository: `ulfe-lmi/slaif-asr-slovenian`.
- Project has completed M1 runtime contract and one-RTX-2080-Ti baseline smoke verification.
- M2 Piper Slovenian TTS vertical slice is complete, while scalable generated
  data governance remains pending.
- M3 prompt-column micro-proof is complete for one tiny synthetic experiment.
  The result supports the prompt-column mechanism on synthetic smoke data but
  does not establish an accepted release parent.
- M3 prompt-column generalization is now represented by a bounded two-round
  GaMS active-curriculum protocol. The tooling and configs are separate from
  completed GPU evidence until the experiment runs.
- The canonical FLEURS development gate is now `fleurs-sl-si-test-full-v2`,
  built from all 834 pinned Slovenian test occurrences with unique
  occurrence-index sample IDs. Historical FLEURS v1 metrics are deprecated
  because v1 represented only 347 unique sample identities. FLEURS v2 now has
  a valid untouched-base ASR baseline from Experiment 0006.
- The deterministic ARTUR-J public-speech gate remains valid and unaffected.
- ADR 0008 introduces `artur-controller-dev-v1` as a separate ARTUR
  controller-development partition for aggregate run-control. It is spent
  development data if used for checkpoint selection and must not be treated as
  an immutable acceptance gate.
- Project-generated Slovenian curriculum Round 1 has run without GaMS or an
  external LLM. Its prompt-column challenger is rejected: it improved selected
  synthetic training examples, did not meet the fixed synthetic-holdout
  promotion threshold, and regressed ARTUR-J. Its FLEURS-v1 component is
  deprecated. Later review found the Round 1 v1 corpus structurally repetitive,
  linguistically defective, and train/holdout template-confounded, so it is
  retired and must not be reused for training, steering, model comparison, or
  promotion.
- Slovenian residual-adapter proof has run on one A100 logical GPU using the
  exact Round 1 corpus and fixed real gates. Rank 16 and rank 64 adapters
  improved the fixed synthetic holdout but regressed ARTUR-J, so the result is
  `SL_RESIDUAL_SYNTHETIC_ONLY` and no adapter is accepted as a parent. Its
  FLEURS-v1 component is deprecated. Because it reused the retired corpus, the
  experiment is corpus-confounded and must not be used as proof that residual
  adapters or added prompt-side capacity are intrinsically unsuitable.
- `docs/training-data-constitution.md` is adopted as the detailed companion to
  `AGENTS.md`. Promotion-oriented training now requires `TRAINING_ELIGIBLE`
  data and a privacy-safe acceptance certificate; schema validity and literal
  duplicate checks are not enough.
- The fail-closed text-stage corpus validator is implemented. It rejects the
  retired corpus identities before parsing, enforces structural fingerprints
  and protected-gate hash indexes, requires complete linguistic review, and can
  emit `TEXT_ACCEPTED`. It does not validate audio or issue
  `TRAINING_ELIGIBLE`.
- The first GaMS corpus-v2 candidate reservoir requested 480 candidates and
  retained 415 structurally admissible rows. A human whole-file `ACCEPT`
  decision bound to the exact corpus hash and row count reproduced
  `TEXT_ACCEPTED`; all 415 rows were then rendered through the external Piper
  boundary and waveform-validated as `AUDIO_ACCEPTED`. Raw generated text,
  audio, manifests, logs, and monitoring CSVs remain ignored. The reservoir is
  still not `TRAINING_ELIGIBLE`: the independent synthetic holdout now has text
  and audio admission, scoring has run on both synthetic partitions, and a
  selected-training manifest is ready, but there is no `TRAINING_ELIGIBLE`
  certificate or training authorization.
- The independent corpus-v2 synthetic holdout
  `sl-corpus-v2-independent-synthetic-holdout-v1` was generated with separately
  pinned `cjvt/GaMS-9B-Instruct`, fixed to 96 rows by deterministic 12-per-cell
  selection, and validated jointly against the accepted 415-row candidate
  source plus the FLEURS-v2 and ARTUR-J protected indexes. Its SHA256 is
  `078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5`. A
  whole-file human `ACCEPT` decision advanced it to `TEXT_ACCEPTED`; Piper
  synthesis and waveform validation advanced it to `AUDIO_ACCEPTED`.
- The candidate source and independent synthetic holdout have
  `SCORING_AUTHORIZED` evidence, and untouched-base ASR scoring has been run
  on both partitions under the batch-1 A100 policy. Selected-training
  construction selected 160 candidate-source rows and produced a
  `SELECTED_TRAINING_MANIFEST_READY` certificate. It does not authorize model
  training, checkpoint promotion, public performance claims, or
  `TRAINING_ELIGIBLE`.
- Work Order 0020 used that selected-training manifest once under a named
  `DIAGNOSTIC_ONLY` exception for prompt-column evidence. Both valid
  prompt-column arms trained only the 2,048-value `sl-SI` prompt-column delta
  and improved the synthetic holdout, but they failed real-gate
  non-regression. The result is
  `CORPUS_V2_PROMPT_COLUMN_SYNTHETIC_ONLY`; no checkpoint is accepted, and the
  untouched Nemotron checkpoint remains the only parent.
- The A100 prompt-column training benchmark selected batch size 8 for
  throughput, but the resulting batched arm was not scientifically equivalent
  to the batch-size-1 reference arm. Future training work should not assume
  minibatch equivalence without a new bounded proof.
- Work Order 0021 tested deterministic speaker-range resampling proxies as the
  only change relative to the clean batch-8 prompt-column arm. The synthetic
  holdout still improved, but real-gate regression was not prevented or
  mitigated. The result is `SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED`; no
  checkpoint is accepted.
- Work Order 0022 tested one frozen-base Slovenian RNNT joint-hidden adapter
  as the only model-surface change relative to the clean batch-8 protocol. The
  original clean Piper audio was used, every pretrained Nemotron tensor stayed
  frozen, shared live progress was emitted during long stages, and evaluation
  used batch size 1. The synthetic holdout improved, but real-gate regression
  increased; the result is `SL_JOINT_ADAPTER_SYNTHETIC_ONLY` and no adapter or
  checkpoint is accepted.
- Work Order 0023 tested the same frozen-base joint-adapter surface with
  Supertonic 3 preset multi-voice synthetic training audio. Eight styles
  trained the adapter, M5/F5 remained held out, Piper audio was not used for
  training, and evaluation used batch size 1. The real-gate regression burden
  dropped from 28.275 to 15.537, so the result is
  `SUPERTONIC3_MULTIVOICE_MITIGATES_PIPER_REGRESSION`; no adapter or
  checkpoint is accepted.
- ADR 0007 reframes the active development strategy as Slovenian-first, with
  Slovenian-English as the likely first bilingual extension. Real Slovenian
  acoustic data is validation-only and must not be used for training,
  synthetic prompt construction, selected-training membership, early stopping,
  hyperparameter tuning, per-sample steering, or adapter-surface selection.
  While training remains synthetic-only, the acoustic encoder stays frozen by
  default. The next model-surface experiments should focus on broader
  frozen-encoder emission adaptation, such as larger joint adapters, decoder
  adapters, joint-plus-decoder adapters, or frozen-encoder joint/decoder
  fine-tuning.
- A100 batched streaming evaluation has been measured on physical GPU 1 with
  FP32 and TF32 disabled. Batch sizes 2 through 128 were faster on FLEURS-v2
  but changed transcripts, so the selected policy is batch size 1 without
  duration bucketing. ARTUR-J confirms batch-1 parity. Corpus-v2 scoring used
  this batch-1 policy.
- The ignored M3 micro-proof checkpoint regressed on ARTUR-J and remains
  unaccepted. Its FLEURS-v1 component is deprecated.
- The repository has a CPU-only GitHub Actions baseline for tracked-file hygiene,
  unit tests, Python compilation, and shell syntax. This CI does not install
  NeMo, download checkpoints or audio, use GPUs, or prove model restoration.
- Executable baseline helpers exist for official-checkpoint download, runtime inspection, tokenizer audit, and forced `sl-SI` cache-aware streaming inference.
- No model weights or raw datasets are part of the repository. Only privacy-safe
  aggregate development-gate metadata and results are committed.
- Selected first base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Selected framework: NVIDIA NeMo.
- Slovenian locale/prompt: `sl-SI`.
- Planned active loop: GaMS -> Slovenian TTS -> current-model failure selection -> bounded training -> acceptance/rollback.
- Pinned primary GaMS generator: `cjvt/GaMS3-12B-Instruct` at
  `1d0b27af5748784482600d24779409e7e1dc9adc`.
- Pinned fallback GaMS generator: `cjvt/GaMS-9B-Instruct` at
  `292744023fa0b7ccc7ae2c3c885a67468e49fa03`, used only if the primary model
  cannot load or generate under the committed 4-bit BF16 policy.
- GaMS3 A100 diagnostic: the FP16-compute 4-bit path produced unusable output,
  while full BF16 and 4-bit NF4 with BF16 compute produced coherent Slovenian.
  The current `.venv-gams` policy therefore uses Transformers 4.55.2 and BF16
  compute. See
  [`gams-generation-bf16-debugging.md`](reviews/gams-generation-bf16-debugging.md).
- Selected initial TTS engine: external `OHF-Voice/piper1-gpl`.
- Selected initial TTS voice: `rhasspy/piper-voices` `sl_SI-artur-medium`.
- Additional governed TTS candidate under Work Order 0034: external lab-origin
  `ulfe-lmi/s6tts` at pinned revision
  `6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c`, voice label
  `s6tts-sl-si-s6-vintage`. S6TTS output is internal synthetic diagnostic
  audio only; public audio/model release is not authorized.
- Work Order 0035 applies the existing transcript-preserving augmentation
  policy to that admitted S6TTS clean view, producing an internal diagnostic
  176000-file S6TTS augmented bank for the fixed scale-2000 corpus. The bank
  is admitted as synthetic audio evidence only; it does not authorize training,
  `TRAINING_ELIGIBLE`, checkpoint acceptance, or public audio/model release.
- ADR 0009 and Work Order 0037 start a fixed-data trainable-surface sweep using
  only the original scale-2000 augmented corpus. Phase 1 trained decoder,
  joint, and exactly `encoder.layers.23`; ARTUR controller-dev selected round 3
  and the operational rule stopped at round 6 after 12,000 optimizer steps and
  96,000 exposures. The selected checkpoint scored 46.292/14.792 on FLEURS-v2
  and 55.920/18.535 on ARTUR-J with zero empty hypotheses. Under the one-sided
  non-regression tolerance, this yields
  `SURFACE04_MATCHES_PR36_WITH_ACCEPTABLE_TRADEOFF`: it does not beat PR #36
  cleanly, but is a credible real-gate tradeoff candidate. Surface05 is
  justified as a separate controlled diagnostic requiring its own work order
  and review. This result does not change `accepted_parent` or issue
  `TRAINING_ELIGIBLE`.
- ADR 0009 Phase 2 and Work Order 0038 tested decoder, joint, and exactly
  `encoder.layers.22` plus `encoder.layers.23`, starting again from the
  untouched Nemotron base. ARTUR controller-dev selected round 3 and the
  operational rule stopped at round 6 after 12,000 optimizer steps and 96,000
  exposures. The selected checkpoint scored 46.564/14.950 on FLEURS-v2 and
  53.473/17.473 on ARTUR-J with zero empty hypotheses. It stayed within the
  best-known one-sided envelope and improved both ARTUR-J metrics, yielding
  `SURFACE05_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF`. This is diagnostic
  evidence only.
- ADR 0009 Phase 3 and Work Order 0039 tested decoder, joint, and exactly
  `encoder.layers.20` through `encoder.layers.23` from the untouched base.
  ARTUR controller-dev selected round 5, which scored 44.506/13.528 on
  FLEURS-v2 and 50.590/15.803 on ARTUR-J with zero empty hypotheses. This is
  `SURFACE06_NEW_BEST_DIRECTIONAL_CANDIDATE`, diagnostic only.
- ADR 0009 Phase 4 and Work Order 0040 tested one fusion-bottleneck diagnostic
  from the untouched base. It kept the Surface06 final-four encoder depth and
  added only `prompt_kernel`, proven in the pinned live model as the separable
  post-concatenation prompt/acoustic projection. ARTUR controller-dev selected
  round 13. The selected checkpoint scored 42.084/12.985 on FLEURS-v2 and
  47.357/14.805 on ARTUR-J with zero empty hypotheses, improving all four
  directional real-gate metrics versus Surface06 and yielding
  `SURFACE07_NEW_BEST_DIRECTIONAL_CANDIDATE`. This remains diagnostic,
  noncanonical evidence. Surface08 and full-encoder training remain
  prohibited.
- Experiment 0028 evaluated the untouched base, PR #36 round 20, Surface06
  round 5, and Surface07 round 13 under canonical batch-1, no-bucketing, FP32,
  TF32-disabled evaluation on FLEURS-v2 and ARTUR-J. Surface07 remained best on
  all four real-gate metrics at 42.090/12.988 on FLEURS-v2 and 47.532/15.025
  on ARTUR-J, with zero empty hypotheses. The classification is
  `CANONICAL_SURFACE07_CONFIRMED_NEW_BEST`; no checkpoint was accepted.
- GitHub is for method and evidence; Hugging Face will be used for model artifacts.
- Pinned model revision: `3fc30f3e2ae5d78d462441f3ce89dda694f89bd7`.
- Pinned NeMo revision for the baseline interface: `8044a3924bfcfe8ef71d792bb73bf274fe853575`.
- Correct checkpoint SHA256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
- Historical M1/M2 development hardware: 48 GB class RAM, 2 x RTX 2080 Ti with
  11 GB each, one GPU used per process.
- Current A100 development hardware: physical GPU 1 selected with
  `CUDA_VISIBLE_DEVICES=1`; PyTorch sees exactly one logical device, `cuda:0`.
- Project-owned GPU helpers now accept exactly one visible A100 or RTX 2080 Ti,
  reject CPU fallback, and reject multiple visible GPUs.
- First M3 trainable surface: one additive `sl-SI` prompt-column delta with
  2048 effective trainable scalars, later merged into only the selected first
  prompt-projection column.
- M3 prompt-column micro-result: Phase A supported, Phase B executed, synthetic
  training WER improved from 92.5 to 38.333 and empty synthetic-training
  hypotheses dropped from 3 to 0. Synthetic holdout WER was unchanged at 87.5.
  Public FLEURS smoke WER regressed from 75.0 to 85.0.
- Historical FLEURS-v1 base baseline: normalized corpus WER 52.734 and CER
  16.423 with 0 empty hypotheses. This is deprecated and must not be used as
  complete-split quality evidence. ARTUR-J normalized corpus WER 67.453 and CER
  29.016 with 12 empty hypotheses remains valid.
- Historical FLEURS-v1 micro-proof diagnostic: normalized WER regressed to
  66.961; this is deprecated. ARTUR-J normalized WER regressed to 76.190 and
  remains valid.
- Round 1 project-generated curriculum diagnostic: selected synthetic training
  normalized WER improved from 89.070 to 51.632, fixed synthetic holdout
  normalized WER moved from 77.563 to 76.983, historical FLEURS-v1 normalized
  WER regressed from 52.734 to 70.885, and ARTUR-J normalized WER regressed
  from 67.453 to 80.996. The challenger is rejected and is not a parent; ARTUR-J
  independently failed promotion.
- Residual-adapter diagnostic: rank 16 fixed synthetic-holdout normalized WER
  improved to 63.926 and rank 64 improved to 54.836, but rank 16 regressed
  historical FLEURS-v1 to 67.076 and ARTUR-J to 78.943, while rank 64 regressed
  historical FLEURS-v1 to 70.430 and ARTUR-J to 81.739. No residual adapter is
  accepted; ARTUR-J independently failed promotion.

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

ADR 0008 now permits `artur-controller-dev-v1` for aggregate real-acoustic
run-control and early stopping only when an explicit work order authorizes it.
Work Order 0032 applies that exception to a scale-2000 decoder+joint RNNT rerun
with per-round ignored checkpoints. FLEURS-v2, ARTUR-J immutable gate data, and
any final blind test remain unavailable for early stopping, checkpoint
selection, hyperparameter selection, prompt construction, or training.

Do not prepare another prompt-column training rerun from the current
single-voice corpus as if it were promotion-eligible. The next useful
development work is governed synthetic-scale data construction followed by
frozen-encoder emission adaptation, then validation-only real-gate comparison.
Batch-32 directional evidence can guide iteration, but canonical batch-1
validation is still required before any acceptance discussion.

Use the rejected Round 1 and residual-adapter aggregate evidence to design the
next controlled work order. The accepted parent remains the untouched Nemotron
base checkpoint; do not treat the micro-proof checkpoint, Round 1 checkpoint, or
residual adapter as an accepted parent.

## Do not do next

- Do not expose real-gate reference text to GaMS.
- Do not use real Slovenian acoustic samples for training or steering.
- Do not train the acoustic encoder while the training signal remains
  synthetic-only.
- Do not create a service API or UI.
- Do not publish a checkpoint.
- Do not add private data to obtain an early score.
- Do not accept a synthetic-only adapter without real-gate non-regression and a
  work order that explicitly permits the next controlled step.

## Strategic questions after the next PR

- What is the zero-shot Slovenian baseline on the approved development set?
- Which exact NeMo revision should become the project pin?
