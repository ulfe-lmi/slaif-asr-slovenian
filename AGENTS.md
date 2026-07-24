# AGENTS.md

This file is the project constitution for autonomous and semi-autonomous coding agents.

## Discovery summary

- **Domain problem:** produce a high-quality, low-latency Slovenian, and
  eventually Slovenian-English, speech recognizer using synthetic-only training
  and validation-only real Slovenian speech.
- **Active strategy:** ADR 0007 adopts a Slovenian-first synthetic development
  track: real Slovenian acoustic data is validation-only, the encoder stays
  frozen while training remains synthetic-only, and broader emission-side
  adaptation requires explicit work orders.
- **Product shape:** a reproducible adaptation, evaluation, and release pipeline around an open-weight streaming ASR base model.
- **Initial base model:** `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- **Training framework:** NVIDIA NeMo, pinned by commit or release in executable work.
- **Adaptation source:** small, failure-directed batches of GaMS-generated Slovenian text rendered by the selected external Piper Slovenian TTS path.
- **Control loop:** generate -> synthesize -> evaluate -> select failures -> train bounded update -> run gates -> accept or roll back.
- **Repository shape:** standalone SLAIF repository. Do not fork or vendor the full NeMo repository.
- **Distribution shape:** GitHub for code and evidence; Hugging Face for adapters or derived model artifacts.
- **Current milestone:** M3 prompt-column proof is complete for one
  micro-experiment. The training-data constitution is adopted, text-stage
  corpus-validation tooling exists, and the first GaMS corpus-v2 candidate
  reservoir has reached `TEXT_ACCEPTED` and `AUDIO_ACCEPTED` as a single-voice
  synthetic candidate pool. A100 real-gate evaluation now has a parity-checked
  batch-1 policy and a valid untouched-base FLEURS-v2 baseline. An independent
  synthetic diagnostic holdout has reached `TEXT_ACCEPTED` and
  `AUDIO_ACCEPTED`, scoring has run on both synthetic partitions, and a
  selected-training manifest is ready under
  `SELECTED_TRAINING_MANIFEST_READY`. A named `DIAGNOSTIC_ONLY` corpus-v2
  prompt-column experiment has run and is synthetic-only: no checkpoint is
  accepted, and true A100 minibatch training was not scientifically equivalent
  to the batch-size-1 reference. A follow-up speaker-range resampling
  diagnostic also remained unsupported: it improved the synthetic holdout but
  did not mitigate real-gate regression. A frozen-base Slovenian RNNT
  joint-adapter diagnostic trained only one new adapter, left every pretrained
  tensor frozen, emitted shared live progress, and also remained synthetic-only.
  A Supertonic 3 multi-voice diagnostic then trained the same frozen-base joint
  adapter on eight preset synthetic voice styles and reduced, but did not
  eliminate, the Piper joint-adapter real-gate regression burden. It remains
  `DIAGNOSTIC_ONLY`: no adapter or checkpoint is accepted.
  `TRAINING_ELIGIBLE` certification, promotion-eligible model training, and
  production ASR work remain incomplete.

## Mission

Build an auditable SLAIF pipeline that can adapt and evaluate open-weight
streaming ASR models for Slovenian and Slovenian-English while keeping real
Slovenian acoustic data reserved for validation and acceptance evidence.

The core user promise is:

> A released SLAIF Slovenian ASR artifact must be reproducible, accurately attributed, honestly evaluated, and no broader in its claims than the available evidence.

## Human, strategic model, and execution-agent roles

- The **human lead** owns domain truth, priorities, risk tolerance, merge decisions, model publication, and release claims.
- The **strategic model** owns architecture synthesis, work-order design, review briefs, risk tracking, and evidence interpretation.
- The **execution agent** implements one bounded PR-sized work order at a time and returns evidence.
- The execution agent must not decide product scope, release readiness, licensing exceptions, or public performance claims.

## Repository truth

The remote repository, pull requests, CI logs, committed documentation, tagged releases, and published model cards are project truth.

Local virtual machines, caches, downloaded checkpoints, generated audio, and uncommitted experiment output are disposable.

If this file conflicts with the live repository, report the conflict before editing.

## Architecture boundaries

The repository will own:

- environment and dependency pins;
- inference wrappers;
- manifest and dataset tooling;
- selective fine-tuning policies;
- active-learning orchestration;
- evaluation and acceptance gates;
- adapter extraction and application tooling;
- release metadata and model cards;
- tests and documentation.

The repository will not own:

- the NeMo framework source tree;
- the NVIDIA base checkpoint;
- the Piper TTS source tree or binary;
- the `sl_SI-artur-medium` voice artifact;
- GaMS model weights;
- generated GaMS candidate pools, active-curriculum manifests, or round outputs;
- the Slovenian TTS implementation unless separately imported under an approved license;
- private or third-party speech corpora;
- production deployment credentials.

## Non-negotiable invariants

1. **No large model or data artifacts in Git.**
   - Never commit `.nemo`, `.ckpt`, `.pt`, `.pth`, `.safetensors`, `.onnx`, TensorRT engines, raw training audio, generated corpora, or experiment checkpoints.
2. **No secrets or personal speech.**
   - Never commit credentials, private transcripts, personal identifiers, internal storage URLs, or raw participant recordings.
3. **Base-model attribution remains explicit.**
   - Never describe a derived checkpoint as a SLAIF foundation model.
   - Use wording such as “SLAIF Slovenian adaptation of NVIDIA Nemotron 3.5 ASR Streaming.”
4. **Licenses remain separate.**
   - Repository code is Apache-2.0.
   - Base and derived model artifacts remain subject to the applicable OpenMDW license and model-card obligations.
5. **Do not replace the tokenizer casually.**
   - The initial strategy reuses the base tokenizer.
   - A tokenizer change requires an ADR, an explicit decoder-reinitialization analysis, and human approval.
6. **Protect acoustic and streaming behavior while optimizing for Slovenian.**
   - Start with a declared trainable surface and prove parameter integrity.
   - Broader non-encoder emission adaptation is permitted only by explicit work
     order and evidence.
   - Encoder training remains prohibited while training data is synthetic-only
     unless a later ADR and human approval explicitly change that rule. ADR
   0009 and Work Orders 0037/0038/0039/0040/0043 provide bounded exceptions for
   exactly the final encoder block, final two encoder blocks, final four
   encoder blocks, and the Work Order 0040 final-four-plus-`prompt_kernel`
   fusion diagnostic on fixed scale-2000 data. Work Order 0043 authorizes
   exactly one Surface08 boundary diagnostic with all encoder layers and the
   proven `prompt_kernel`, while the frontend and prompt identity remain
   frozen. It is not general full-encoder authorization and does not authorize
   prompt identity changes, Surface09, or full-model training.
7. **Real speech decides checkpoint acceptance.**
   - Synthetic improvement alone is insufficient.
   - Real Slovenian acoustic data is validation-only and must not be used for
     training, synthetic prompt construction, selected-training membership,
     early stopping, or per-sample steering, except that
     `artur-controller-dev-v1` may be used for aggregate run-control and early
     stopping only when ADR 0008 and an explicit work order authorize it.
     Immutable gates and final blind tests remain unavailable for early
     stopping or selection.
8. **Skipped is not passed.**
   - Every report distinguishes passed, failed, skipped, not run, blocked, and out of scope.
9. **No performance claim without a committed protocol.**
   - Metrics must name dataset, split, normalization, latency/context setting, checkpoint hash, and evaluation code revision.
10. **The agent never merges its own PR.**

## Data policy

Read `docs/data-policy.md` before touching manifests, speech, transcripts, generated text, or metrics.

Required defaults:

- use absolute or repository-resolved paths only in local manifests;
- never commit local absolute paths;
- store public manifests only when redistribution is permitted;
- separate controller-development, immutable gate, and final blind-test partitions;
- do not expose immutable-gate or final-test sentences to GaMS;
- record source, license, consent status, generation provenance, and hashes;
- preserve a synthetic/real distinction in every report.

Current canonical real development gates:

- `fleurs-sl-si-test-full-v2`, the complete FLEURS Slovenian `sl_si` test split
  at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, with occurrence IDs
  derived from deterministic source-row indexes;
- `artur-j-public-gate-v1`, the deterministic ARTUR-J public-speech project
  gate.

Current real controller-development partition:

- `artur-controller-dev-v1`, an ARTUR public-speech partition governed by ADR
  0008. It may be used only for aggregate run-control and explicitly
  authorized future early stopping. It is spent development data, not immutable
  acceptance evidence.

Historical `fleurs-sl-si-test-full-v1` evidence is deprecated because repeated
upstream source IDs caused duplicate sample IDs and WAV overwrites. It must not
be used as complete-split quality evidence.

## Training-data constitution

Read `docs/training-data-constitution.md` before generating candidate text,
selecting curriculum samples, creating training or holdout partitions,
synthesizing TTS data, ingesting real training speech, scoring candidates for
difficulty, training any model parameter, or interpreting synthetic-data
results.

Non-negotiable rules:

- Corpus IDs, row numbers, group labels, batch labels, filenames, and
  provenance markers must never enter spoken or target text.
- Schema validity, literal duplicate checks, and character-ngram checks do not
  establish training eligibility.
- Generated text must pass multi-view structural fingerprints,
  concentration analysis, partition-family checks, and Slovenian linguistic
  review before TTS or training.
- Training and holdout must be disjoint by content and family, not merely by
  ID; every acoustic variant of one underlying utterance remains in one
  partition.
- Synthetic holdout is diagnostic only. Real speech decides checkpoint
  acceptance.
- Hard-example selection operates only on an already accepted corpus and must
  preserve template, source, domain, and voice diversity.
- A privacy-safe data acceptance certificate is required before training.
- The Round 1 v1 corpora identified in the training-data constitution are
  permanently retired from training, steering, model comparison, and promotion.
- Skipped, blocked, unknown, or unrun quality checks prevent
  `TRAINING_ELIGIBLE` status.
- Use `scripts/validate_training_corpus.py` and
  `configs/data_quality/training_text_v1.json` for new text-stage admission.
  The legacy Round 1 validator is historical only and is not an admission
  authority for new corpora.
- The GaMS corpus-v2 candidate reservoir has passed whole-file human review
  expansion, text admission, Piper synthesis, and waveform validation through
  `AUDIO_ACCEPTED`. The separately sourced 96-row corpus-v2 independent
  synthetic diagnostic holdout has reached `TEXT_ACCEPTED` and
  `AUDIO_ACCEPTED`. The scoring certificate permitted ASR scoring and
  selected-training construction only; that scoring is complete and a
  selected-training manifest now has `SELECTED_TRAINING_MANIFEST_READY`
  status. That manifest was used once under the Work Order 0020
  `DIAGNOSTIC_ONLY` exception for prompt-column evidence; the result was
  synthetic-only and did not accept a checkpoint. Work Order 0022 used the same
  selected-training manifest under a separate `DIAGNOSTIC_ONLY` exception for
  one frozen-base RNNT joint-hidden adapter; it was also synthetic-only and did
  not accept an adapter or checkpoint. Work Order 0023 tested Supertonic 3
  preset multi-voice synthetic audio under another `DIAGNOSTIC_ONLY`
  exception. It mitigated the Piper joint-adapter real-gate regression burden
  but did not accept an adapter or checkpoint. No `TRAINING_ELIGIBLE` decision
  exists, and promotion-oriented model training remains prohibited.

## Model and dependency policy

- Use the official checkpoint and official NeMo interfaces.
- Pin the model revision and NeMo revision for every reproducible run.
- Verify unstable APIs from primary documentation or source before implementation.
- Do not vendor NeMo wholesale.
- Copy upstream source only when a work order explicitly requires a small modification; preserve copyright and license headers and record the upstream path and commit.
- Do not add a dependency merely to avoid understanding existing project or NeMo functionality.
- Every new dependency needs a purpose, license check, and reproducibility impact statement.

## Runtime policy

Execution belongs in a disposable, rebuildable GPU environment.

Agents may:

- install required local tools in the approved execution VM;
- download the official base checkpoint into ignored storage;
- create disposable caches and test outputs;
- start local services needed for tests;
- use only the GPUs and visibility settings authorized by the active work order.

Current development hardware policy:

- Historical M1 and M2 evidence used one NVIDIA RTX 2080 Ti process-visible GPU.
- The current A100 development host uses physical GPU 1 selected with
  `CUDA_VISIBLE_DEVICES=1`; PyTorch must see exactly one logical CUDA device,
  `cuda:0`.
- Project-owned GPU execution helpers must use the shared single-GPU policy:
  exactly one visible A100, RTX 2080 Ti, or NVIDIA GeForce RTX 3090 is accepted,
  multiple visible GPUs are rejected, CPU fallback is rejected, and code must
  not assume that the physical selector is zero. An RTX 3090 must expose at
  least 22 GiB VRAM.
- Other physical GPUs remain unused unless a later work order explicitly
  permits them.
- A100 real-gate evaluation currently uses the measured batch policy in
  `configs/evaluation/a100_streaming_batch_policy.json`: batch size 1,
  no duration bucketing, FP32, TF32 disabled. Batch size 1 remains the
  scientific reference mode because larger tested batches changed transcripts.
- Work Order 0020 measured true prompt-column training minibatches on A100.
  Batch size 8 improved throughput but was not scientifically equivalent to
  the batch-size-1 reference arm, so future training work must not assume
  minibatch equivalence without a new work order.
- Cache-aware inference uses FP32 under the pinned NeMo implementation.
- RTX 2080 Ti remains a supported smaller development platform. Future 2080 Ti
  training should use FP16 AMP rather than BF16 unless a later work order
  changes the policy.

Agents must not:

- access production systems;
- use production credentials;
- mutate external datasets;
- publish a model;
- create a Hugging Face release;
- change organization or repository security settings;
- ask the human to perform routine dependency installation that the execution VM permits.

## TTS boundary

The selected initial Slovenian TTS engine is `OHF-Voice/piper1-gpl` with voice
`rhasspy/piper-voices` `sl_SI-artur-medium`.

Piper is GPL-3.0-or-later and remains an external executable dependency:

- install it only into repository-local `.venv-piper`;
- keep source under ignored `.external/piper1-gpl`;
- invoke it through argv subprocess calls with `shell=True` forbidden;
- do not import Piper from the Apache-licensed `slaif_asr` package;
- do not commit Piper source, binaries, voice files, generated WAVs, logs, or
  local manifests;
- do not represent generated audio as Apache-2.0 merely because repository
  orchestration code is Apache-2.0.

The `sl_SI-artur-medium` voice metadata is inconsistent across repository,
model-card, and ARTUR source records. Apply the conservative ARTUR CC BY-SA 4.0
attribution and publication policy until later legal review. Public synthetic
audio release is not authorized by M2 ingestion work.

S6TTS is a lab-origin legacy Slovenian TTS system that may be used only as an
external local executable dependency for synthetic diagnostic audio when a work
order explicitly authorizes it. Keep S6TTS source, binaries, runtime data,
dictionaries, diphone/acoustic resources, generated WAVs, logs, and local
manifests outside this repository unless a later import/license ADR authorizes
otherwise. Generated S6TTS audio remains ignored synthetic material. Public
distribution of S6TTS-generated audio, or models trained on it, is not
authorized until a provenance and release review permits it.

## GaMS boundary

GaMS is an external local candidate-text generator. Use only pinned Hugging Face
revisions, keep model weights and generated candidate pools out of Git, and run
model generation only with the GPU visibility authorized by the active work
order. Do not send immutable real-gate or final-test reference text to GaMS.
Synthetic candidate-pool failures may steer later rounds only when the work
order explicitly permits it.

## Workflow

For implementation work:

1. Read `AGENTS.md`, `CLAUDE.md`, relevant ADRs, and the assigned work order.
2. Verify current `main`, open PR state, and a clean or explicitly known working tree.
3. Create a feature branch from current `main`.
4. Keep scope to one coherent PR.
5. Commit only related files.
6. Push the branch and open a PR.
7. Do not merge.
8. Return the required evidence report.

Preferred branch prefixes:

```text
docs/
chore/
feat/
fix/
test/
exp/
release/
```

Do not push directly to `main`.

## Scope control

Unless the work order explicitly says otherwise:

- do not perform broad refactors;
- do not change public artifact names;
- do not change the selected base model;
- do not introduce a web service or UI;
- do not add deployment infrastructure;
- do not add a database;
- do not change licenses;
- do not publish weights;
- do not create or modify real datasets;
- do not claim production readiness.

When a useful adjacent change is discovered, report it as follow-up rather than implementing it silently.

## Testing and evidence

Each work order must name exact verification. Agents must also run the narrowest relevant checks for changed files.

Minimum evidence for Python implementation work, once Python tooling exists:

```text
format/lint command and exact result
type-check command and exact result, when configured
focused unit tests and exact result
relevant integration or GPU test and exact result
git diff --check
large-file and secret scan
```

GPU-dependent tests must state:

- GPU model;
- CUDA and PyTorch versions;
- precision;
- checkpoint revision;
- context/latency setting;
- command;
- result;
- peak memory when relevant.

A test that cannot run because of environment limitations is `ENVIRONMENT_BLOCKED`, not passed.

## Experiment discipline

Every experiment must record:

- parent checkpoint hash;
- base-model revision;
- NeMo revision;
- configuration;
- trainable parameter list and count;
- dataset/manifest hashes;
- random seeds;
- hardware;
- exact commands;
- training metrics;
- per-dataset evaluation;
- acceptance or rollback decision.

The accepted checkpoint, not simply the newest checkpoint, is the parent of the next active-learning round.

## Documentation contract

Update documentation when behavior, interfaces, data contracts, evaluation protocols, or release scope change.

Important documents:

- `docs/architecture.md`
- `docs/data-policy.md`
- `docs/testing-strategy.md`
- `docs/evaluation-protocol.md`
- `docs/release-policy.md`
- `docs/project-handoff.md`
- relevant ADRs
- relevant work order

Do not use documentation to imply that planned behavior exists.

## Security

- Never commit secrets.
- Never print credentials in logs.
- Never include private speech or transcript excerpts in public issues or PRs.
- Use fake placeholders in examples.
- Treat external model and dataset downloads as supply-chain inputs.
- Preserve exact checksums or revisions.
- If sensitive material appears, stop, avoid copying it further, and report privately.

See `SECURITY.md`.

## Definition of done for a PR

A PR is ready for strategic review only when:

- scope matches the work order;
- acceptance criteria are mapped to evidence;
- relevant tests were run or honestly classified;
- no unrelated files changed;
- documentation is aligned;
- no large artifacts or secrets are committed;
- changed interfaces are described;
- known limitations and follow-ups are explicit;
- a structured agent report is supplied.

## Required final agent report

```markdown
## Agent Report

Branch:
Commit:
Pull request:

## Goal completed
- ...

## Files changed
- path: reason

## Evidence
- command: exact result
- command: exact result

## Experiment details
- Base model revision:
- NeMo revision:
- Hardware:
- Trainable parameters:
- Dataset/manifest hashes:

## Documentation impact
- ...

## Safety confirmations
- No production secrets or private speech committed.
- No model weights or generated corpora committed.
- No skipped or blocked test reported as passed.
- No unrelated files changed.
- No publication or merge performed.

## Known limitations
- ...

## Follow-up recommended
- ...
```
