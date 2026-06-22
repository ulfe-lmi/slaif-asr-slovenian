# AGENTS.md

This file is the project constitution for autonomous and semi-autonomous coding agents.

## Discovery summary

- **Domain problem:** produce a high-quality, low-latency Slovenian speech recognizer without relearning transferable multilingual acoustic and streaming capabilities.
- **Product shape:** a reproducible adaptation, evaluation, and release pipeline around an open-weight streaming ASR base model.
- **Initial base model:** `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- **Training framework:** NVIDIA NeMo, pinned by commit or release in executable work.
- **Adaptation source:** small, failure-directed batches of GaMS-generated Slovenian text rendered by the selected external Piper Slovenian TTS path.
- **Control loop:** generate -> synthesize -> evaluate -> select failures -> train bounded update -> run gates -> accept or roll back.
- **Repository shape:** standalone SLAIF repository. Do not fork or vendor the full NeMo repository.
- **Distribution shape:** GitHub for code and evidence; Hugging Face for adapters or derived model artifacts.
- **Current milestone:** M3 prompt-column proof is complete for one
  micro-experiment. Broader M2 data governance and production ASR work remain
  incomplete.

## Mission

Build an auditable SLAIF pipeline that can adapt and evaluate open-weight streaming ASR models for Slovenian while preserving transferable acoustic, multilingual, and streaming behavior.

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
6. **Preserve transferable behavior.**
   - Start with the smallest declared trainable surface.
   - Escalation from prompt-specific adaptation to shared decoder/joint or encoder weights requires measured evidence and a work order.
7. **Real speech decides checkpoint acceptance.**
   - Synthetic improvement alone is insufficient.
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

- M1 and M2 use one NVIDIA RTX 2080 Ti process-visible GPU.
- The default physical device is GPU 0, selected with `CUDA_VISIBLE_DEVICES=0`.
- The second RTX 2080 Ti remains unused unless a later work order explicitly permits it.
- A100 is not a default prerequisite. It becomes mandatory only through a later work order backed by measured memory, throughput, or authoritative benchmarking requirements.
- Cache-aware inference uses FP32 under the pinned NeMo implementation.
- Future 2080 Ti training should use FP16 AMP rather than BF16 unless a later work order changes the policy.

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
