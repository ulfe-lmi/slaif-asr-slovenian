# Work Order 0011: Adopt the Training-Data Constitution

Status: ready for execution
Repository: `ulfe-lmi/slaif-asr-slovenian`

This is a bounded governance and documentation task.

Do not implement the new corpus validator, generate data, synthesize audio,
load a model, run ASR inference, or use a GPU in this work order.

## Governing instructions

Read and obey:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/data-policy.md`
- `docs/training-plan.md`
- `docs/testing-strategy.md`
- `docs/evaluation-protocol.md`
- relevant ADRs and experiment reports

Verify current `main`, open pull requests, and the working tree before editing.

Expected state at issuance:

- PR #13, `fix: rebuild FLEURS gate with unique occurrence IDs`, is merged;
- `fleurs-sl-si-test-full-v2` is the canonical FLEURS development gate;
- the historical FLEURS v1 gate is deprecated;
- no training-data constitution has yet been committed.

Live repository state is authoritative. Report any material difference before
proceeding.

## Human-provided source document

The human lead will place this untracked file in the repository root:

```text
training-data-constitution.md
```

This is the approved source document for the new constitutional companion
policy. Read it completely before editing.

If the file is absent, empty, truncated, or cannot be read, stop and report the
problem. Do not reconstruct it from memory, old chat, or another document.

Move the source document to:

```text
docs/training-data-constitution.md
```

Do not commit a second copy at repository root.

## Branch and pull request

Use:

```text
Branch:
docs/adopt-training-data-constitution

Commit:
docs: adopt training-data constitution

Pull request title:
docs: adopt training-data constitution
```

Create a work-order record at:

```text
docs/work-orders/0011-adopt-training-data-constitution.md
```

If 0011 already exists in live main, use the next unused work-order number and
report the adjustment.

Do not merge the pull request.

## Goal

Adopt `docs/training-data-constitution.md` as the authoritative detailed
companion to the project constitution.

The intended hierarchy is:

- `AGENTS.md` contains concise mandatory project law.
- `docs/training-data-constitution.md` contains detailed training-data
  doctrine, algorithms, admission stages, incident findings, certificates, and
  review requirements.
- `docs/data-policy.md` continues to govern general privacy, provenance,
  licensing, storage, and partition handling.
- An approved work order may make requirements stricter.
- A named exception may weaken a requirement only with explicit human approval,
  a written rationale, and a narrowed scientific claim.

Do not copy the full companion document into `AGENTS.md`.

## Adoption edits to the source document

Preserve the substance of the human-provided document.

Make only the edits required for repository adoption and consistency:

- change the status from Proposed constitutional companion policy to Adopted
  constitutional companion policy;
- retain version 1.0 unless an actual substantive policy change is required;
- retain the adoption date;
- ensure the intended path is `docs/training-data-constitution.md`;
- ensure links and repository-relative paths are valid;
- align references to the canonical real gates with current repository truth;
- preserve the exact retired corpus hashes;
- preserve the distinction between normative and non-normative sections;
- preserve the appendices unless a repository conflict requires a documented
  editorial adjustment.

Do not silently weaken any MUST, MUST NOT, hard gate, retirement rule,
partition rule, certificate requirement, or experiment-interpretation rule.

Do not alter incident counts or hashes merely for editorial convenience.

If the historical corpus artifacts are available in ignored local storage, the
agent may independently verify aggregate incident figures with a deterministic
local script or one-off command. Do not commit raw corpora or verification
input.

If those artifacts are unavailable, preserve the figures as human-approved
incident findings and clearly state in the agent report that they were not
independently recomputed during this documentation PR.

## Required `AGENTS.md` amendment

Add a concise section named:

```text
## Training-data constitution
```

Use Appendix A of the supplied document as the basis.

The section must require agents to read
`docs/training-data-constitution.md` before:

- generating candidate text;
- selecting curriculum samples;
- creating training or holdout partitions;
- synthesizing TTS data;
- ingesting real training speech;
- scoring candidates for difficulty;
- training any model parameter;
- interpreting synthetic-data results.

The section must preserve these non-negotiable rules:

- Corpus bookkeeping identifiers never enter spoken or target text.
- Schema validity and literal duplicate checks do not establish training
  eligibility.
- Multi-view structural fingerprints, concentration analysis,
  partition-family checks, and Slovenian linguistic review are required before
  TTS or training.
- Train and holdout are disjoint by content and family, not merely by ID.
- All acoustic variants of one utterance remain in one partition.
- Synthetic holdout is diagnostic, not real-generalization evidence.
- Real speech decides checkpoint acceptance.
- Hard-example selection operates only on an already accepted corpus.
- A privacy-safe data acceptance certificate is required before training.
- The identified Round 1 v1 corpora are permanently retired.
- Skipped, blocked, unknown, or unrun checks prevent `TRAINING_ELIGIBLE`
  status.

Keep this section concise enough that `AGENTS.md` remains usable as the
top-level constitution.

## Required `CLAUDE.md` amendment

Add a short critical reminder that:

- `AGENTS.md` remains authoritative;
- all data-related work must also read
  `docs/training-data-constitution.md`;
- only `TRAINING_ELIGIBLE` data may enter promotion-oriented training;
- the retired Round 1 corpora must not be reused.

Do not duplicate the full policy.

## ADR requirement

Add a concise accepted ADR, preferably:

```text
docs/adr/0006-training-data-admission-policy.md
```

The ADR must record:

### Context

Round 1 passed narrow schema and duplicate validation while remaining
structurally repetitive, linguistically defective, and train/holdout
template-confounded.

Hard-example selection subsequently amplified artifacts already admitted into
the source pool.

Synthetic improvement therefore did not establish real Slovenian
generalization.

### Decision

- adopt `docs/training-data-constitution.md`;
- require an explicit data-status state machine;
- require a privacy-safe acceptance certificate before training;
- prohibit model training on data that has not reached the required status;
- retire the three v1 corpus identities;
- separate corpus acceptance from model experimentation.

### Consequences

- corpus validation becomes a first-class pre-GPU stage;
- future data work requires stronger structural and linguistic evidence;
- historical experiments remain auditable;
- architecture conclusions drawn from corpus-confounded experiments are
  narrowed;
- later implementation work is required for reusable validation tooling.

Update ADR 0003, Failure-directed synthetic curriculum, with a short
qualification that:

- failure-directed selection remains accepted;
- selection may begin only after corpus admission;
- high ASR error cannot rehabilitate malformed or structurally invalid text;
- ADR 0006 and the training-data constitution govern corpus eligibility.

Do not mark ADR 0003 rejected unless repository evidence supports doing so.

## Historical experiment disposition

Update these reports with a prominent, concise qualification:

- `docs/experiments/0004-slovenian-curriculum-round-1.md`
- `docs/experiments/0005-slovenian-residual-adapter-proof.md`

For Experiment 0004, state that:

- the challenger rejection remains valid;
- the training corpus was later found structurally repetitive, linguistically
  defective, and unsuitable for real-generalization training;
- its synthetic holdout was ID-disjoint but not content-family-disjoint;
- the experiment must not be cited as evidence that a clean curriculum would
  fail.

For Experiment 0005, state that:

- execution and parameter-integrity evidence remain historical evidence;
- both adapters remain rejected as checkpoint parents;
- ARTUR-J independently demonstrated real-speech regression;
- the experiment is corpus-confounded;
- it must not be cited as proof that residual adapters, their placement, or
  added capacity are intrinsically unsuitable.

Do not rewrite historical numerical results, hashes, commands,
configurations, or promotion decisions.

Do not point historical experiment configurations at new data or FLEURS v2.
Historical reproducibility must remain intact.

## Retired corpus identities

The following identities must be recorded exactly as permanently retired:

Candidate pool:

```text
0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17
```

Synthetic holdout:

```text
ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9
```

Selected training manifest:

```text
92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4
```

They must not be used for:

- future training;
- model-surface comparison;
- adapter-rank comparison;
- learning-rate selection;
- early stopping;
- generator steering;
- promotion;
- public corpus-quality claims.

They may remain available in ignored local storage for audit and
regression-test design.

## Documentation integration

Add concise links and consistent policy language where directly relevant.

Required review targets:

- `README.md`
- `CHANGELOG.md`
- `docs/architecture.md`
- `docs/data-policy.md`
- `docs/training-plan.md`
- `docs/testing-strategy.md`
- `docs/evaluation-protocol.md`
- `docs/roadmap.md`
- `docs/project-handoff.md`
- `docs/adr/0003-failure-directed-synthetic-curriculum.md`
- `docs/experiments/0004-slovenian-curriculum-round-1.md`
- `docs/experiments/0005-slovenian-residual-adapter-proof.md`

Required effects:

- `README.md` points contributors to the new policy without reproducing it.
- `docs/data-policy.md` explains the relationship between general data
  governance and training-data admission.
- `docs/architecture.md` shows corpus admission before TTS, scoring,
  selection, or model training.
- `docs/training-plan.md` requires `TRAINING_ELIGIBLE` data before
  promotion-oriented training.
- `docs/testing-strategy.md` identifies the future reusable validator and
  adversarial regression-fixture obligation.
- `docs/evaluation-protocol.md` distinguishes synthetic diagnostic holdout
  from real-speech promotion gates.
- `docs/roadmap.md` records training-data validation and corpus v2 as
  prerequisites for the next model-training experiment.
- `docs/project-handoff.md` records the incident, corpus retirement, and the
  new constitutional rule.
- `CHANGELOG.md` records policy adoption without implying that validator code
  has already been implemented.

Do not mechanically edit every document. Make only changes needed to remove
contradiction, establish discoverability, and record the decision.

## Non-goals

Do not:

- implement `slaif_asr/data_quality.py`;
- implement `scripts/validate_training_corpus.py`;
- add a data-quality configuration;
- generate a data certificate;
- regenerate or clean the v1 corpus;
- generate corpus v2;
- run GaMS;
- run Piper;
- run Nemotron;
- run ASR scoring;
- run A100 inference or training;
- benchmark batching;
- change model architecture;
- change promotion thresholds;
- create or modify real datasets;
- publish any artifact;
- merge the pull request.

The validator implementation must be a later bounded work order.

## Privacy and repository safety

The committed policy may include:

- aggregate corpus counts;
- cryptographic hashes;
- short synthetic examples needed to explain the incident;
- algorithms, thresholds, reason codes, and certificate schemas.

It must not include:

- the complete candidate pool;
- the complete selected-training corpus;
- the complete synthetic holdout;
- raw protected FLEURS or ARTUR-J references;
- real-gate hypotheses;
- generated audio;
- local manifests;
- local absolute paths;
- private speech;
- credentials;
- model artifacts.

Inspect the moved policy before commit to ensure it contains no accidental
local path or raw protected content.

## Acceptance criteria

The PR is ready for strategic review only when:

- the source file has been moved to
  `docs/training-data-constitution.md`;
- its status is adopted;
- there is no duplicate root-level copy;
- `AGENTS.md` contains the concise mandatory section;
- `CLAUDE.md` points to the policy;
- the new ADR records the decision and consequences;
- ADR 0003 is properly qualified;
- Experiments 0004 and 0005 contain the narrowed scientific interpretation;
- all retired hashes are exact;
- current documentation points to FLEURS v2 as canonical;
- no historical result or configuration was silently rewritten;
- no raw data or model artifact was committed;
- documentation links resolve;
- repository checks pass;
- no GPU or model execution occurred;
- the PR remains unmerged.

## Verification

Run:

```text
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pip check
bash -n scripts/*.sh
git diff --check
git diff --cached --check
```

Inspect tracked artifacts:

```text
git status --short
git ls-files | grep -E '\.(wav|flac|mp3|ogg|m4a|nemo|ckpt|pt|pth|safetensors|onnx|engine|plan)$' || true
```

Inspect constitutional integration:

```text
rg -n "training-data-constitution|TRAINING_ELIGIBLE|DIAGNOSTIC_ONLY|RETIRED" \
  AGENTS.md CLAUDE.md README.md CHANGELOG.md docs
```

Inspect retired corpus hashes:

```text
rg -n \
  "0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17|ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9|92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4" \
  AGENTS.md docs
```

If a command is unavailable or blocked, report it accurately. Do not classify
it as passed.

## Required final agent report

The final report must include:

- branch, base main commit, commit, and pull request;
- source handling;
- files changed;
- constitutional integration;
- incident disposition;
- evidence;
- independent audit status;
- documentation impact;
- safety confirmations;
- known limitations;
- follow-up recommended.
