# Architecture

## Product category

This project is a **reproducible ASR adaptation and evaluation pipeline**, not a general speech platform and not a fork of NVIDIA NeMo.

Its first purpose is to adapt an existing multilingual, cache-aware streaming recognizer to Slovenian with minimal disruption of transferable behavior.

## Base system

Initial base model:

```text
nvidia/nemotron-3.5-asr-streaming-0.6b
```

Training and inference framework:

```text
NVIDIA NeMo
```

The base model remains an external dependency and must be pinned by revision. The repository records the exact runtime contract after loading the checkpoint rather than rebuilding the architecture from a generic configuration file.

## Logical components

```text
+-----------------------+
| GaMS curriculum       |
| candidate generator   |
+-----------+-----------+
            |
            v
+-----------------------+
| Corpus admission:     |
| text quality,         |
| fingerprints,         |
| partition families,   |
| linguistic review,    |
| data certificate      |
+-----------+-----------+
            |
            v
+-----------------------+
| Slovenian TTS adapter |
+-----------+-----------+
            |
            v
+-----------------------+
| Candidate manifest    |
| and provenance store  |
+-----------+-----------+
            |
            v
+-----------------------+       +----------------------+
| Accepted ASR model    +------>| Streaming evaluator  |
+-----------+-----------+       +----------+-----------+
            |                              |
            |                    failures, margins,
            |                    latency disagreement
            |                              |
            v                              v
+-----------------------+       +----------------------+
| Active selector       |<------+ Error miner          |
+-----------+-----------+       +----------------------+
            |
            v
+-----------------------+
| Bounded fine-tuning   |
| challenger            |
+-----------+-----------+
            |
            v
+-----------------------+
| Acceptance gates      |
| real SL + transfer +  |
| latency + integrity   |
+--------+--------------+
         |
   accept|reject
         v
+-----------------------+
| Next accepted parent  |
| or rollback           |
+-----------------------+
```

## Adaptation ladder

The first implementation must make the trainable surface explicit and measurable.

1. **Slovenian prompt-specific adaptation**
   - modify only the part of prompt conditioning activated by `sl-SI`;
   - strongest transfer-preservation guarantee.
   - first micro-proof implemented the narrowest version: one additive
     first-linear `sl-SI` input-column delta, merged only into that column.
2. **Prompt-projection adaptation**
   - train the small prompt projection;
   - requires multilingual regression gates because the component is shared.
3. **Emission adaptation**
   - train prompt projection, RNNT prediction network, and joint network;
   - keep the FastConformer encoder frozen.
4. **Top encoder layers**
   - permitted only after real-speech evidence shows an acoustic ceiling.
5. **Full fine-tune**
   - reference baseline or later option, never automatic escalation.

Each stage requires its own work order and acceptance criteria.

## Trust boundaries

### Public GitHub repository

May contain:

- source code;
- configurations;
- tests;
- public evaluation summaries;
- documentation;
- hashes and revisions;
- small synthetic text examples without private material.

Must not contain:

- model weights;
- raw speech corpora;
- private transcripts;
- generated training audio;
- credentials;
- local storage paths.

### Execution environment

A disposable GPU VM or cluster allocation may contain:

- downloaded official checkpoint;
- caches;
- generated audio;
- manifests;
- experiment checkpoints;
- temporary logs.

It must not contain production secrets or irreplaceable project truth.

### Model release repository

A separate Hugging Face repository may contain:

- adapter/delta tensors;
- application script;
- merged `.nemo` checkpoint when approved;
- model card;
- evaluation results;
- provenance and license metadata.

### Data release repository

A separate dataset repository may be created only when data licenses, TTS rights, privacy, and redistribution have been reviewed.

## Data partitions

- **Controller development:** can drive error mining and GaMS curriculum.
- **Immutable gate:** used for repeated acceptance; raw reference text is hidden from GaMS. The current gates are `fleurs-sl-si-test-full-v2`, the complete FLEURS Slovenian test split with occurrence-index sample IDs, and the deterministic ARTUR-J public project gate.
- **Final blind test:** used only at major release milestones.
- **Synthetic candidate pool:** generated each round and mostly discarded after selection.
- **Replay reservoir:** balanced memory of previously difficult examples.
- **Multilingual regression:** evaluation-only speech from supported base-model languages.

Before a generated or acquired corpus reaches TTS, ASR scoring, hard-example
selection, or promotion-oriented training, it must satisfy
[`training-data-constitution.md`](training-data-constitution.md). Schema
validity and literal duplicate checks are not enough. The required evidence
includes multi-view structural fingerprints, concentration analysis,
cross-partition family checks, Slovenian linguistic review, and a privacy-safe
data acceptance certificate. Only `TRAINING_ELIGIBLE` data may enter
promotion-oriented training.

## Interface contracts

### Candidate text

Must include:

- ID;
- spoken text;
- target text;
- phenomenon tags;
- source error clusters;
- generation seed;
- generator revision.

### TTS render

Must include:

- candidate ID;
- mono 16 kHz audio path;
- duration;
- exact text;
- TTS revision;
- voice/prosody metadata.

Word timing is not required for RNNT training.

The initial M2 TTS implementation uses `OHF-Voice/piper1-gpl` as an external
GPL-3.0-or-later executable and `rhasspy/piper-voices` `sl_SI-artur-medium` as
the selected Slovenian voice. Piper source, voice artifacts, generated native
22,050 Hz WAVs, resampled 16 kHz WAVs, logs, and local manifests stay in ignored
runtime storage. The Apache-licensed `slaif_asr` package must not import Piper
or vendor its source.

### NeMo manifest

Minimum:

```json
{
  "audio_filepath": "/local/absolute/path.wav",
  "duration": 4.2,
  "text": "Slovenski prepis.",
  "lang": "sl-SI",
  "target_lang": "sl-SI"
}
```

Local paths must never be committed.

### Experiment record

Every challenger records:

- parent checkpoint;
- revisions and hashes;
- trainable surface;
- manifests and hashes;
- hardware and precision;
- commands;
- metrics;
- acceptance decision.

The first M3 prompt-column challenger remains an ignored local artifact. Its
aggregate report is committed, but the delta, merged checkpoint, manifests, and
per-run outputs are not.

The active-curriculum experiment keeps the same prompt-column trainable surface
and adds a pinned external GaMS generator boundary. GaMS runs in `.venv-gams`
with one visible GPU, returns strict JSON, and receives only synthetic
candidate-pool failures plus permitted aggregate gate categories. Fixed
real-gate reference text and synthetic-holdout raw errors remain outside GaMS
prompts.

The canonical real Slovenian development gates are
`fleurs-sl-si-test-full-v2` and `artur-j-public-gate-v1`. Local manifests,
audio, raw references, raw hypotheses, and per-sample predictions remain
ignored; committed artifacts are limited to privacy-safe metadata and aggregate
reports. Historical `fleurs-sl-si-test-full-v1` evidence is deprecated because
non-unique upstream source IDs caused duplicate sample IDs and WAV overwrites.

The same storage boundary applies to project-generated curriculum rounds that
do not use GaMS: the repository owns configuration, validation, selection,
execution code, and aggregate evidence, while generated text/audio/manifests,
raw hypotheses, deltas, and checkpoints remain ignored local artifacts.

The Round 1 v1 curriculum corpora are retired because later review found
corpus-wide artificial carrier templates, structural train/holdout overlap, and
pervasive Slovenian defects. They remain audit material only and must not be
used for future training, model-surface comparison, adapter-rank comparison,
steering, or promotion.

The Slovenian residual-adapter proof adds a project-owned adapter wrapper around
the frozen prompt kernel. Adapter artifacts are saved separately from the base
checkpoint and remain ignored local outputs. Experiment 0005 showed that rank
16 and rank 64 residual adapters improved the fixed synthetic holdout but
regressed both real gates, so neither adapter is an accepted parent.

## Architectural non-goals

The first release will not provide:

- a browser UI;
- a hosted transcription API;
- speaker diarization;
- translation;
- bundled TTS;
- a data-labeling platform;
- automatic public model publication;
- autonomous architecture escalation;
- production deployment infrastructure.

These may be evaluated later through ADRs and separate milestones.
