# Training Data Constitution for SLAIF Slovenian ASR

**Status:** Adopted constitutional companion policy
**Version:** 1.0
**Date:** 2026-06-23
**Applies to:** synthetic text, TTS audio, real-speech training data, diagnostic holdouts, curriculum selection, and any experiment that changes model parameters
**Intended repository path:** `docs/training-data-constitution.md`

---

## 1. Purpose and constitutional status

This document defines the non-negotiable rules by which training data for SLAIF Slovenian ASR is proposed, generated, validated, partitioned, synthesized, selected, admitted to training, and interpreted after evaluation.

It exists because a corpus can satisfy a schema, contain no literal duplicate rows, pass a narrow near-duplicate check, and still be scientifically unfit. The failed Slovenian synthetic curriculum demonstrated that data validation must test the *learning signal*, not merely file validity.

This policy is a **constitutional companion** to `AGENTS.md`:

- `AGENTS.md` should contain the short mandatory rules and a requirement to read this document before touching training data.
- This document should contain the detailed doctrine, algorithms, acceptance gates, evidence requirements, and incident record.
- A work order may make a rule stricter, but must not silently weaken this policy.
- Any exception requires an explicit human-approved work order, a written rationale, and a documented effect on scientific claims.
- When this document conflicts with a human-approved active work order, the work order governs only the named experiment. The exception does not silently amend the constitution.

The central rule is:

> **No text, audio, or selected curriculum sample is eligible for model training merely because it is syntactically valid. Training eligibility requires evidence of linguistic quality, structural diversity, partition independence, provenance, and acoustic suitability.**

---

## 2. Normative language

The terms **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative.

- **MUST / MUST NOT**: a hard acceptance rule. Violation blocks the affected stage.
- **SHOULD / SHOULD NOT**: the default rule. Deviation requires a recorded rationale.
- **MAY**: permitted but not required.

A validator result of `passed` means that every required check actually ran and passed. `Skipped`, `not run`, `blocked`, and `unknown` are not passing results.

---

## 3. Executive decisions

The following decisions are effective once this policy is adopted:

1. The Round 1 candidate pool, selected-training set, and synthetic holdout identified below are **retired** from future training, steering, hyperparameter selection, adapter comparison, and promotion decisions.
2. Their historical experiment records remain immutable and auditable.
3. Experiment 0004 remains a valid rejection of its challenger.
4. Experiment 0005 remains valid as an execution and parameter-integrity record, but is **corpus-confounded** and must not be cited as evidence that residual adapters are intrinsically unsuitable.
5. The old synthetic holdout is not an independent generalization set. It may remain a historical synthetic-domain diagnostic only.
6. No new TTS synthesis, ASR scoring, A100 training, or adapter comparison may begin from a newly generated corpus until the corpus has obtained `TRAINING_ELIGIBLE` status under this policy.
7. Synthetic-only improvement is never sufficient for model promotion. Real speech remains decisive.
8. A one-voice synthetic corpus may be used for pipeline validation or bounded overfit proofs, but is not by itself promotion-eligible evidence for real-speech generalization.

---

## 4. Incident record: Slovenian Curriculum Round 1

### 4.1 Affected data

| Artifact | Size | Historical identity |
|---|---:|---|
| Candidate pool | 320 sentences | SHA256 `0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17` |
| Selected training | 160 sentences | training-manifest SHA256 `92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4` |
| Synthetic holdout | 96 sentences | SHA256 `ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9` |

### 4.2 Aggregate findings

| Diagnostic | Candidate pool | Selected training | Synthetic holdout |
|---|---:|---:|---:|
| Rows | 320 | 160 | 96 |
| Rows starting with literal `Skupina` | 320 | 160 | 4 |
| Total occurrences of `postaji` | 640 | 320 | 0 |
| Rows containing `pri postaji` | 320 | 160 | 0 |
| Unique carrier-stripped sentence bodies | 195 | 119 | 89 |
| Rows redundant after carrier stripping | 125 | 41 | 7 |
| Holdout rows whose body appears in candidate pool | — | — | 54 / 96 |
| Holdout rows whose body appears in selected training | — | — | 41 / 96 |
| Distinct artificial holdout tail forms | — | — | 25 |
| Rows with an artificial `pri + adjective + noun` tail | — | — | 96 / 96 |

For this diagnostic:

- candidate bodies were measured after removing the generated wrapper `Skupina N pove, da` and the trailing numbered `pri postaji ...` scaffold;
- holdout bodies were measured after removing the final generated `pri + adjective + noun` tail;
- comparison used the project’s normalized lowercase, punctuation-insensitive text form.

These diagnostic transformations are not proposed as the complete future validator. They expose the structural failure that the v1 validator missed.

### 4.3 The dominant artificial template

Every candidate used the outer form:

```text
Skupina N pove, da <sentence body> pri postaji N pri postaji N.
```

Every selected training sample inherited that form.

The token `postaji` therefore appeared exactly twice per candidate and exactly twice per selected training sample. Numeric row variation made each literal opening and ending look different to the old prefix and suffix counters even though the learning scaffold was the same.

Examples include:

```text
Skupina 1 pove, da ... pri postaji 1 pri postaji 1.
Skupina 2 pove, da ... pri postaji 2 pri postaji 2.
```

The row/group identifier was placed inside the spoken sentence. This is prohibited under the new constitution. Metadata identifiers belong in metadata only.

### 4.4 Holdout contamination by shared template bodies

The synthetic holdout did not use the numbered `Skupina N` wrapper, but it reused many of the same sentence bodies and appended a different artificial tail.

After carrier removal:

- 54 of 96 holdout rows had a body found in the candidate pool;
- 41 of 96 holdout rows had a body found in selected training;
- the latter represents 42.7% of the holdout rows.

The holdout was therefore ID-disjoint but not content-family-disjoint. Different IDs, filenames, random order, or generation calls did not make it independent.

### 4.5 Systematic holdout carrier

All 96 holdout rows ended with one of only 25 forms matching:

```text
pri <adjective> <noun>
```

Examples include forms analogous to:

```text
pri preprost arhivu
pri velik ribniku
pri zanesljiv laboratoriju
```

The adjective forms were not inflected for the locative construction required by `pri`. This made the synthetic tail both repetitive and systematically ungrammatical.

### 4.6 Repeated sentence bodies

The corpus contained many repeated sentence bodies after the artificial wrapper and tail were removed. Examples included bodies repeated four to eight times, such as sentences corresponding to:

```text
Špela želi čisto skodelico za zeliščni čaj.
Na Žalah čuvaj prižge svečo ob mraku.
Stare hiše ob reki niso obnovili brez dovoljenja.
V skladišču čakajo tri škatle in dvanajst map.
Sestanek bo petnajstega marca ob sedmi uri.
```

This repetition reduced the effective corpus size and made a nominally 320-row candidate pool much smaller in learning diversity.

### 4.7 Linguistic defects

The corpus contained pervasive Slovenian grammatical and semantic defects, including:

- adjective–noun agreement errors;
- incorrect case after prepositions;
- incorrect case and preposition selection for place names;
- gender disagreement in pronouns and participles;
- malformed command and question constructions;
- uninflected slot values inserted into inflected sentence frames;
- lowercased personal names where capitalization was expected;
- semantically implausible carrier clauses and tails.

Representative patterns included forms analogous to:

```text
zanesljiv dovolilnico
natančen spremembe
iz Kamniku
za Novem mestu
velik pot
ko se bo Ana vrnil
```

This policy does not assign a complete error count because the corpus did not undergo a sentence-by-sentence native-speaker annotation. The observed defects are nevertheless frequent enough that removing only the outer wrapper would not rehabilitate the corpus.

### 4.8 Why the v1 validator passed

The v1 implementation:

- detected exact duplicates after ordinary ASR normalization;
- measured character 5-gram Jaccard similarity;
- rejected pairs only at a threshold of `0.82` or higher;
- counted the first two literal words;
- counted the final three literal words;
- did not mask row numbers;
- did not derive carrier-stripped or entity-masked skeletons;
- did not test structural overlap between training and holdout;
- did not perform a linguistic-quality gate.

For the affected candidate pool, the most similar pair under the existing character 5-gram implementation reached approximately:

```text
0.8157894736842105
```

That pair consisted of the same sentence body with different group/station numbers. It fell narrowly below the configured `0.82` threshold and was accepted.

The prefix and suffix guards were defeated because:

```text
Skupina 1 != Skupina 2
pri postaji 1 != pri postaji 2
```

under literal token comparison.

The validator therefore produced a technically correct result for its narrow definitions while failing to measure the actual structural redundancy.

### 4.9 Selection amplified the problem

The hard-example selection stage operated only after the malformed and repetitive corpus had already been accepted. It selected examples that the base model found difficult, but difficulty was partly caused by:

- artificial wrappers;
- malformed Slovenian;
- repeated station tails;
- single-voice synthetic acoustics;
- unnatural lexical combinations.

Hard-example mining cannot repair an invalid source pool. It can preferentially select the artifacts that make the pool invalid.

### 4.10 Scientific consequence

The observed experiment pattern is consistent with learning a synthetic template prior:

- selected synthetic training improved strongly;
- residual adapters improved the synthetic holdout;
- the historical FLEURS-v1 component regressed, but that gate is now deprecated
  because it did not preserve unique audio occurrences;
- ARTUR-J regressed and remains unaffected.

This supports the conclusion that the model learned the synthetic corpus distribution rather than robust Slovenian ASR behavior.

It does **not** prove that:

- residual adapters cannot improve Slovenian;
- the selected adapter location is wrong;
- additional parameter capacity is useless;
- synthetic data can never help.

Those architecture-level conclusions require a linguistically sound, structurally diverse, acoustically credible corpus.

---

## 5. Constitutional principles

### 5.1 Quality precedes quantity

A row count is not a diversity measure. A 10,000-row corpus derived from a few sentence frames may contain less useful learning signal than a carefully curated 500-row corpus.

Corpus reports MUST distinguish:

- physical row count;
- unique normalized surface forms;
- unique number-masked forms;
- unique entity-masked forms;
- unique carrier-stripped bodies;
- unique template families;
- effective speaker/voice diversity;
- source-domain diversity.

### 5.2 Identifiers are out-of-band

The following MUST NOT appear in spoken or target text merely to make rows unique:

- row numbers;
- group numbers;
- batch numbers;
- candidate IDs;
- UUIDs;
- station IDs;
- filenames;
- prompt IDs;
- split labels;
- provenance labels.

These values belong in metadata fields.

Naturally occurring numbers are allowed when they are semantically required by the utterance, but they must not encode corpus bookkeeping.

### 5.3 Natural language is the target

Training text MUST be plausible Slovenian that a real speaker could naturally say in the intended domain.

A sentence is not acceptable merely because:

- every token is Slovenian;
- it contains a requested phoneme or morphology category;
- a schema labels it as `ordinary`, `dual`, or `technical`;
- it is long enough;
- it differs from other rows by a name, number, or noun substitution.

### 5.4 Partition independence is structural

Training and holdout partitions MUST be disjoint by more than ID.

They MUST be checked for overlap in:

- normalized surface text;
- number-masked text;
- entity-masked text;
- carrier-stripped text;
- template family;
- source document or source recording;
- speaker or TTS seed family where relevant;
- high-similarity fuzzy clusters.

### 5.5 Partition before synthesis

Text families and source material MUST be assigned to partitions before audio variants are synthesized.

All acoustic variants of the same underlying utterance MUST remain in the same partition. A speed-perturbed or differently voiced version of a training sentence must not appear in holdout.

### 5.6 Synthetic data is supplemental evidence

Synthetic text and audio MAY improve lexical, morphological, and domain coverage. They do not automatically reproduce real-speaker variation, prosody, disfluency, channel effects, or conversational structure.

A synthetic-only holdout measures performance in the synthetic domain. It MUST NOT be described as evidence of real-speech generalization.

### 5.7 Real speech decides promotion

No checkpoint becomes an accepted parent solely because it improves:

- selected synthetic training;
- synthetic holdout;
- TTS-generated evaluation;
- training loss.

Promotion requires the precommitted real-speech gates and all integrity rules.

### 5.8 Validation is fail-closed

If a required quality test cannot run, the corpus is not training-eligible.

A generator or validator MUST NOT silently:

- drop failed rows until a quota is met without reporting rejection reasons;
- lower thresholds to obtain the requested count;
- repair malformed text without recording the transformation;
- substitute a different source or model revision;
- reuse holdout text to fill a candidate shortage;
- proceed to TTS or GPU stages after a hard quality failure.

### 5.9 Generated labels are not evidence

Fields such as `phenomena`, `category`, `intent`, or `domain` are claims. Validators MUST verify the associated property where practical, and human review MUST confirm it when automated verification is unreliable.

### 5.10 Data acceptance is a first-class decision

Every corpus version MUST receive an explicit decision:

```text
DRAFT
TEXT_REJECTED
TEXT_ACCEPTED
AUDIO_REJECTED
AUDIO_ACCEPTED
TRAINING_ELIGIBLE
DIAGNOSTIC_ONLY
RETIRED
```

Only `TRAINING_ELIGIBLE` data may enter model training, unless a work order explicitly defines a bounded pipeline or overfit proof that uses `DIAGNOSTIC_ONLY` data and prohibits quality claims.

---

## 6. Partition architecture

Every experiment MUST name and separate the following roles where applicable.

### 6.1 Candidate source pool

Purpose:

- supply possible training utterances;
- support quality validation and failure-directed selection.

Requirements:

- not yet training data;
- all rows pass text-quality gates before ASR scoring;
- no immutable gate or final-test text;
- no holdout reuse;
- provenance complete.

### 6.2 Selected training partition

Purpose:

- update model parameters.

Requirements:

- selected only from an accepted candidate pool;
- template, source, speaker, and voice concentration caps enforced;
- no holdout, immutable-gate, or blind-test overlap;
- training manifest locked and hashed before training.

### 6.3 Synthetic diagnostic holdout

Purpose:

- measure synthetic-domain fit;
- detect memorization within the chosen synthetic production process.

Requirements:

- independently sourced or generated;
- no shared template family with training;
- no shared underlying sentence body;
- no shared audio variants;
- not used for training, selection, early stopping, or prompt steering;
- never treated as real-generalization evidence.

### 6.4 Real development gates

Purpose:

- decide whether a challenger generalizes to real Slovenian speech.

Requirements:

- immutable after construction;
- reference text never sent to a generator;
- source-recording and speaker boundaries preserved;
- metrics and normalization pinned;
- only aggregate failure information may steer future generation when the work order permits it.

Current canonical real development gates are `fleurs-sl-si-test-full-v2` and
`artur-j-public-gate-v1`. Historical `fleurs-sl-si-test-full-v1` evidence is
deprecated and must not be used as complete-split quality evidence.

### 6.5 Final blind evaluation

Purpose:

- support release claims after development decisions are complete.

Requirements:

- inaccessible to generation, selection, training, threshold tuning, and architecture choice;
- evaluated only under a release-candidate work order;
- retained as blind evidence until the release decision.

### 6.6 Real calibration/training data

Purpose:

- correct synthetic-to-real mismatch using rights-cleared Slovenian speech.

Requirements:

- legally and ethically usable for training;
- disjoint from every real development and blind gate at source-recording and speaker level where feasible;
- partitioned before transcription correction or augmentation decisions that could leak across splits;
- provenance and consent/license status recorded.

---

## 7. Required data model

Each text candidate SHOULD carry at least:

```json
{
  "schema_version": "...",
  "candidate_id": "out-of-band stable ID",
  "language": "sl-SI",
  "spoken_text": "...",
  "target_text": "...",
  "partition_role": "...",
  "source_type": "authentic_text | generated_text | real_speech_transcript",
  "source_id": "...",
  "source_family_id": "...",
  "template_family_id": "... or null",
  "generator": {
    "model": "...",
    "revision": "...",
    "prompt_revision": "...",
    "seed": "..."
  },
  "quality": {
    "surface_hash": "...",
    "number_masked_hash": "...",
    "entity_masked_hash": "...",
    "skeleton_hash": "...",
    "linguistic_review": "pending | accepted | rejected",
    "review_reason_codes": []
  },
  "phenomena": [],
  "domain": "...",
  "license": "..."
}
```

Each audio manifestation SHOULD additionally carry:

```json
{
  "utterance_family_id": "links all acoustic variants of one text",
  "speaker_or_voice_id": "...",
  "tts_engine": "... or null",
  "tts_revision": "... or null",
  "voice_revision": "... or null",
  "synthesis_seed": "... or null",
  "audio_transform_chain": [],
  "sample_rate": 16000,
  "duration_seconds": 0.0,
  "audio_sha256": "...",
  "source_recording_id": "... or null"
}
```

IDs MUST be stable and collision-free, but MUST NOT be injected into the sentence text.

---

## 8. End-to-end admission process

No stage may be skipped merely because later GPU stages are expensive or available.

### Stage 0 — Precommit the data work order

Before generation, the work order MUST specify:

- scientific question;
- intended model surface;
- source types;
- target domains;
- partition roles and sizes;
- generation model and revision, if any;
- linguistic review plan;
- deduplication/fingerprint algorithms;
- acoustic diversity plan;
- rejection thresholds;
- real gates;
- acceptance and rollback criteria;
- artifacts permitted in Git;
- whether the data can support promotion or only diagnostics.

### Stage 1 — Construct a domain and phenomenon plan

A corpus MUST be based on a coverage matrix, not a bag of loosely labeled sentences.

The matrix SHOULD cover the actual intended usage, such as:

- spontaneous and prepared public speech;
- interviews and dialogue;
- ordinary informational prose;
- questions and requests;
- commands;
- names, places, institutions, and inflected entities;
- dates, times, quantities, measurements, and addresses;
- Slovenian dual forms;
- clitics and function words;
- morphology and case government;
- `č`, `š`, and `ž` coverage;
- technical language and carefully governed code-switching;
- hesitations or discourse markers when the target domain contains them;
- short, medium, and long utterances.

Category quotas alone are insufficient. The plan MUST also state expected register, sentence length, domain, and source diversity.

### Stage 2 — Generate or acquire text

Preferred source order:

1. rights-cleared authentic Slovenian transcripts or text from target-like domains;
2. carefully curated authentic Slovenian text that is plausible when spoken;
3. generated Slovenian text used to fill measured coverage gaps.

Generated text MUST be requested as natural standalone utterances. The generator MUST NOT be asked to make rows unique by placing row identifiers, group labels, or batch labels in the text.

The generator SHOULD produce more candidates than required so that validation can reject aggressively without lowering standards.

### Stage 3 — Validate each row

Per-row validation MUST check:

- NFC and whitespace normalization;
- allowed language and characters;
- plausible length;
- no metadata leakage;
- no prohibited gate text or protected hash overlap;
- no unresolved generation instructions;
- no artificial carrier phrase;
- grammatical and semantic plausibility;
- spoken/target mapping correctness;
- valid provenance;
- category plausibility.

### Stage 4 — Compute the fingerprint suite

Every row MUST receive multiple fingerprints described in Section 9. No single near-duplicate metric is sufficient.

### Stage 5 — Validate the collection

The complete pool MUST be analyzed for:

- exact duplicates;
- number-masked duplicates;
- entity-masked duplicates;
- carrier-stripped duplicates;
- template-family concentration;
- repeated openings and endings after masks;
- repeated token n-grams;
- fuzzy near-duplicate clusters;
- source and domain concentration;
- category concentration;
- sentence-length distribution;
- lexical concentration;
- linguistic rejection rate.

### Stage 6 — Validate cross-partition independence

Before any audio is generated, candidate, training, and holdout partitions MUST pass all cross-partition checks in Section 10.

### Stage 7 — Human linguistic acceptance

For bounded experimental corpora up to 5,000 utterances, every training and holdout sentence MUST be reviewed by a competent Slovenian speaker before `TEXT_ACCEPTED` status.

For larger corpora, a human-approved sampling and escalation plan MAY be used, but:

- all automatically suspicious rows still require review;
- the plan must estimate residual error;
- a corpus with systematic generator errors is rejected, not repaired by sampling.

### Stage 8 — Synthesize or ingest audio

Only `TEXT_ACCEPTED` partitions may enter TTS.

Audio generation or ingestion MUST preserve partition and family boundaries. Acoustic variants of a text remain in one partition.

### Stage 9 — Acoustic validation

Every audio item MUST pass:

- sample-rate and format checks;
- duration bounds;
- non-silence checks;
- clipping and amplitude checks;
- TTS/provider success checks;
- transcript/audio family alignment;
- unique audio identity;
- provenance capture.

The collection MUST pass speaker/voice and transform-concentration checks.

### Stage 10 — Issue a data acceptance certificate

A privacy-safe machine-readable certificate MUST be generated and committed before training. It must contain counts, hashes, algorithms, thresholds, review results, overlap results, and the final status.

### Stage 11 — Score and select

ASR difficulty scoring happens only after data acceptance. Selection must preserve quality and diversity constraints.

### Stage 12 — Lock the training manifest

The selected manifest and all relevant hashes MUST be frozen before training begins.

---

## 9. Fingerprint and duplicate-detection suite

The validator MUST use multiple complementary views.

### 9.1 Surface-normalized form

Purpose:

- exact duplicate detection after ordinary punctuation, case, Unicode, and whitespace normalization.

Example:

```text
Ali lahko prideš danes?
ali lahko prideš danes
```

must collide.

### 9.2 Number-masked form

Replace digit sequences and normalized numeric identifiers with placeholders.

Example:

```text
Skupina 1 pove ... postaja 1
Skupina 2 pove ... postaja 2
```

must become the same structural form.

Naturally meaningful numbers may still vary, but number masking is required to expose templates. A collision is reviewed against the declared phenomenon and template family.

### 9.3 Metadata-token form

Mask or reject tokens matching corpus bookkeeping patterns, including:

```text
candidate 17
row 17
group 17
skupina 17
batch 3
sample 0042
station 42 when used only as a row scaffold
```

A metadata-token match in spoken text is a hard failure unless the utterance naturally requires that concept and the work order explicitly documents it.

### 9.4 Entity-masked form

Replace recognized or declared entities with typed placeholders:

```text
<PERSON>
<PLACE>
<ORG>
<DATE>
<TIME>
<QUANTITY>
<DEVICE>
```

This exposes sentence frames that differ only by slot substitution.

A pinned, license-approved Slovenian NLP analyzer MAY assist with lemmatization, morphology, or entity recognition, but the project MUST NOT rely on one analyzer as the only validator.

### 9.5 Carrier-stripped form

The validator MUST detect corpus-wide high-frequency prefixes, suffixes, and wrappers and compute a body after removing candidate carriers.

Carrier detection MUST be data-driven as well as rule-driven. It must not depend only on a known blacklist such as `Skupina N pove, da`.

Signals include:

- an opening or ending shared by an abnormal fraction of rows;
- repeated words surrounding a numeric slot;
- a phrase that carries no domain meaning but appears across categories;
- a prefix/suffix whose removal reveals many exact body duplicates.

### 9.6 Token-shingle similarity

Compute token 2- through 5-gram overlap on at least:

- surface-normalized text;
- number-masked text;
- entity-masked text;
- carrier-stripped text.

### 9.7 Character-shingle similarity

Character n-gram similarity MAY remain as a useful signal, but MUST NOT be the sole near-duplicate test.

Thresholds MUST be calibrated with adversarial fixtures that include:

- identical templates with different numbers;
- identical templates with different names;
- inflectional variants;
- punctuation variants;
- inserted carrier phrases;
- repeated sentence bodies with different tails.

### 9.8 Lemma/POS skeleton

Where tooling permits, compute a morphosyntactic skeleton that reduces content slots while preserving structure.

Conceptual example:

```text
<PERSON> <VERB> <ADJ> <NOUN> <PLACE>
```

The skeleton is a supplementary signal. Slovenian inflection errors may themselves make automatic analysis unreliable, so human review remains required.

### 9.9 Semantic similarity

Embedding or cross-encoder similarity MAY identify paraphrases missed by lexical methods.

Semantic similarity MUST NOT automatically reject all related sentences. It should:

- form review clusters;
- detect holdout/training paraphrase leakage;
- identify low-diversity generated batches;
- supplement, not replace, structural fingerprints.

### 9.10 Template-family clustering

Every candidate MUST either:

- belong to a declared template family; or
- be assigned to a discovered family by the validator.

A template family includes rows that share the same carrier or grammatical frame with slot substitutions.

Controlled minimal pairs are allowed only when:

- they serve a named linguistic purpose;
- the family is declared;
- the family size is capped;
- all members remain in one partition;
- they do not dominate the corpus.

### 9.11 Default concentration policy

The exact numeric thresholds MUST live in a versioned configuration and be precommitted by work order. The default constitutional posture is:

- exact surface duplicates: zero;
- exact carrier-stripped duplicates: zero unless a declared minimal-pair family justifies them;
- exact training/holdout skeleton overlap: zero;
- template families crossing partitions: zero;
- undeclared metadata carriers: zero;
- any single undeclared carrier applying to more than 5% of a generated partition: hard failure;
- any single declared template family exceeding 5% of a generated partition: human approval required and not promotion-eligible by default;
- suspicious fuzzy pairs above the configured threshold: manual review or same-partition clustering;
- thresholds may be tightened for small corpora.

These defaults are intended to prevent another corpus-wide scaffold. They are not permission to optimize up to the limit.

---

## 10. Cross-partition disjointness

### 10.1 Required zero-overlap checks

Training and holdout MUST have zero overlap in:

- candidate ID;
- source item ID;
- audio hash;
- normalized surface hash;
- carrier-stripped body hash;
- number-masked skeleton hash;
- declared template family;
- underlying utterance family;
- source recording where the evaluation design requires recording independence.

### 10.2 Near-overlap review

All high-similarity cross-partition pairs MUST be listed in a local review artifact. Each pair must be:

- rejected;
- moved into the same partition family;
- or accepted with a written reason showing that it is an ordinary language coincidence rather than leakage.

### 10.3 Generator independence is not enough

The following do not prove holdout independence:

- a different random seed;
- a separate generation API call;
- a different candidate ID prefix;
- a different batch number;
- a different superficial tail;
- a different TTS voice;
- shuffling;
- generating the holdout first.

Independence is established by source, structure, family, and similarity evidence.

### 10.4 Variants remain together

The following are one utterance family and MUST remain in one partition:

- the same text spoken by several TTS voices;
- speed-perturbed copies;
- pitch-perturbed copies;
- reverberated or channel-augmented copies;
- punctuation-only transcript variants;
- casing variants;
- minor normalization variants;
- corrected and uncorrected versions of the same source sentence.

### 10.5 Speaker/source isolation

For real speech:

- source recordings SHOULD not cross training and evaluation partitions;
- speakers SHOULD not cross final blind boundaries where speaker-independent claims are intended;
- segments from the same recording MUST remain in one partition;
- near-identical scripted readings by the same speaker SHOULD remain in one family.

---

## 11. Linguistic quality policy

### 11.1 Required review dimensions

Each sentence MUST be judged on:

- grammaticality;
- morphology and agreement;
- case government;
- preposition selection;
- entity inflection;
- semantic plausibility;
- naturalness when spoken;
- register/domain fit;
- category-label correctness;
- pronunciation plausibility for TTS;
- target transcription correctness.

### 11.2 Review outcomes

Allowed outcomes:

```text
ACCEPT
REJECT_GRAMMAR
REJECT_SEMANTICS
REJECT_UNNATURAL
REJECT_TEMPLATE
REJECT_METADATA_LEAK
REJECT_DUPLICATE
REJECT_DOMAIN
REJECT_TRANSCRIPTION
REVISE_AND_REREVIEW
```

A revised sentence is a new reviewed version. It must receive new fingerprints and pass collection-level checks again.

### 11.3 Native-speaker gate

For the project’s current bounded corpus sizes, automated grammar checks are advisory. Native-speaker acceptance is mandatory before training.

The reviewer SHOULD see sentences without generator confidence or category prestige cues where practical, so fluent-looking provenance does not bias acceptance.

### 11.4 Systematic error rule

If review discovers a systematic generator error, generation stops and the prompt/template is repaired. The project MUST NOT manually patch hundreds of rows from a broken template and then treat them as independent generated evidence.

### 11.5 Minimal pairs

Minimal pairs are valuable for phonetic or morphological coverage, but they must be explicitly designed. Random slot substitution is not a minimal-pair methodology.

A valid minimal-pair set states:

- the contrast being tested;
- the linguistic correctness of every member;
- the maximum family size;
- why the set belongs in training;
- why it does not leak into holdout.

---

## 12. Text-generation policy

### 12.1 Generator role

A language model proposes candidates. It does not certify them.

Project code owns:

- schema construction;
- IDs;
- normalization;
- provenance;
- validation;
- rejection;
- partition assignment;
- serialization.

### 12.2 Prompt requirements

A generation prompt MUST:

- request natural standalone Slovenian utterances;
- forbid row/group/batch identifiers in text;
- forbid artificial common wrappers and tails;
- request grammatical inflection rather than uninflected slot insertion;
- state the intended register and domain;
- state length and phenomenon requirements;
- demand lexical and syntactic diversity;
- prohibit copying examples verbatim;
- avoid exposing immutable gate text;
- use only aggregate steering information permitted by the work order.

### 12.3 Overgenerate and reject

The generator SHOULD produce a surplus. The validator should reject low-quality rows rather than relax standards to reach an exact quota.

The generation report MUST state:

- requested rows;
- generated rows;
- accepted rows;
- rejected rows by reason;
- duplicate and template cluster counts;
- re-generation attempts;
- final shortfall, if any.

A shortfall is preferable to a corrupted corpus.

### 12.4 Separate generation strategies

Training and synthetic holdout SHOULD use genuinely separate source strategies, not just separate seeds. Examples include:

- authentic text for holdout and generated text for training;
- independent source collections;
- different domain sources with precommitted roles;
- separately authored human text.

Regardless of strategy, cross-partition validation remains mandatory.

### 12.5 No hidden template factory

Programmatic generation MAY be used only when the templates are explicit, linguistically validated, bounded, and tracked by `template_family_id`.

A programmatic template factory MUST NOT masquerade as a diverse natural-language corpus.

---

## 13. TTS and acoustic-data policy

### 13.1 Single-voice limitation

A single TTS voice produces one narrow acoustic domain. It may be useful for:

- pipeline validation;
- tokenizer and transcript checks;
- micro-overfit proofs;
- controlled lexical diagnostics.

It is not sufficient by itself to support a claim of real-speaker generalization.

A synthetic-only training corpus is promotion-eligible only under an explicit work order that provides credible acoustic diversity and still passes real gates. By default, promotion-oriented training SHOULD include rights-cleared real Slovenian speech or multiple materially distinct voices/speakers.

### 13.2 Diversity dimensions

A promotion-oriented acoustic plan SHOULD consider:

- multiple speakers or TTS voices;
- gender and age diversity where legally and ethically available;
- speaking-rate diversity;
- prosodic diversity;
- utterance-length diversity;
- microphone/channel diversity;
- controlled room/reverberation variation;
- limited noise conditions relevant to deployment;
- spontaneous versus read style;
- public-speech/interview versus informational style.

More variants are not automatically better. Each transform must serve a stated deployment or robustness purpose.

### 13.3 Voice concentration

The data certificate MUST report counts and duration by speaker/voice.

A corpus dominated by one voice MUST be marked accordingly. It MUST NOT be described as multi-speaker because it contains speed or pitch variants of one voice.

### 13.4 Transform provenance

Every acoustic transform MUST be recorded in order, including parameters and tool revision.

Example:

```json
[
  {"transform": "tts", "voice": "...", "revision": "..."},
  {"transform": "resample", "from_hz": 22050, "to_hz": 16000},
  {"transform": "speed", "factor": 0.95},
  {"transform": "room_ir", "id": "..."}
]
```

### 13.5 No cross-partition acoustic leakage

Different voices or transforms of the same sentence do not create independent holdout examples. All variants remain in the same utterance family and partition.

### 13.6 Acoustic augmentation does not fix bad text

Speed perturbation, noise, reverberation, or a second voice cannot rehabilitate malformed, repetitive, or leaked text. Text acceptance always precedes acoustic augmentation.

### 13.7 Real calibration data

A small, rights-cleared, speaker-diverse real Slovenian training partition is strategically preferred over increasing synthetic adapter capacity when the main observed failure is synthetic-to-real transfer.

Real calibration data MUST remain disjoint from real gates and final blind evaluation.

---

## 14. Failure-directed selection policy

### 14.1 Quality before difficulty

Only linguistically and structurally accepted candidates may be scored for ASR difficulty.

A high WER on malformed text is not evidence that the sentence is a valuable training example.

### 14.2 Selection objective

Selection SHOULD balance:

- genuine base-model errors;
- underrepresented linguistic phenomena;
- domain coverage;
- sentence length;
- speaker/voice diversity;
- template-family diversity;
- random controls;
- replay examples needed to prevent catastrophic drift.

### 14.3 Concentration caps survive selection

Selection MUST re-run collection diagnostics. A diverse candidate pool can become a concentrated training set if difficulty mining repeatedly selects one template, one voice, or one malformed phenomenon.

The selected set MUST report:

- family counts;
- voice counts;
- source-domain counts;
- duplicate/skeleton counts;
- overlap with holdout;
- comparison with the full accepted pool.

### 14.4 Empty hypotheses

Empty or highly erroneous base hypotheses MAY indicate valuable examples, but also may indicate:

- broken audio;
- malformed text;
- TTS failure;
- out-of-domain artifacts;
- manifest mismatch.

Such rows require QA before selection.

### 14.5 Controls and replay

Hard-example selection SHOULD include:

- deterministic controls;
- representative easy examples;
- replay from accepted prior data when a parent model exists.

Controls must still satisfy all quality rules.

---

## 15. Data quality gates

### 15.1 Hard text gates

The corpus fails if any of the following occurs:

- unapproved metadata identifiers in spoken text;
- raw immutable-gate or final-test text overlap;
- exact normalized duplicates;
- exact carrier-stripped training/holdout overlap;
- template families crossing training and holdout;
- unresolved unsafe provenance or license;
- unreviewed training text at current bounded scale;
- systematic grammatical generator failure;
- hidden quota filling after rejection;
- missing required fingerprints;
- mismatch between declared and actual row counts or hashes.

### 15.2 Hard audio gates

The corpus fails if any of the following occurs:

- missing or invalid audio;
- duplicate audio paths that overwrite distinct occurrences;
- transcript/audio family mismatch;
- unrecorded TTS or transform provenance;
- cross-partition variants of one utterance family;
- source-recording leakage prohibited by the split design;
- CPU fallback when the work order requires GPU TTS;
- committed raw audio contrary to repository policy.

### 15.3 Scientific eligibility gates

A corpus may pass text and audio validity yet remain `DIAGNOSTIC_ONLY` if:

- it has one synthetic voice;
- its domain is intentionally narrow;
- it is designed for micro-overfit;
- it lacks real calibration;
- its size or coverage is insufficient for promotion-oriented conclusions.

### 15.4 No aggregate-only approval

A summary count such as `0 near duplicates` is not enough. The data certificate must identify:

- algorithms;
- normalizations;
- thresholds;
- cluster distribution;
- worst-case pairs;
- cross-partition results;
- review coverage.

---

## 16. Required data acceptance certificate

Before training, commit a privacy-safe certificate such as:

```text
docs/data-certificates/<corpus-version>.json
```

It MUST include at least:

```json
{
  "schema_version": "1.0",
  "corpus_id": "...",
  "status": "TRAINING_ELIGIBLE",
  "decision_date": "YYYY-MM-DD",
  "source_revisions": {},
  "generator_revisions": {},
  "prompt_revisions": {},
  "partition_counts": {},
  "partition_hashes": {},
  "surface_unique_counts": {},
  "number_masked_unique_counts": {},
  "entity_masked_unique_counts": {},
  "carrier_stripped_unique_counts": {},
  "template_family_counts": {},
  "top_prefixes": [],
  "top_suffixes": [],
  "top_ngrams": [],
  "within_partition_duplicate_counts": {},
  "cross_partition_overlap_counts": {},
  "fuzzy_review_counts": {},
  "linguistic_review": {
    "reviewed": 0,
    "accepted": 0,
    "rejected_by_reason": {}
  },
  "speaker_voice_distribution": {},
  "audio_validation": {},
  "protected_gate_overlap_count": 0,
  "validator_code_revision": "...",
  "config_sha256": "...",
  "reviewer_approval": "recorded out of band or approved repository identity",
  "limitations": []
}
```

The certificate MUST NOT contain:

- raw protected references;
- private transcripts;
- local absolute paths;
- credentials;
- unpublished personal speech;
- generated audio.

---

## 17. Required reports and visual diagnostics

A data PR SHOULD include privacy-safe reports for:

- corpus funnel: generated → row-valid → collection-valid → linguistically accepted → audio-valid → selected;
- sentence-length distribution;
- category/domain distribution;
- source-family distribution;
- template-family size histogram;
- speaker/voice duration distribution;
- top normalized openings and endings;
- top 2–6 token n-grams after number/entity masking;
- nearest-neighbor similarity distribution;
- cross-partition overlap review;
- rejection reasons;
- examples of validator adversarial fixtures, using synthetic test strings rather than protected data.

Reports must make concentration visible. A single scalar diversity score is insufficient.

---

## 18. Validator implementation requirements

The repository SHOULD provide a reusable module rather than experiment-specific ad hoc checks.

Suggested ownership:

```text
slaif_asr/data_quality.py
scripts/validate_training_corpus.py
configs/data_quality/<policy-version>.json
tests/test_data_quality.py
```

Implementation status: the repository now provides the text-stage validator in
`slaif_asr/data_quality.py` and `scripts/validate_training_corpus.py` with
policy `configs/data_quality/training_text_v1.json`. This implements structural
text admission through `TEXT_ACCEPTED`; it does not implement acoustic
validation, does not issue a data acceptance certificate, and cannot emit
`TRAINING_ELIGIBLE`.

The first GaMS corpus-v2 candidate reservoir has been generated as a DRAFT
source pool with a local native-speaker review pack and privacy-safe aggregate
report. It is not `TEXT_ACCEPTED`, no review approval has been fabricated, and
it cannot be used for TTS, ASR scoring, selection, or training until the later
review and certificate stages succeed.

The first review-admission pass has also been implemented. The local edited
review sheet records 415 `ACCEPT` outcomes, but every row lacks the required
`review_revision`. Under this constitution, that is incomplete review metadata,
so the reservoir remains `DRAFT`; its accepted-review sidecar is empty, and no
TTS, scoring, selection, training, or certificate is authorized.

The implementation MUST:

- be deterministic;
- use versioned normalizers;
- fail closed;
- report every enabled check;
- support within- and cross-partition analysis;
- expose machine-readable results;
- avoid quadratic behavior at scales where it becomes impractical, using indexed or approximate candidate retrieval followed by exact scoring;
- preserve privacy and avoid printing protected text by default;
- include adversarial regression tests.

### 18.1 Mandatory regression fixtures

Tests MUST include at least:

1. same template, different row numbers;
2. same template, different names;
3. same body, different artificial suffix;
4. same body, different artificial prefix;
5. training/holdout same body with different IDs;
6. punctuation and casing variants;
7. inflectional variants that remain suspicious;
8. legitimate unrelated sentences sharing common function words;
9. declared minimal-pair family;
10. metadata identifier embedded in speech;
11. acoustic variants of one text crossing partitions;
12. one source recording split into multiple partitions;
13. malformed Slovenian slot insertion examples;
14. threshold-boundary pairs equivalent to the old `0.815789...` failure.

### 18.2 Test design rule

A test is meaningful only if the old validator would have accepted the bad fixture and the new validator rejects or escalates it for review.

---

## 19. Experiment interpretation rules

### 19.1 Data validity and model validity are separate

An experiment can have:

- correct code;
- correct GPU execution;
- correct parameter freezing;
- correct metrics;
- and an invalid or confounded corpus.

Reports MUST separately state:

```text
runtime validity
parameter-integrity validity
data validity
evaluation validity
scientific interpretation
promotion decision
```

### 19.2 Confounded data limits conclusions

When data is later found unfit:

- historical numerical results remain recorded;
- promotion decisions based on real-gate regression remain valid unless the evaluation itself was invalid;
- architecture-level conclusions must be narrowed;
- the corpus is retired;
- no rerun should use the same bad corpus merely to obtain cleaner metrics.

### 19.3 Training loss is not corpus quality

A falling loss can demonstrate capacity to fit the supplied distribution. It does not establish:

- grammaticality;
- diversity;
- independence;
- real-speech transfer;
- acoustic robustness.

### 19.4 Synthetic holdout interpretation

A synthetic holdout may answer:

> Does the update generalize to unseen outputs of a demonstrably independent synthetic production process?

It cannot by itself answer:

> Does the update improve real Slovenian speech recognition?

---

## 20. Disposition of the v1 curriculum

The following corpus identities are permanently retired:

```text
candidate pool:
0c92c60c58d60b629ef275527ed31b7eba5e3eab90fc988928666a121aa86b17

synthetic holdout:
ed10fe7eb49e034d47857a9639a1022d4ad8ab70f6a8c741e6e2b12f1069bec9

selected training manifest:
92b195e2cecb69ee3096ac6644eb65ae592ba60d8cf31d265c45c6eec9d781a4
```

They MUST NOT be used for:

- future training;
- adapter-rank comparison;
- learning-rate selection;
- early stopping;
- generator steering;
- synthetic promotion gates;
- architecture acceptance;
- public corpus-quality claims.

They MAY be used for:

- validator regression fixtures after replacing raw examples with safe minimal reproductions;
- historical audit;
- explaining the failure mode;
- testing that the new quality system rejects equivalent structures.

Historical experiment reports SHOULD receive a prominent note:

> The v1 synthetic curriculum was later found to contain corpus-wide artificial carrier templates, structural train/holdout overlap, and pervasive linguistic errors. Results remain historical execution evidence but do not support conclusions about general Slovenian adaptation capacity.

---

## 21. Required work-order content for future corpus generation

Every corpus-generation work order MUST answer:

1. What scientific gap is the corpus intended to address?
2. Why is synthetic data appropriate for that gap?
3. Which authentic sources are available or unavailable?
4. What domains and registers are targeted?
5. How are train, synthetic holdout, real calibration, real gates, and blind test separated?
6. How are IDs kept out of text?
7. Which fingerprint views are enabled?
8. What are the exact thresholds and review triggers?
9. How is Slovenian linguistic correctness established?
10. What acoustic diversity exists?
11. What makes the synthetic holdout structurally independent?
12. What data status is required before TTS, scoring, and training?
13. What privacy-safe certificate will be committed?
14. What conditions retire the corpus?
15. Which claims remain prohibited even if the experiment succeeds?

A work order that cannot answer these questions is not ready for execution.

---

## 22. Recommended generation strategy for the next corpus

The next promotion-oriented corpus SHOULD follow this order:

1. Repair and freeze evaluation truth independently of training data.
2. Define a target-domain matrix informed only by aggregate real-gate behavior.
3. Acquire a small rights-cleared, speaker-diverse real Slovenian calibration partition if feasible.
4. Source or author natural Slovenian text before using unconstrained templating.
5. Use GaMS or another generator to fill specific measured gaps, not to manufacture the entire corpus through one carrier pattern.
6. Overgenerate substantially and reject aggressively.
7. Run all structural fingerprints before any TTS.
8. Conduct full native-speaker review at the current corpus scale.
9. Synthesize with multiple materially distinct voices or explicitly classify the corpus as acoustically narrow.
10. Keep all variants of one utterance in one partition.
11. Select failures only from the accepted pool while enforcing family and voice caps.
12. Train a conservative frozen-base adaptation first.
13. Evaluate synthetic diagnostics and real gates separately.
14. Promote only on real-speech evidence.

The next corpus should be considered a new scientific object with a new version and certificate. It must not be described as a cleaned continuation of v1.

---

## 23. Research rationale (non-normative)

This policy is principally grounded in the project’s own failure evidence. It is also consistent with established findings that:

- near-duplicate data can increase memorization and contaminate evaluation;
- single lexical similarity methods miss noisy or structurally transformed duplicates;
- TTS-based ASR adaptation has a recognized synthetic-to-real distribution gap;
- low-diversity synthetic speech is not equivalent to real multi-speaker data;
- acoustic augmentation can improve robustness, but does not replace text quality or partition integrity.

Useful background references:

- Lee et al., *Deduplicating Training Data Makes Language Models Better*, arXiv:2107.06499.
- Silcock et al., *Noise-Robust De-Duplication at Scale*, arXiv:2210.04261.
- Park et al., *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, arXiv:1904.08779.
- Su et al., *Task Arithmetic can Mitigate Synthetic-to-Real Gap in Automatic Speech Recognition*, arXiv:2406.02925.
- Liu et al., *Towards Selection of Text-to-speech Data to Augment ASR Training*, arXiv:2306.00998.
- Ogun et al., *An Exhaustive Evaluation of TTS- and VC-based Data Augmentation for ASR*, arXiv:2503.08954.

These references provide rationale, not automatic project requirements. Project policy is governed by this constitution, approved work orders, licenses, and measured Slovenian evidence.

---

# Appendix A — Required `AGENTS.md` insertion

The detailed policy should not be copied wholesale into `AGENTS.md`. Add a concise constitutional section such as the following:

```markdown
## Training-data constitution

Read `docs/training-data-constitution.md` before generating, selecting,
synthesizing, validating, or training on any text or speech data.

Non-negotiable rules:

- Corpus IDs, row numbers, group labels, batch labels, filenames, and provenance
  markers must never be inserted into spoken or target text.
- Data is not training-eligible merely because schema, exact-duplicate, or
  character-ngram checks pass.
- Generated text must pass multi-view structural fingerprints, corpus-level
  concentration analysis, cross-partition family checks, and Slovenian
  linguistic review before TTS or GPU work.
- Training and holdout must be disjoint by normalized text, carrier-stripped
  body, number/entity-masked skeleton, source family, template family, and
  utterance/audio family—not only by ID.
- Every acoustic variant of one underlying utterance remains in one partition.
- Synthetic holdout is diagnostic only. Real speech decides checkpoint
  acceptance.
- A one-voice synthetic corpus may support pipeline or micro-overfit proofs but
  is not by itself evidence of real-speech generalization.
- Hard-example mining may operate only on an already accepted corpus and must
  preserve template, source, domain, and voice diversity.
- A privacy-safe data acceptance certificate with hashes, algorithms,
  thresholds, review coverage, and overlap results is required before training.
- The v1 Round 1 corpora identified in the training-data constitution are
  retired and must not be reused for training, steering, model comparison, or
  promotion.
- Required quality checks that are skipped, blocked, or not run prevent
  `TRAINING_ELIGIBLE` status.
```

---

# Appendix B — Data decision brief template

```markdown
# Data Decision Brief

## Corpus identity
- Corpus ID:
- Version:
- Candidate hash:
- Training-manifest hash:
- Holdout hash:
- Validator revision:
- Quality-config hash:

## Intended use
- Scientific question:
- Training surface:
- Promotion-eligible or diagnostic-only:

## Source and rights
- Sources:
- Licenses/consent:
- Generator model/revision:
- TTS engine/voice revisions:

## Text-quality evidence
- Rows generated:
- Rows accepted:
- Rejections by reason:
- Native-speaker review coverage:
- Surface unique:
- Number-masked unique:
- Entity-masked unique:
- Carrier-stripped unique:
- Template-family distribution:
- Largest family:
- Metadata leakage count:
- Protected-gate overlap count:

## Partition evidence
- Exact train/holdout overlap:
- Skeleton train/holdout overlap:
- Template families crossing partitions:
- Near-overlap pairs reviewed:
- Source-family overlap:
- Utterance-family overlap:

## Acoustic evidence
- Speakers/voices:
- Duration by speaker/voice:
- Transform distribution:
- Invalid audio:
- Duplicate audio paths:
- Source-recording overlap:

## Decision
- Status: DRAFT | TEXT_REJECTED | TEXT_ACCEPTED | AUDIO_REJECTED |
  AUDIO_ACCEPTED | TRAINING_ELIGIBLE | DIAGNOSTIC_ONLY | RETIRED
- Decision reasons:
- Prohibited claims:
- Human approval:
```

---

# Appendix C — Agent report additions for data work

For any data-related work order, the execution agent’s final report MUST add:

```markdown
## Data-quality evidence
- Corpus status:
- Validator code revision:
- Quality-config hash:
- Generated / accepted / rejected counts:
- Rejections by reason:
- Unique counts for every fingerprint view:
- Largest template families:
- Metadata leakage count:
- Protected-gate overlap count:
- Cross-partition exact/skeleton/family overlap counts:
- Fuzzy pairs requiring review:
- Linguistic review coverage and result:
- Speakers/voices and duration distribution:
- Audio validation result:
- Data-certificate path and SHA256:

## Data safety confirmations
- No immutable-gate or blind-test text was sent to a generator.
- No row/group/batch identifier was inserted into spoken text.
- No holdout sample or utterance family entered training or selection.
- No acoustic variant crossed partitions.
- No required quality test was skipped or reported as passed when blocked.
- No rejected corpus proceeded to TTS, scoring, or training.
- No raw corpus, audio, private transcript, or local path was committed.
```

---

# Appendix D — Constitutional review checklist

Before accepting a data PR, the strategic model and human lead should be able to answer **yes** to all applicable questions:

- Is the scientific purpose explicit?
- Is the intended claim no broader than the data can support?
- Are IDs completely out-of-band?
- Is the text natural and linguistically reviewed?
- Did validation use more than one duplicate metric?
- Were numbers and entities masked for structural analysis?
- Were carriers and template families detected?
- Is the holdout structurally independent rather than merely ID-disjoint?
- Are all variants of one utterance in one partition?
- Are source recording and speaker boundaries correct?
- Is acoustic diversity honestly characterized?
- Did hard-example selection operate only after quality acceptance?
- Does the selected set remain diverse?
- Is there a committed privacy-safe certificate?
- Did all required checks actually run?
- Are synthetic metrics clearly separated from real-speech gates?
- Does the PR avoid rehabilitating or reusing the retired v1 corpus?
- Is no GPU/model work being used to compensate for bad data?

If any required answer is no or unknown, the corpus is not ready for training.
