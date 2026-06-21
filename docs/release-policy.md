# Release Policy

## Separate release surfaces

### GitHub

Contains:

- code;
- configuration;
- tests;
- documentation;
- public aggregate evaluation;
- tagged software releases.

### Hugging Face model repository

Contains, after approval:

- adapter/delta or merged checkpoint;
- application/inference instructions;
- model card;
- license and attribution;
- configuration;
- evaluation results;
- hashes and provenance.

### Optional dataset repository

Created only after a separate data-release review.

## Preferred first model release

The preferred first public artifact is an adapter or delta:

```text
ulfe-lmi/slaif-asr-slovenian-nemotron-3.5-adapter
```

Users obtain the NVIDIA base checkpoint separately and apply the SLAIF adaptation.

A merged `.nemo` release is considered later when redistribution and model-license obligations have been reviewed.

## Naming

Use:

> SLAIF Slovenian adaptation of NVIDIA Nemotron 3.5 ASR Streaming

Avoid:

- “SLAIF foundation ASR model”;
- wording that obscures NVIDIA as the base-model provider;
- performance superlatives without protocol-qualified evidence.

## Required model-card content

- base model and exact revision;
- derived artifact type;
- architecture and trainable surface;
- Slovenian prompt and tokenizer policy;
- training-data summary;
- synthetic-data and TTS disclosure;
- active-learning disclosure;
- real-speech evaluation;
- streaming context/latency matrix;
- multilingual regression;
- known limitations;
- intended and out-of-scope uses;
- license and attribution;
- reproduction commands;
- hashes.

## Versioning

Suggested progression:

```text
v0.1 prompt-specific research adapter
v0.2 prompt-kernel adapter
v0.3 emission-adapted checkpoint
v1.0 evaluated Slovenian streaming release
```

A higher version is not automatically better; every model card must identify the selected adaptation stage.

## Release gates

Before public model publication:

- repository release commit is tagged;
- relevant CI and GPU evaluation are green;
- final release configuration is committed;
- checkpoint hashes are recorded;
- model license has been reviewed;
- model card matches the actual artifact;
- data rights and privacy are documented;
- known limitations are public;
- rollback or withdrawal procedure exists;
- human release authority approves.

Execution agents may prepare artifacts but may not publish them without explicit human approval.
