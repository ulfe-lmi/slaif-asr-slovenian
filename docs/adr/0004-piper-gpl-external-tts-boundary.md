# ADR 0004: Piper GPL external TTS boundary

- Status: Accepted
- Date: 2026-06-22

## Context

M2 begins the synthetic-data ingestion path with a real Slovenian TTS engine and
voice. The selected engine is `OHF-Voice/piper1-gpl`; the selected initial voice
is `rhasspy/piper-voices` `sl_SI-artur-medium`.

Piper is licensed under GPL-3.0-or-later. The repository code remains
Apache-2.0, so the TTS engine must stay outside the `slaif_asr` package and out
of Git-tracked source.

## Decision

Use Piper only as an external executable dependency:

- clone `OHF-Voice/piper1-gpl` into ignored `.external/piper1-gpl`;
- pin the source revision to `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`;
- install it into repository-local `.venv-piper`;
- invoke it through an argv subprocess with `shell=True` forbidden;
- do not import Piper from the Apache-licensed `slaif_asr` package;
- do not copy Piper source or redistribute a Piper binary in this repository;
- preserve upstream license and source attribution.

Generated audio is synthetic training material produced through a GPL-licensed
external engine and a separately licensed voice. It is not represented as
Apache-2.0 merely because the orchestration code is Apache-2.0.

## Voice license boundary

The selected voice is:

```text
rhasspy/piper-voices
revision 217ddc79818708b078d0d14a8fae9608b9d77141
sl/sl_SI/artur/medium/sl_SI-artur-medium
```

The voice metadata is inconsistent:

- Hugging Face repository metadata declares MIT.
- The per-voice model card references the source dataset under CC BY 4.0.
- The authoritative CLARIN.SI ARTUR audio record declares CC BY-SA 4.0.

This project therefore applies the conservative ARTUR CC BY-SA 4.0 attribution
and publication policy until a later legal review decides otherwise.

## Consequences

Positive:

- the first M2 vertical slice uses the real selected Slovenian TTS;
- GPL code remains outside the Apache package boundary;
- voice artifacts and generated audio remain local ignored runtime artifacts;
- license discrepancy is recorded before any data or model publication.

Costs and risks:

- `.venv-piper` is a separate environment from the NeMo `.venv`;
- ONNX Runtime GPU libraries must be present for CUDA execution;
- public synthetic-audio release is not authorized by this PR;
- final public or commercial model publication requires later license and
  speaker/publicity-rights review.
