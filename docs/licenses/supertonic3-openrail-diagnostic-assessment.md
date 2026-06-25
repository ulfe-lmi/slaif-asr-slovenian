# Supertonic 3 OpenRAIL Diagnostic Assessment

Status: diagnostic boundary record

This document records the project boundary for internal Work Order 0023
execution. It is not legal advice.

## Identities

- Code package: `supertonic==1.3.1`
- Code license: MIT
- Model repository: `Supertone/supertonic-3`
- Model revision: `724fb5abbf5502583fb520898d45929e62f02c0b`
- Model name: `supertonic-3`
- Model license: BigScience OpenRAIL-M
- Language used by this diagnostic: `sl`
- Built-in preset styles used: `M1`, `M2`, `M3`, `M4`, `M5`, `F1`, `F2`, `F3`, `F4`, `F5`
- Isolated conversion dependency: `audioop-lts==0.2.2`, used only to preserve
  the existing deterministic Python `audioop.ratecv` fallback in the
  Supertonic environment when SoX is unavailable.
- Execution override: the human required local GPU synthesis on 2026-06-25;
  governed Supertonic execution uses `CUDAExecutionProvider` and does not
  change the publication boundary.

## Diagnostic Scope

The work-order approval authorizes internal generation and evaluation only for
Work Order 0023. The diagnostic tests whether preset Supertonic 3 Slovenian
synthetic voices reduce the synthetic-to-real regression seen with one Piper
voice.

The model license states that it claims no rights in output subject to its
restrictions. A downstream ASR model trained on generated output may fall
within the license definition of a derivative or otherwise remain subject to
OpenRAIL-M use restrictions.

## Publication Boundary

This work order does not authorize:

- distribution or publication of generated Supertonic audio;
- distribution or publication of a trained ASR adapter or checkpoint;
- sale, hosted inference, or API exposure of generated audio or trained
  artifacts;
- model release or public quality claims.

Any later artifact release requires human legal and release review, including
compliance with the applicable OpenRAIL-M restrictions.
