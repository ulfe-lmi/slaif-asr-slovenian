# Third-Party Licenses and Attribution

This file records third-party components used by the local SLAIF Slovenian ASR
pipeline. It does not change the repository code license.

## Repository Code

Original code and documentation in this repository remain Apache-2.0.

## NVIDIA Nemotron 3.5 ASR Streaming

- Source: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Role: external base ASR checkpoint
- License: NVIDIA Open Model Development and Weight License 1.1
- Artifact status: not committed to Git

Derived model artifacts must preserve base-model attribution and license
obligations.

## NVIDIA NeMo

- Source: `https://github.com/NVIDIA-NeMo/NeMo`
- Pinned revision: `8044a3924bfcfe8ef71d792bb73bf274fe853575`
- Role: external ASR runtime and training framework
- License: Apache-2.0
- Source status: local ignored checkout under `.external/NeMo`

## Piper TTS

- Source: `https://github.com/OHF-Voice/piper1-gpl`
- Pinned revision: `b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6`
- Role: external Slovenian TTS executable for local synthetic speech generation
- License: GPL-3.0-or-later
- Source status: local ignored checkout under `.external/piper1-gpl`

Piper is invoked as an external subprocess from `.venv-piper`. The Apache
`slaif_asr` package must not import Piper or include Piper source.

## Piper Voice: sl_SI-artur-medium

- Voice repository: `https://huggingface.co/rhasspy/piper-voices`
- Pinned revision: `217ddc79818708b078d0d14a8fae9608b9d77141`
- Voice: `sl_SI-artur-medium`
- Language: `sl_SI`
- Speaker count: 1
- Quality: medium
- Native sample rate: 22,050 Hz
- Voice trainer: `ppisljar`, as stated by the voice model card
- Artifact status: local ignored files under `.external/piper-voices`

Voice files:

```text
sl/sl_SI/artur/medium/sl_SI-artur-medium.onnx
sl/sl_SI/artur/medium/sl_SI-artur-medium.onnx.json
sl/sl_SI/artur/medium/MODEL_CARD
```

License metadata is inconsistent:

- Hugging Face repository metadata declares MIT.
- The per-voice model card references the source dataset under CC BY 4.0.
- The authoritative CLARIN.SI ARTUR audio record declares CC BY-SA 4.0.

Conservative project policy:

- apply ARTUR CC BY-SA 4.0 attribution and publication policy;
- do not imply endorsement by the speaker, ARTUR authors, Rhasspy, Piper, or
  Open Home Foundation;
- do not publish generated synthetic audio in this PR;
- require later legal review before public or commercial model publication,
  especially for speaker and publicity rights.

Required attribution for generated synthetic audio:

- Open Home Foundation Piper project;
- `rhasspy/piper-voices`;
- voice `sl_SI-artur-medium`;
- voice trainer `ppisljar`;
- ARTUR 1.0 authors and CLARIN.SI handle `11356/1776`;
- applicable license links;
- statement that generated audio was resampled and used as synthetic ASR
  training material.

License links:

- Piper: <https://github.com/OHF-Voice/piper1-gpl>
- Voice repository: <https://huggingface.co/rhasspy/piper-voices>
- Voice model card: <https://huggingface.co/rhasspy/piper-voices/blob/217ddc79818708b078d0d14a8fae9608b9d77141/sl/sl_SI/artur/medium/MODEL_CARD>
- ARTUR handle: <http://hdl.handle.net/11356/1776>
- ARTUR license: <https://creativecommons.org/licenses/by-sa/4.0/>
- Model-card dataset reference: <https://huggingface.co/datasets/ppisljar/artur_studio_tts/>
- Model-card dataset license: <https://creativecommons.org/licenses/by/4.0/>

## GaMS Generator

- Primary source: `cjvt/GaMS3-12B-Instruct`
- Primary pinned revision: `1d0b27af5748784482600d24779409e7e1dc9adc`
- Fallback source: `cjvt/GaMS-9B-Instruct`
- Fallback pinned revision: `292744023fa0b7ccc7ae2c3c885a67468e49fa03`
- Role: external local Slovenian candidate-text generator
- License: Gemma Terms of Use
- Artifact status: not committed to Git
- Runtime environment: repository-local `.venv-gams`
- Loading policy: Transformers, Accelerate, bitsandbytes, 4-bit NF4, double
  quantization, FP16 compute, one visible GPU selected with
  `CUDA_VISIBLE_DEVICES=0`

GaMS model weights remain external supply-chain inputs. They must not be
committed, redistributed, or used with CPU offload, GPU 1, model sharding, or a
floating Hugging Face revision in this project.
