# Evaluation Datasets

This document records real-speech evaluation gates. These are development
gates, not final blind tests.

## Immutable Development Gates

### FLEURS Slovenian Test

- Gate identifier: `fleurs-sl-si-test-full-v2`
- Repository: `google/fleurs`
- Configuration: `sl_si`
- Split: `test`
- Pinned revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
- License: CC BY 4.0
- Rows: 834
- Total duration: 8173.140 seconds
- Manifest SHA256:
  `8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513`
- Reference-manifest SHA256:
  `483961a368bb92cd3fcabe1649cde9655402ff2572bb54eceb26223aaa11ce83`
- Public metadata hash:
  `5f062f6a8afe441eca17ee9bdb5a88d93aaaabfdb92597dec2bac1e4d3af220f`
- Policy: the complete official test split is used. Rows are not sampled or
  excluded for model-performance reasons.
- Identity policy: sample IDs and WAV filenames are derived from deterministic
  source-row enumeration indexes, `0..833`. The upstream FLEURS `source_id` is
  retained as provenance but is not unique and is not used as an identity.

Raw references, hypotheses, local manifests, and audio remain ignored local
artifacts. Committed metadata contains row identifiers, hashes, counts,
durations, and checksums only.

Historical `fleurs-sl-si-test-full-v1` metadata is deprecated because repeated
upstream source IDs caused duplicate sample IDs and WAV overwrites. Its 834
manifest rows represented only 347 unique sample identities, so v1 metrics must
not be used as complete-split quality evidence. See
[`fleurs-sl-si-test-full-v1.deprecated.md`](evaluation-gates/fleurs-sl-si-test-full-v1.deprecated.md).

### ARTUR-J Public Gate

- Gate identifier: `artur-j-public-gate-v1`
- Transcriptions handle: `11356/1772`
- Audio handle: `11356/1776`
- License: CC BY-SA 4.0
- Domain: `Artur-J-Splosni`
- Transcript mode: standardized orthographic transcription, `std`
- Target size: 256 utterances
- Built size: 256 utterances from 64 source recordings
- Total duration: 1049.590 seconds
- Manifest SHA256:
  `66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763`
- Verified archives:
  - `Artur_1.0_TRS.tgz` MD5 `6f21947593ccdea7dc23ecc3c9a7c012`
  - `Artur-J-Audio_00.tar` MD5 `bc8b4e0625fce2b47d99ed7da8db7393`
  - `Artur-J-Audio_01.tar` MD5 `6e4e6684a424d8efeefe1c891536899d`

The ARTUR-J project gate is reproducible but is not an official ARTUR test
split. `Artur-B-Studio`, `Artur-B-Izloceno`, `Artur-N`, `Artur-P`, and
pronunciation-based `pog` transcripts are excluded. `Artur-B-Studio` is excluded
because it is associated with the ARTUR studio voice domain selected for Piper
TTS.

## Future Optional Gate

Common Voice Scripted Speech 26.0 for Slovenian is a desirable third independent
read-speech gate. It is distributed through Mozilla Data Collective under
CC0-1.0, but dataset-term acceptance and download/API access are outside this
work order. No credentials or account setup are required for this PR.

## Leakage Policy

Neither development gate may enter training. Raw references from these gates
must not be sent to GaMS, a project text generator, Piper candidate
generation, or any other synthetic-data generator. Later generation may use only
aggregate categories and metrics explicitly allowed by a work order.
