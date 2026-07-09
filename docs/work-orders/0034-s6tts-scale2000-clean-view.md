# Work Order 0034: S6TTS Vintage Slovenian Clean Voice Admission for Scale-2000

Status: in progress

Branch: `exp/s6tts-scale2000-clean-view`

This work order adds S6TTS as a governed external local TTS candidate for one
additional clean synthetic voice view of the existing scale-2000 text corpus.

## Scope

- Pin `ulfe-lmi/s6tts` to
  `6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c`.
  The original work order named
  `b0c7d3fe7e7b0a06e05bf50e61f774a9daa5e8b6`; the human lead explicitly
  authorized updating to the fixed S6TTS revision after the missing-diphone
  bug fix landed upstream.
- Build `s6cli` locally under ignored `.external/s6tts`.
- Synthesize one clean S6TTS view for the fixed 16,000-row scale-2000 text
  corpus when local runtime data is available.
- Validate the generated WAVs as mono signed 16-bit PCM at 16 kHz.
- Commit only privacy-safe aggregate provenance, certificate, and report
  evidence.

## Non-Goals

- No ASR model training.
- No checkpoint acceptance.
- No `TRAINING_ELIGIBLE` certificate.
- No public audio or model release claim.
- No S6TTS source, binaries, runtime data, dictionaries, generated WAVs, local
  manifests, logs, CSV/TSV files, predictions, checkpoints, or local absolute
  paths in Git.

## Runtime Storage

The committed config uses repository-relative `runs/...` tokens for audit.
Execution may redirect those local outputs to NVME by setting
`SLAIF_ASR_RUNS_ROOT` or passing `--runs-root` to the synthesis script. Public
reports must not contain the resolved local absolute path.

## Decision

The final report must classify the result as one of:

- `S6TTS_SCALE2000_CLEAN_VIEW_AUDIO_ACCEPTED`
- `S6TTS_SMOKE_ACCEPTED_FULL_VIEW_BLOCKED`
- `S6TTS_REJECTED_SYNTHESIS_QUALITY`
- `S6TTS_BLOCKED_PROVENANCE`
- `EXPERIMENT_INVALID`
