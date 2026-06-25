# Experiment 0011: Corpus-v2 Supertonic 3 Multi-voice Joint-adapter Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic trains one frozen-base RNNT joint adapter on Supertonic 3 preset voice-style synthetic audio. The data status is `DIAGNOSTIC_ONLY`; no checkpoint or adapter is accepted as a parent.

## Authorization

- Certificate status: `DIAGNOSTIC_ONLY`
- Certificate SHA256: `415888dff4aae5110f36b4dd0a22b564131af4fbb4abbe7739d7a8f0a28ffd66`
- Audio certificate SHA256: `6f0171cfd248f2bb1fce24f7faacf1cf33fd16b88d81c486ff4a2b784abd21e3`

## Synthetic Audio

- Training final WAVs: 1280
- Held-out final WAVs: 192
- Training styles: `M1, M2, M3, M4, F1, F2, F3, F4`
- Held-out styles: `M5, F5`

## Aggregate Metrics

| Split | Base WER/CER | Piper joint WER/CER | Supertonic joint WER/CER | Empty base/Piper/Supertonic |
|---|---:|---:|---:|---:|
| piper_selected_training | 93.032/61.623 | 24.253/11.083 | 62.624/26.183 | 41/0/6 |
| piper_synthetic_holdout | 84.317/47.295 | 69.876/29.156 | 74.224/31.034 | 17/0/1 |
| fleurs_v2 | 52.703/16.423 | 64.733/25.541 | 60.622/20.983 | 1/0/0 |
| artur_j | 67.453/29.016 | 73.263/30.333 | 70.511/26.921 | 12/0/0 |

## Supertonic Diagnostics

- Training-voice probe base/adapter WER/CER: 63.62/28.771 -> 25.973/7.494
- Held-out voice holdout base/adapter WER/CER: 59.705/27.782 -> 50.0/15.125
- M5 held-out base/adapter WER/CER: 61.18/30.838 -> 51.242/17.185
- F5 held-out base/adapter WER/CER: 58.23/24.727 -> 48.758/13.064

## Decision

- Piper holdout gain: `True`
- Supertonic held-out voice gain: `True`
- Piper joint burden: 28.275
- Supertonic joint burden: 15.537
- Burden reduction: 45.050398%
- Scientific classification: `SUPERTONIC3_MULTIVOICE_MITIGATES_PIPER_REGRESSION`
- Accepted parent: `none`

## Limitations

- All training remains synthetic.
- Supertonic preset styles are not real speakers or demographic evidence.
- Supertonic model-license obligations may attach to downstream trained models.
- No real calibration speech exists.
- FLEURS-v2 and ARTUR-J are development gates, not a final blind test.
