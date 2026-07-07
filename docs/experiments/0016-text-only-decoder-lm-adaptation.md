# Experiment 0016: Text-only Decoder-LM Adaptation

Classification: `TEXT_ONLY_DECODER_LM_DEGRADES_ASR`

This diagnostic trains only a decoder-side residual LM adapter with a temporary next-token LM head. No audio, TTS, real-gate transcripts, RNNT loss, or acoustic encoder training was used.

## Data
- Corpus: `sl-corpus-v5-scale8000-training-v1`
- Text SHA256: `e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd`
- Train/validation rows: 60800 / 3200

## Training
- Adapter: `sl-si-decoder-lm-adapter-v1`
- Adapter trainable parameters: 165888
- Temporary LM head: training only; excluded from ASR inference
- Initial/final validation loss: 9.47637 / 2.37086
- Initial/final validation perplexity: 13047.74044 / 10.706596

## Directional Metrics
| Split | Base WER/CER | Scale-2000 WER/CER | Scale-8000 clean WER/CER | Text-only WER/CER | Empty base/scale2000/scale8000/text |
|---|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025/46.762 | 55.435/20.073 | 62.267/21.979 | 100.0/100.0 | 17/0/0/96 |
| supertonic_heldout_voice_holdout | 58.307/27.712 | 27.407/7.597 | 28.882/8.004 | 100.0/100.0 | 32/0/0/192 |
| fleurs_v2 | 52.685/16.406 | 51.589/16.238 | 51.268/16.259 | 100.0/100.0 | 1/0/0/834 |
| artur_j | 67.322/28.62 | 60.114/20.63 | 60.856/20.708 | 100.0/100.0 | 12/0/0/256 |

## Decision
- Real-regression burden: 234.967
- Accepted parent: `none`

This report is diagnostic-only. Canonical batch-1 evaluation would be required before any acceptance discussion.
