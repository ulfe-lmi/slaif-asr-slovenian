# Experiment 0019: Scale-2000 Decoder+Joint RNNT with ARTUR Controller-Dev Early Stopping

Classification: `ARTUR_EARLYSTOP_SELECTED_CHECKPOINT_DIRECTIONAL_REGRESSES`

This is diagnostic-only evidence. ARTUR controller-dev may be used for aggregate run-control under ADR 0008; immutable gates remain unavailable for checkpoint selection.

Stopped round: `9`. Selected round: `6`.

Operational stop rule: after the human runtime override, training stopped after three further evaluated rounds failed to produce a new raw best ARTUR controller-dev WER. The checkpoint selection rule remained the predeclared earliest-within-tolerance ARTUR controller-dev rule.

## Controller-Dev Early-Stop Curve

| Round | Optimizer step | Exposures seen | Train loss | Synthetic anchor probe | Synthetic scale probe | ARTUR controller-dev WER | CER | Empty | Delete | Insert | Substitute | Selected eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | - | 56.041 | 55.680 | 66.467 | 27.409 | 13 | 0.000 | 0.000 | 0.000 | no |
| 1 | 2000 | 16000 | 15.968 | 11.367 | 14.308 | 61.683 | 24.891 | 0 | 0.000 | 0.000 | 0.000 | no |
| 2 | 4000 | 32000 | 6.848 | 15.153 | 18.937 | 58.095 | 21.431 | 0 | 0.000 | 0.000 | 0.000 | no |
| 3 | 6000 | 48000 | 5.682 | 17.449 | 19.084 | 58.693 | 21.750 | 0 | 0.000 | 0.000 | 0.000 | no |
| 4 | 8000 | 64000 | 7.527 | 18.268 | 20.207 | 57.582 | 21.075 | 0 | 0.000 | 0.000 | 0.000 | no |
| 5 | 10000 | 80000 | 6.576 | 17.249 | 20.284 | 57.369 | 22.308 | 0 | 0.000 | 0.000 | 0.000 | no |
| 6 | 12000 | 96000 | 5.074 | 13.033 | 19.387 | 54.720 | 19.530 | 0 | 0.000 | 0.000 | 0.000 | yes |
| 7 | 14000 | 112000 | 6.029 | 14.571 | 20.941 | 57.070 | 21.496 | 0 | 0.000 | 0.000 | 0.000 | no |
| 8 | 16000 | 128000 | 5.224 | 18.622 | 20.512 | 57.070 | 22.076 | 0 | 0.000 | 0.000 | 0.000 | no |
| 9 | 18000 | 144000 | 6.971 | 14.813 | 18.917 | 55.105 | 21.431 | 0 | 0.000 | 0.000 | 0.000 | no |

## Post-Selection Directional Metrics

| Split | Base directional WER/CER/empty | Scale-2000 joint-adapter WER/CER/empty | PR #36 round-20 decoder+joint WER/CER/empty | Selected early-stop checkpoint WER/CER/empty |
|---|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025/46.762/17 | 55.435/20.073/0 | 34.317/13.765/0 | 44.565/16.428/0 |
| supertonic_heldout_voice_holdout | 58.307/27.712/32 | 27.407/7.597/0 | 14.752/4.682/0 | 18.711/6.196/0 |
| fleurs_v2 | 52.685/16.406/1 | 51.589/16.238/0 | 46.195/15.604/0 | 48.023/15.946/0 |
| artur_j | 67.322/28.62/12 | 60.114/20.63/0 | 56.793/20.177/0 | 57.274/20.375/0 |

## Safety

- No immutable gate may be used for early stopping.
- No raw controller-dev references or hypotheses are included.
- No checkpoint, prediction, local manifest, audio, or model artifact is committed.
- `accepted_parent` remains `none`.
