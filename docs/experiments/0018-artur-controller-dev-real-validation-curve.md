# Experiment 0018: ARTUR Controller-Dev Real Validation Curve

Classification: `ARTUR_CONTROLLER_DEV_READY_CURVE_BLOCKED_CHECKPOINTS_UNAVAILABLE`

This report introduces `artur-controller-dev-v1` as real-acoustic controller-development data. It is development data for aggregate run-control, not an immutable acceptance gate.

No training was run. Per-round PR #36 checkpoints were unavailable locally, so the retrospective curve is blocked until those checkpoints exist.

## Partition

- Partition ID: `artur-controller-dev-v1`
- Rows: 256
- Audio duration seconds: 1045.746
- Manifest SHA256: `7944cbd82107e4aa8cfd3c5ca991d652e4ec3450ba8805efbc98e7c3aeec34f9`
- Reference hash-set SHA256: `5827dfd3afe9b5165eadc101a380aff824c727e9a9a70c078aa5c90a5f9aedf0`
- Audio hash-set SHA256: `65a8c720198b8c1205c0bd4f4c4827e686f789d8b44a6b335471f3780ebcd461`

## Retrospective Curve

| Round | Synthetic anchor probe | Synthetic scale probe | ARTUR controller-dev WER | CER | Empty | Delete | Insert | Substitute | Available |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 56.371 | 53.588 | - | - | - | - | - | - | yes |
| 1 | 12.763 | 14.512 | - | - | - | - | - | - | no |
| 2 | 17.815 | 18.726 | - | - | - | - | - | - | no |
| 3 | 15.380 | 20.172 | - | - | - | - | - | - | no |
| 4 | 21.448 | 19.017 | - | - | - | - | - | - | no |
| 5 | 15.324 | 19.839 | - | - | - | - | - | - | no |
| 6 | 15.459 | 20.397 | - | - | - | - | - | - | no |
| 7 | 15.295 | 22.007 | - | - | - | - | - | - | no |
| 8 | 20.547 | 20.269 | - | - | - | - | - | - | no |
| 9 | 17.913 | 19.746 | - | - | - | - | - | - | no |
| 10 | 13.529 | 15.139 | - | - | - | - | - | - | no |
| 11 | 14.796 | 16.676 | - | - | - | - | - | - | no |
| 12 | 15.027 | 14.984 | - | - | - | - | - | - | no |
| 13 | 13.097 | 15.445 | - | - | - | - | - | - | no |
| 14 | 12.481 | 14.298 | - | - | - | - | - | - | no |
| 15 | 11.193 | 14.967 | - | - | - | - | - | - | no |
| 16 | 12.745 | 14.601 | - | - | - | - | - | - | no |
| 17 | 12.291 | 15.980 | - | - | - | - | - | - | no |
| 18 | 11.888 | 14.485 | - | - | - | - | - | - | no |
| 19 | 11.448 | 14.807 | - | - | - | - | - | - | no |
| 20 | 11.250 | 12.897 | - | - | - | - | - | - | no |

## Early-Stop Rule Status

The rule is encoded in `configs/run_control/artur-controller-dev-early-stop-v1.json`, but no checkpoint selection was made because fewer than two post-training round checkpoints were available.

## Safety

- No immutable gate was used for early stopping.
- No raw references or hypotheses are included.
- No raw audio, predictions, checkpoints, or local manifests are committed.
- `accepted_parent` remains `none`.
