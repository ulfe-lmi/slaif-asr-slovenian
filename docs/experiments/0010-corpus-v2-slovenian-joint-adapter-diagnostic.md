# Experiment 0010: Corpus-v2 Slovenian Joint-adapter Diagnostic

Status: **completed in PR; pending strategic review**

This diagnostic trains one NeMo-native residual adapter in the frozen RNNT joint hidden layer. The data status is `DIAGNOSTIC_ONLY`; no checkpoint or adapter is accepted as a parent.

## Authorization

- Certificate status: `DIAGNOSTIC_ONLY`
- Certificate SHA256: `8870703a34bd2ae90022ec341ac27b4d1bdb15ddf71dfd53de670f7ca2b75203`
- Adapter config SHA256: `2c970413e7668e79f8e9fd61c2af213ae1095b988c76ec61d02c5be15c605eb7`

## Adapter

- Module: `model.joint`
- Name: `sl-si-joint-adapter-v1`
- Joint hidden dimension: 640
- Trainable parameters: 42240

## Aggregate Metrics

| Split | Base WER/CER | Prompt-column WER/CER | Joint-adapter WER/CER | Empty base/prompt/joint |
|---|---:|---:|---:|---:|
| selected_training | 93.032/61.623 | 69.955/26.405 | 24.253/11.083 | 41/0/0 |
| synthetic_holdout | 84.317/47.295 | 73.137/27.474 | 69.876/29.156 | 17/2/0 |
| fleurs_v2 | 52.703/16.423 | 61.47/20.347 | 64.733/25.541 | 1/0/0 |
| artur_j | 67.453/29.016 | 71.123/25.796 | 73.263/30.333 | 12/0/0 |

## Decision

- Synthetic-holdout gain: `True`
- Prompt-column burden: 16.361
- Joint-adapter burden: 28.275
- Scientific classification: `SL_JOINT_ADAPTER_SYNTHETIC_ONLY`
- Accepted parent: `none`

## Limitations

- Single original Piper voice family.
- No real calibration speech.
- Synthetic holdout is not real-generalization evidence.
- Development gates are not a final blind test.
