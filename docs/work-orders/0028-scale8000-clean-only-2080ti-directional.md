# Work Order 0028: Scale-8000 Clean-Only 2080 Ti Directional Training

Status: in progress

This work order trains and directionally evaluates the PR #32 scale-8000
clean-only corpus. It uses the accepted 64,000-row text partition and the
576,000 already generated clean Piper/Supertonic views. It does not generate
new text, generate augmentation, use M5/F5 for training, run batch-1 canonical
evaluation, or accept a model.

The scientific comparison is scale-8000 clean-only against the committed
scale-2000 augmented directional evidence. The intended experimental change is
larger accepted semantic text plus clean multi-voice coverage while preserving
the frozen-base Slovenian RNNT joint adapter, optimizer, learning rate,
effective batch size, precision policy, and directional evaluation suite.

Training is authorized only on exactly one visible NVIDIA GeForce RTX 2080 Ti.
The second RTX 2080 Ti may be probed, but it is not used for training. If
microbatch size 8 does not fit, gradient accumulation preserves effective batch
size 8 without changing learning rate or optimizer-step count.

The result remains `DIAGNOSTIC_ONLY`; `TRAINING_ELIGIBLE` is not issued and
`accepted_parent` remains `none`.
