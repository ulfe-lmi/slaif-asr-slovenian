# Work Order 0036: S6TTS Hard-Voice Impact Decoder+Joint RNNT Diagnostic

Status: in progress

This bounded diagnostic tests whether adding the newly admitted S6TTS clean and
augmented synthetic views improves hard S6TTS recognition while preserving the
current scale-2000 decoder+joint RNNT behavior on existing directional gates.

The experiment keeps the PR #36 model surface and training budget fixed:

- trainable surface: decoder + joint only;
- frozen: encoder, prompt pathway, tokenizer, adapters;
- objective: audio-conditioned RNNT loss only;
- semantic rows: 16,000;
- total exposures: 320,000;
- effective batch size: 8;
- maximum optimizer steps: 40,000;
- evaluation: directional batch-32 only.

The only intended data change is schedule composition. Four of twenty exposure
rounds are replaced by S6TTS views: one clean S6TTS round and three augmented
S6TTS rounds. This gives S6TTS a controlled 20% exposure share without
increasing the total training budget.

Runtime override: per human instruction during execution, this diagnostic uses
the ADR 0008 ARTUR controller-development partition for aggregate per-round
run-control. The runner saves ignored per-round checkpoints, evaluates
`artur-controller-dev-v1` after each completed exposure round using the
committed batch-1 controller-dev policy, and stops after three evaluated
post-training rounds without a new raw best controller-dev WER. The final
directional suite evaluates only the checkpoint selected by the controller-dev
rule.

Immutable FLEURS-v2 and ARTUR-J gates remain unavailable for early stopping,
checkpoint selection, hyperparameter selection, prompt construction, or
training. They are used only after controller-dev selection for directional
comparison.

This remains `DIAGNOSTIC_ONLY`. It cannot issue `TRAINING_ELIGIBLE`, accept a
checkpoint, publish a model, or authorize public distribution of S6TTS-derived
audio or models.
