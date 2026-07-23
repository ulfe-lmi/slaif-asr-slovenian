# Fixed Scale-2000 Trainable-Surface Sweep

## Fixed Experimental Frame

The program holds the original scale-2000 augmented corpus v4, its 320,000
exposure schedule, ARTUR controller-dev batch-1 run-control, post-selection
directional batch-32 suite, `sl-SI` target, and `[56,3]` context fixed. The
comparator is PR #36 round-20 decoder+joint RNNT.

Each diagnostic changes only one trainable surface. Immutable gates never
select checkpoints. This is a sequence of bounded PR-sized diagnostics, not one
large sweep PR.

## Surface Ladder

| ID | Trainable surface | Program status |
|---|---|---|
| `SURFACE_00_DECODER_JOINT_BASELINE` | decoder + joint | Tested in PR #36 |
| `SURFACE_01_DECODER_JOINT_PLUS_JOINT_PROJECTIONS` | decoder + joint + separable RNNT joint pre/post projections | Planned |
| `SURFACE_02_DECODER_JOINT_PLUS_PREDICTION_DECODER_EXPANDED` | decoder + joint + separable prediction-network internals | Planned |
| `SURFACE_03_DECODER_JOINT_PLUS_PROMPT_ACOUSTIC_FUSION` | decoder + joint + separable post-concat prompt/acoustic fusion bridge | Planned |
| `SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK` | decoder + joint + final encoder block | Reviewed: acceptable tradeoff in PR #43 |
| `SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS` | decoder + joint + final two encoder blocks | Active Phase 2 diagnostic, Work Order 0038 |
| `SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS` | decoder + joint + final four encoder blocks | Requires prior positive Surface04/05 evidence |
| `SURFACE_07_TOP_ENCODER_PLUS_FUSION_COMBINED` | decoder + joint + best top-encoder depth + fusion bridge | Planned |
| `SURFACE_08_FULL_ENCODER` | full encoder | Prohibited under synthetic-only training |
| `SURFACE_09_FULL_MODEL` | full model | Prohibited without future real training data and governance review |

## Review Rule

After Surface04, stop for strategic review. Surface05 is justified only if
Surface04 beats or matches PR #36 with an acceptable tradeoff. If Surface04
regresses, investigate smaller emission/fusion surfaces instead of expanding
further into the encoder.

Surface04 matched PR #36 with an acceptable one-sided tradeoff, so Work Order
0038 authorizes Surface05 as the sole Phase 2 experiment. Surface06 remains
unauthorized until Surface05 evidence is reviewed in a separate strategic
decision.
