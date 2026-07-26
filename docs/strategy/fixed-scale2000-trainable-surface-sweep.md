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
| `SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS` | decoder + joint + final two encoder blocks | Reviewed: acceptable tradeoff in PR #44 |
| `SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS` | decoder + joint + final four encoder blocks | Reviewed: new best directional candidate in PR #45 |
| `SURFACE_07_TOP_ENCODER_PLUS_PROMPT_ACOUSTIC_FUSION` | decoder + joint + final four encoder blocks + proven `prompt_kernel` bridge | Reviewed: new best directional candidate in PR #46 |
| `SURFACE_08_FULL_ENCODER` | decoder + joint + all encoder layers + proven `prompt_kernel`, with frontend and prompt identity frozen | Reviewed: new best directional candidate in Experiment 0029 |
| `SURFACE_09_FULL_MODEL` | full model | Prohibited without future real training data and governance review |

## Review Rule

After Surface04, stop for strategic review. Surface05 is justified only if
Surface04 beats or matches PR #36 with an acceptable tradeoff. If Surface04
regresses, investigate smaller emission/fusion surfaces instead of expanding
further into the encoder.

Surface04 matched PR #36 with an acceptable one-sided tradeoff, so Work Order
0038 authorized Surface05 as the sole Phase 2 experiment. ARTUR controller-dev
selected round 3. The selected checkpoint stayed within the best-known
one-sided real-gate envelope, improved both ARTUR-J metrics, and retained zero
empty hypotheses, yielding
`SURFACE05_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF`. Work Order 0039 authorized
Surface06 as the sole Phase 3 boundary diagnostic. ARTUR controller-dev
selected round 5, and the selected checkpoint established the current
best-known directional envelope on FLEURS-v2 and ARTUR-J with zero empty
hypotheses, yielding `SURFACE06_NEW_BEST_DIRECTIONAL_CANDIDATE`.

Work Order 0040 tested Surface07 as the sole Phase 4 diagnostic. It kept the
Surface06 top-encoder depth fixed and added only the proven separable
`prompt_kernel` fusion bridge. ARTUR controller-dev selected round 13. The
selected checkpoint improved all four directional real-gate metrics versus
Surface06, scoring 42.084/12.985 on FLEURS-v2 and 47.357/14.805 on ARTUR-J
with zero empty hypotheses. This is
`SURFACE07_NEW_BEST_DIRECTIONAL_CANDIDATE`, diagnostic only. The next step is
strategic review and canonical evaluation of named challengers. Experiment
0028 subsequently confirmed Surface07 as the strongest evaluated canonical
challenger. Work Order 0043 now authorizes one exceptional Surface08 boundary
diagnostic to test whether full encoder exposure helps or overfits. ARTUR
controller-dev selected round 6, and the operational rule stopped at round 9
after 18,000 optimizer steps and 144,000 exposures. The selected checkpoint
scored 41.878/13.186 on FLEURS-v2 and 41.765/13.553 on ARTUR-J with zero empty
hypotheses, yielding `SURFACE08_NEW_BEST_DIRECTIONAL_CANDIDATE`. This result
remains diagnostic, does not make full-encoder training the default direction,
and does not authorize Surface09.

## Post-Sweep Data-Axis Diagnostics

Work Order 0045 is a separate, named data-axis follow-up. It keeps the
Surface08 boundary, untouched base initialization, exposure cap, controller
policy, and directional suite fixed while replacing the scale-2000 offline
training schedule with scale-8000 clean synthetic audio plus deterministic
in-memory transcript-preserving augmentation.

The implementation is consolidated with the OTF infrastructure as one
main-targeting integration. It does not alter the completed fixed-scale2000
surface ladder, authorize Surface09, or make scale-8000 full-encoder training a
general policy.

Experiment 0031 completed this diagnostic. ARTUR controller-dev selected round
5 under the predeclared earliest-within-tolerance rule; the raw-best
controller round was 6, and training stopped at round 9 after 144,000 virtual
exposures. The selected checkpoint scored 39.353/12.002 on FLEURS-v2 and
39.974/12.251 on ARTUR-J with zero empty hypotheses, improving all four
directional real-gate WER/CER metrics versus Surface08 scale-2000. Three OTF
workers sustained a 0.994036 aggregate fill rate. The classification is
`SCALE8000_OTF_SURFACE08_NEW_BEST_DIRECTIONAL`; it remains noncanonical,
diagnostic-only evidence and does not accept or promote a checkpoint.

Work Order 0046 authorized one matched StrongAug v1 follow-up to Experiment
0031. Surface08, the untouched base, scale-8000 clean reservoir, optimizer,
effective batch, controller policy, directional suite, and 144,000-exposure
cap remain fixed. Only deterministic OTF augmentation intensity changes:
rounds 1-4 use a 70/30 standard/strong mixture and rounds 5-9 use 25/75. This
is an augmentation-boundary diagnostic, not another surface expansion and not
general authorization for StrongAug training.

Experiment 0032 completed that follow-up. ARTUR controller-dev selected round
6, and training stopped at round 9 after 144,000 virtual exposures. StrongAug
v1 improved ARTUR-J WER to 39.406 but scored 39.904/12.343 on FLEURS-v2,
outside the predeclared tolerance from Experiment 0031 standard OTF. Its
classification is `STRONGAUG_V1_REAL_GATE_REGRESSION`; the standard OTF recipe
remains the stronger scale-8000 data-axis result. No checkpoint is accepted or
promoted.

Work Order 0047 authorized one matched ParametricVoiceAug v1 follow-up to
Experiment 0031. It keeps Surface08, untouched-base initialization, scale-8000
clean data, optimizer, effective batch, controller policy, directional suite,
and the 144,000-exposure cap fixed. The only scientific variable is a
deterministic language-agnostic speech-shape augmentation family covering
bounded pitch, tempo, spectral-envelope, aperiodicity, channel, companding,
room, procedural-noise, and dynamics transforms. Rounds 1-4 use a 50/50
standard/parametric mixture; rounds 5-9 use 10/90. Experiment 0031 standard OTF
is the control. Experiment 0032 StrongAug v1 is a negative comparator, not the
starting recipe or default.

Experiment 0033 completed this follow-up. ARTUR controller-dev selected round
5, and the declared stop rule ended training at round 8 after 128,000 virtual
exposures. ParametricVoiceAug v1 improved ARTUR-J to 39.318/12.017 and improved
both synthetic holdouts versus Experiment 0031, but FLEURS-v2 regressed to
40.371/12.186. The +1.018 FLEURS-v2 WER regression exceeds the predeclared
+0.50 tolerance, so the classification is
`PARAMETRIC_VOICEAUG_V1_REAL_GATE_REGRESSION`. Standard OTF remains the
stronger balanced scale-8000 recipe. No checkpoint is accepted or promoted.

## Strategy Conclusion

- Model surface: keep Surface08 fixed for the next controlled data-axis
  comparison.
- Augmentation: keep Experiment 0031 standard deterministic OTF fixed.
- Exposure budget: keep the Experiment 0031 budget fixed.
- Next changed variable: independently constructed Slovenian text-reservoir
  composition, with broader lexical support and rebalanced frequency.
- Broad StrongAug and ParametricVoiceAug escalation: stop.
- Acoustic mismatch remains a plausible secondary factor. Future acoustic work,
  if separately authorized, should isolate one low-probability operation family
  instead of another compound augmentation policy.

The aggregate vocabulary and frequency analyses make corpus composition the
primary testable hypothesis. They do not prove that lexical or frequency
mismatch is the sole source of ASR error.
