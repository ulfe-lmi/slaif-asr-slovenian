# Experiment 0033: Surface08 Scale-8000 OTF ParametricVoiceAug v1

Classification: `PARAMETRIC_VOICEAUG_V1_REAL_GATE_REGRESSION`

This matched diagnostic keeps Experiment 0031's Surface08 model, untouched-base initialization, scale-8000 clean reservoir, optimizer, effective batch, controller policy, directional suite, and 144,000-exposure cap fixed. Only the deterministic OTF augmentation family changes. ParametricVoiceAug is language-agnostic waveform DSP and uses no target identity.

## Result

- Selected round: 5; stopped round: 8 (`three_rounds_without_new_raw_best`).
- Training exposures: 128,000; unique semantic rows: 64,000.
- Aggregate OTF fill rate: 0.994486.
- Selected checkpoint SHA256: `ffe421a671036b0b5bc8a75315c519c50a28da0be17c93e3caca503ce77dd9cd`.
- `accepted_parent` remains `none`; this is noncanonical diagnostic evidence.

## Training

- GPU: NVIDIA GeForce RTX 3090; CUDA 12.6; PyTorch 2.7.1+cu126.
- Precision: fp32; TF32: false; visible GPUs: 1.
- Physical microbatch: 2; accumulation: 4; effective batch: 8.
- Learning rates: decoder 0.0005; joint 0.0005; encoder 5e-06; prompt_kernel 5e-05.
- Trainable parameters: 633,400,352; frozen parameters: 4,596,736.
- Total stage wall time across 2 attempt(s): 13453.452 seconds; training compute: 9395.424 seconds.
- Peak GPU memory: 19470.729 MiB allocated / 20138.000 MiB reserved.
- Parameter integrity: 643 authorized tensors changed; 0 unauthorized tensors changed.
- One attempt was interrupted after round 6 because another process switched the shared Git checkout. The complete round-6 checkpoint and optimizer state were hash-verified and resumed from an isolated worktree.

## Augmentation Families

| Family | Fraction | Parameter range | Tool/backend | License/provenance | Notes |
|---|---:|---|---|---|---|
| Standard OTF | 0.298227 | existing 11 profiles | existing repo DSP | existing | one standard profile per selected sample |
| F0/pitch | 0.324781 | common +/-1-2; rare +/-3 semitones | librosa pitch_shift + soxr | ISC + LGPL-2.1-or-later | deterministic, in memory |
| Speed/tempo | 0.054414 | 0.90-1.10 | SciPy polyphase resampling | BSD-3-Clause | deterministic, in memory |
| Formant/spectral warp | 0.431477 | factor 0.84-1.16 | SciPy STFT log-envelope warp | BSD-3-Clause | deterministic, in memory |
| Aperiodicity/noisiness | 0.268016 | 20-32 dB breathiness SNR | NumPy/SciPy high-pass excitation | BSD-3-Clause | deterministic, in memory |
| Reverb/RIR | 0.109250 | RT60 0.15-0.80 s | repo deterministic synthetic RIR | Apache-2.0 orchestration | deterministic, in memory |
| Bandpass/channel | 0.160969 | mild filters; limited 300-3400 Hz telephone | repo SciPy filters | BSD-3-Clause | deterministic, in memory |
| Codec/companding | 0.107469 | G.711, 12 kHz round-trip, moderate quantization | CPython 3.12 audioop/repo DSP | PSF-2.0 / Apache-2.0 | deterministic, in memory |
| Noise | 0.108813 | 8-25 dB SNR | procedural NumPy/SciPy | BSD-3-Clause | deterministic, in memory |
| Gain/compression/clipping | 0.054852 | -6 to +6 dB; ratio 1.5-3.0 | repo NumPy dynamics | BSD-3-Clause | deterministic, in memory |

| Round range | Standard OTF % | ParametricVoiceAug % | Purpose |
|---|---:|---:|---|
| 1-4 | 50 | 50 | linguistic coverage plus early acoustic pressure |
| 5-9 | 10 | 90 | heavy robustness pressure on repeated rows |

## Dependencies / Provenance

| Tool/dependency | Version/revision | License | Used for | Vendored? | Notes |
|---|---|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause | array, FFT, and deterministic waveform operations | no | pinned project training environment |
| scipy | 1.18.0 | BSD-3-Clause | filtering, convolution, resampling, STFT, and spectral-envelope warping | no | Python wheel in the pinned project training environment |
| librosa | 0.11.0 | ISC | duration-preserving pitch perturbation | no | Python wheel in the pinned project training environment |
| soxr | 1.1.0 | LGPL-2.1-or-later | librosa pitch-shift resampling backend | no | Python wheel dependency of librosa |
| CPython audioop | 3.12 stdlib | PSF-2.0 | existing repository G.711 mu-law and A-law companding | no | CPython standard library |

## Determinism

| Check | Result |
|---|---|
| Same exposure, same worker count | True |
| Same exposure, different worker count | True |
| Restart replay | True |
| Worker-order independence | True |
| Chain replay | True |
| Parameter replay | True |
| Transcript preserved | True |

The local audit covered at least five examples per augmentation family and mild, medium, and strong cases. Automated waveform checks found no destructive cases. Human listening was `NOT_RUN`; it is not reported as passed.

## Controller-Dev Curve

| Round | Step | Exposures | Unique rows | Train loss | Anchor | Scale | ARTUR-dev WER/CER/empty | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | NOT_APPLICABLE | 48.291606 | 47.956826 | 66.467 / 27.409 / 13 | false |
| 1 | 2000 | 16000 | 16000 | 8.503491 | 4.281526 | 6.443006 | 44.169 / 14.705 / 0 | false |
| 2 | 4000 | 32000 | 32000 | 11.049546 | 3.671435 | 5.181891 | 42.29 / 14.415 / 0 | false |
| 3 | 6000 | 48000 | 48000 | 9.483284 | 2.835552 | 4.632549 | 42.418 / 15.322 / 0 | false |
| 4 | 8000 | 64000 | 64000 | 8.668217 | 2.850507 | 4.604993 | 41.393 / 14.314 / 0 | false |
| 5 | 10000 | 80000 | 64000 | 9.076229 | 2.283079 | 4.069123 | 39.94 / 13.987 / 0 | true |
| 6 | 12000 | 96000 | 64000 | 8.505891 | 2.488286 | 3.872386 | 40.41 / 13.973 / 0 | true |
| 7 | 14000 | 112000 | 64000 | 8.233975 | 2.063646 | 3.666893 | 41.435 / 14.372 / 0 | false |
| 8 | 16000 | 128000 | 64000 | 7.790179 | 2.879115 | 3.821623 | 41.051 / 15.482 / 0 | false |

ARTUR controller-dev aggregate metrics alone selected the checkpoint. FLEURS-v2 and ARTUR-J were evaluated only after selection.

## Live OTF Loader

| Round | Examples/s | Audio-sec/s | Fill rate | Consumer wait | Queue p50/p95 | Worker failures | Notes |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 13.516942 | 53.968691 | 0.995300 | 0.470030% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 2 | 13.054402 | 51.836473 | 0.994855 | 0.514475% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 3 | 13.621667 | 54.192401 | 0.995365 | 0.463470% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 4 | 13.374720 | 53.384250 | 0.988346 | 1.165429% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 5 | 13.865230 | 55.236964 | 0.996095 | 0.390471% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 6 | 13.892755 | 54.964921 | 0.996068 | 0.393225% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 7 | 13.538319 | 53.627928 | 0.997793 | 0.220711% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |
| 8 | 14.187889 | 56.408534 | 0.992103 | 0.789709% | 15.0 / 15.0 | 0 | 3 spawned workers; in-memory only |

## Directional Metrics

| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Scale-8000 standard OTF | StrongAug v1 | ParametricVoiceAug v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| piper_synthetic_holdout | 86.025 / 46.762 / 17 | 34.317 / 13.765 / 0 | 23.137 / 7.429 / 0 | 20.497 / 6.112 / 0 | 13.199 / 3.785 / 0 | 11.957 / 3.645 / 0 | 13.043 / 3.673 / 0 |
| supertonic_heldout_voice_holdout | 58.307 / 27.712 / 32 | 14.752 / 4.682 / 0 | 7.842 / 2.145 / 0 | 5.202 / 1.850 / 0 | 4.658 / 1.472 / 0 | 5.202 / 1.850 / 0 | 3.960 / 1.163 / 0 |
| fleurs_v2 | 52.685 / 16.406 / 1 | 46.195 / 15.604 / 0 | 42.084 / 12.985 / 0 | 41.878 / 13.186 / 0 | 39.353 / 12.002 / 0 | 39.904 / 12.343 / 0 | 40.371 / 12.186 / 0 |
| artur_j | 67.322 / 28.620 / 12 | 56.793 / 20.177 / 0 | 47.357 / 14.805 / 0 | 41.765 / 13.553 / 0 | 39.974 / 12.251 / 0 | 39.406 / 12.357 / 0 | 39.318 / 12.017 / 0 |

Values are normalized WER / CER / empty hypotheses.

## Interpretation

- ParametricVoiceAug v1 improved ARTUR-J over standard scale-8000 OTF by 0.656 WER and 0.234 CER absolute.
- It regressed FLEURS-v2 versus standard scale-8000 OTF by 1.018 WER and 0.184 CER absolute. The WER regression exceeds the declared +0.50 tolerance.
- Piper improved by 0.156 WER and 0.112 CER, and Supertonic improved by 0.698 WER and 0.309 CER, versus standard scale-8000 OTF.
- Zero empty hypotheses, stable controller behavior, and valid audio audits rule out label destruction. The result is a domain tradeoff, not a generally better recipe.
- Standard scale-8000 OTF remains the stronger balanced recipe. ParametricVoiceAug should not replace it without a smaller, isolated probability or operation-family follow-up.

## Known Limitations

- Directional batch-32 evaluation is noncanonical.
- Human listening of local audit audio was `NOT_RUN`; automated waveform validation passed.
- The experiment evaluates the combined ParametricVoiceAug family and does not isolate individual operation contributions.
- ARTUR controller-dev is spent development data and cannot provide checkpoint acceptance evidence.
- Training stopped at 128,000 exposures under the declared three-post-best-round rule, below the 144,000 hard cap.
- The run resumed once from a complete round-6 checkpoint after an external shared-worktree switch interrupted worker startup.

## Boundaries

- No real speech, S6TTS, scale-2000 audio, scale-32000, or immutable gate was used for training.
- No target identity, voice-cloning model, downloaded voice model, or external noise corpus was used.
- No augmented WAV was pre-rendered, written, or used for training.
- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.
- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.
