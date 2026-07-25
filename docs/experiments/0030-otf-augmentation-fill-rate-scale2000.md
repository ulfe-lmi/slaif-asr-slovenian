# Experiment 0030: Scale-2000 On-the-Fly Augmentation Fill-Rate Probe

- Classification: `OTF_AUG_FILL_RATE_MARGINAL_PRIMARY_TARGET`
- Status: `DIAGNOSTIC_ONLY`
- Training started: no
- Evaluation started: no
- WER/CER computed: no

## Design

Three spawned CPU workers decoded clean scale-2000 PCM WAVs, selected one of the existing 11 transcript-preserving profiles with a content-keyed HMAC identity, transformed the waveform in memory, and delivered physical microbatches of two through a bounded prefetch queue. The consumer slept at fixed target rates to simulate Surface08-style GPU demand.

- Source clean files: 144000
- Semantic rows: 16000
- Workers: 3 spawned processes
- Queue depth: 16 microbatches
- Augmentation key: `scale2000-otf-transcript-preserving-v1`
- Offline replay: `SCALE2000_OFFLINE_AUGMENTATION_REPLAY_NOT_EXACT`
- Augmented WAVs written: none

## Fill-Rate Results

| Target examples/s | Prepared examples/s | Fill rate | Ready microbatch rate | Underruns | p50 wait ms | p95 wait ms | p99 wait ms | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.8 | 1222.967 | 0.995659 | 0.890625 | 56 | 0.014 | 2.404 | 2.900 | no |
| 16.0 | 1044.133 | 0.994953 | 0.753906 | 126 | 0.422 | 1.937 | 2.235 | no |
| 24.0 | 985.300 | 0.993937 | 0.701172 | 153 | 0.014 | 1.805 | 2.103 | no |
| 32.0 | 1222.153 | 0.991945 | 0.685547 | 161 | 0.017 | 1.715 | 2.032 | no |

Prepared examples/s estimates three-worker transform capacity from aggregate worker compute time. Fill rate is `1 - consumer_wait / total_consumer_time`. A microbatch is ready when the ordered item can be acquired without the queue becoming empty.

## Determinism

| Check | Result |
|---|---|
| Same virtual exposure, same worker count | passed |
| Same virtual exposure after restart | passed |
| Worker order independent | passed |
| Worker count independent | passed |
| Augmentation key recorded | passed |
| Transcript preserved | passed |

## Runtime

- CPU: 12th Gen Intel(R) Core(TM) i9-12900K
- Physical/logical CPU count: 7 / 7
- Storage: kernel-reported rotational block storage
- Available RAM at start: 19.603 GiB
- Python: 3.12.3
- Audio decode: Python wave plus NumPy
- Augmentation backend: existing NumPy/SciPy/audioop transcript-preserving transforms
- Benchmark wall time: 244.860 seconds

## Limitations

- The virtual stream reuses the admitted profile family but does not exactly replay the offline round and source-voice assignment.
- The consumer is a timed CPU simulator; no GPU kernels or model training ran.
- Results describe this host, clean-WAV storage, bounded queue depth, and three-worker configuration.
- SpecAugment is outside this waveform-only fill-rate probe.

## Safety

- No training or model evaluation ran.
- No WER/CER was computed.
- No scale-8000, S6TTS, or real speech was used.
- No generated audio, checkpoints, predictions, raw transcripts, local manifests, or local paths are committed.
