# Nemotron 3.5 ASR Streaming 0.6B: Slovenian Inference and Adaptive Fine-Tuning Plan

- **Base model:** `nvidia/nemotron-3.5-asr-streaming-0.6b`
- **Target locale:** `sl-SI`
- **Hardware:** GPU execution uses exactly one process-visible NVIDIA A100 or
  RTX 2080 Ti. Historical M1/M2 evidence used one RTX 2080 Ti. Current A100
  experiments select physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`, which maps
  to logical `cuda:0`.
- **Training loop:** GaMS generates Slovenian text → external Piper Slovenian TTS renders audio → current ASR model evaluates it → failures are selected → a small update is trained → real and multilingual gates accept or reject the update
- **Prepared:** 2026-06-21

---

## 1. Executive decision

Nemotron 3.5 ASR Streaming is a materially better engineering fit for this project than Voxtral Realtime.

The released model already has:

- an explicit Slovenian language prompt: `sl-SI`;
- a multilingual SentencePiece tokenizer intended to recognize Slovenian text;
- an official NeMo fine-tuning script;
- a companion fine-tuning notebook;
- standard NeMo/Lhotse audio-and-transcript dataloaders;
- ordinary RNNT training loss that performs the speech/text alignment internally;
- native cache-aware streaming inference;
- one checkpoint supporting 80, 160, 320, 560, and 1120 ms operating points;
- a standard `.nemo` output that can be deployed through the same inference path after fine-tuning.

Consequently, this project no longer needs:

- word-level TTS timestamps;
- a custom frame-synchronous target builder;
- special wait/word-boundary target tokens;
- forced alignment;
- a custom decoder-only audio/text sequence merger.

The selected initial Piper Slovenian TTS path only has to produce:

```text
audio waveform + exact Slovenian transcript
```

and the NeMo manifest adds:

```json
"target_lang": "sl-SI"
```

The recommended adaptation ladder is:

1. **Slovenian prompt-column only** — modify only the part of the prompt projection activated by `sl-SI`; preserve the encoder, decoder, joint network, tokenizer, and every other language prompt.
2. **Prompt kernel** — allow the small prompt projection to adapt.
3. **Prompt kernel + RNNT decoder + joint** — adapt emission behavior while keeping the large FastConformer acoustic encoder frozen.
4. **Last encoder layers** — only when real-speech errors demonstrate an acoustic/phonetic ceiling.
5. **Full fine-tune** — retain as the official reference baseline, not the first production choice.

The first prompt-column micro-proof has now been executed on one RTX 2080 Ti.
It supports only the narrow mechanism claim: a 2048-scalar Slovenian prompt
column delta can overfit the tiny synthetic smoke set while preserving exact
parameter isolation. It does not establish a production adaptation or an
accepted parent checkpoint because the diagnostic public real-smoke sample
regressed.

The real development gates now include `fleurs-sl-si-test-full-v2`, the
complete FLEURS Slovenian test split with occurrence-index sample IDs, and the
deterministic ARTUR-J public-speech gate. Historical FLEURS v1 aggregate
metrics are deprecated because v1 did not preserve unique audio occurrences.
The first project-generated curriculum round kept the same 2048-scalar
trainable surface, used Piper synthesis, untouched-base pre-scoring,
deterministic hard-example selection, prompt-column-only training, and fixed
synthetic plus real gates. It was rejected because fixed synthetic-holdout
improvement was insufficient and ARTUR-J regressed; the historical FLEURS-v1
component also regressed but is deprecated. Promotion still requires
non-regression on both real gates; synthetic training-set improvement alone
never accepts a parent.

The first residual-adapter proof reused the same Round 1 corpus and fixed gates
with rank 16 and rank 64 Slovenian-only residual adapters. Both adapters
improved the fixed synthetic holdout, but both regressed ARTUR-J. The
historical FLEURS-v1 component also regressed but is deprecated. No residual
adapter is accepted as a parent. This suggests that increasing prompt-side
capacity against the same single-voice synthetic corpus is not sufficient
evidence for real-speech generalization.

The adaptive loop must never generate a huge static synthetic corpus. Each round should generate a bounded candidate batch, synthesize it, run the current model, select the actual failures, train a small update, and either accept or roll it back.

---

## 2. Verified facts and source status

### 2.1 Primary sources inspected

- [Nemotron 3.5 ASR Streaming model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Pinned model repository revision used in this plan](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/3fc30f3e2ae5d78d462441f3ce89dda694f89bd7)
- [NVIDIA fine-tuning article](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr)
- [NVIDIA Riva companion fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [NeMo source revision inspected here](https://github.com/NVIDIA-NeMo/NeMo/tree/8044a3924bfcfe8ef71d792bb73bf274fe853575)
- [Prompt-aware streaming training configuration](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml)
- [Generic ASR fine-tuning script](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/examples/asr/speech_to_text_finetune.py)
- [Cache-aware streaming inference script](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)
- [Prompt-aware RNNT model implementation](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/nemo/collections/asr/models/rnnt_bpe_models_prompt.py)
- [Prompt-aware Lhotse dataset implementation](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py)
- [NeMo installation README at the inspected revision](https://github.com/NVIDIA-NeMo/NeMo/blob/8044a3924bfcfe8ef71d792bb73bf274fe853575/README.md)

### 2.2 Model contract

| Property | Verified value |
|---|---|
| Architecture | Cache-Aware FastConformer + RNNT + language prompt |
| Parameter count | approximately 600M |
| Encoder | 24-layer cache-aware FastConformer according to the model card |
| Encoder representation | 1024 dimensions |
| Prompt input | 128-way one-hot language vector, repeated across time |
| Prompt fusion | concatenate acoustic representation and language one-hot, then project |
| Decoder | RNNT prediction network + joint network |
| Input sampling rate | 16 kHz |
| Output | punctuated, capitalized text |
| Slovenian status | adaptation-ready |
| Slovenian prompt | `sl-SI` |
| Slovenian prompt index in public config | `62` |
| Streaming points | 80, 160, 320, 560, 1120 ms |
| Model license | OpenMDW 1.1 |
| NeMo code license | Apache 2.0 |
| Model artifact | approximately 2.37 GB `.nemo` file |

### 2.3 Important distinction: checkpoint architecture vs. fine-tuning YAML

The generic training YAML currently contains architecture fields such as `n_layers: 42`, while the model card describes the released checkpoint as a 24-layer encoder.

This is not a reason to rebuild the model from that YAML.

The official `speech_to_text_finetune.py` flow:

1. restores the complete model architecture and weights from the `.nemo` checkpoint;
2. reuses its tokenizer unless explicitly told to replace it;
3. supplies new dataloaders, optimizer settings, SpecAugment, and trainer settings.

Therefore:

> Treat the downloaded `.nemo` checkpoint as the architecture authority. Use the YAML as a fine-tuning/data/optimizer scaffold. Always dump `model.cfg` after loading and never instantiate a fresh model from the generic YAML for this project.

---

## 3. Why the adaptation is simpler than Voxtral

The training batch produced by the prompt-aware Lhotse dataset is:

```text
audio
audio length
transcript token IDs
transcript length
one language-prompt index per sample
```

The model then:

1. extracts acoustic features;
2. runs the FastConformer;
3. concatenates the selected one-hot language prompt;
4. runs the prompt projection;
5. computes RNNT loss from the transcript.

RNNT performs the monotonic speech/text alignment as part of its loss. No word end times are required.

For synthetic data, the minimum valid record is therefore:

```json
{
  "audio_filepath": "/absolute/path/example.wav",
  "duration": 3.82,
  "text": "To je natančen slovenski prepis.",
  "lang": "sl-SI",
  "target_lang": "sl-SI"
}
```

---

# Part I — Fast setup

## 4. Recommended directory layout

```text
sl-nemotron/
├── NeMo/
├── models/
│   └── nemotron35/
│       └── nemotron-3.5-asr-streaming-0.6b.nemo
├── data/
│   ├── real_dev/
│   ├── immutable_gate/
│   ├── multilingual_regression/
│   ├── synthetic/
│   │   ├── round_000/
│   │   └── round_001/
│   └── manifests/
├── scripts/
│   ├── inspect_model.py
│   ├── make_manifest.py
│   ├── validate_manifest.py
│   ├── finetune_selective.py
│   ├── evaluate_outputs.py
│   ├── mine_errors.py
│   └── active_cycle.py
├── configs/
│   ├── active_cycle.yaml
│   ├── gates.yaml
│   └── text_policy.yaml
├── runs/
└── README.md
```

---

## 5. System setup on current development hardware

The inspected NeMo Speech revision requires Python 3.12 or newer and PyTorch 2.7 or newer. The repository baseline uses a local `.venv` with CUDA 12.6 PyTorch wheels on one RTX 2080 Ti. A100 is not a default prerequisite.

### 5.1 System packages

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs ffmpeg sox libsox-fmt-all libsndfile1 \
  build-essential pkg-config curl jq
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

### 5.2 Clone and pin NeMo

```bash
mkdir -p "$HOME/work/sl-nemotron"
cd "$HOME/work/sl-nemotron"

git clone https://github.com/NVIDIA-NeMo/NeMo.git
cd NeMo
git checkout 8044a3924bfcfe8ef71d792bb73bf274fe853575
```

### 5.3 Install the inspected stack

For the repository M1/M2 runtime:

```bash
export CUDA_VISIBLE_DEVICES=0
scripts/setup_runtime_env.sh --recreate
source .venv/bin/activate
```

Verify:

```bash
python - <<'PY'
import torch
import nemo
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("NeMo:", nemo.__version__)
PY
```

If restoration of the trusted NVIDIA `.nemo` checkpoint raises a PyTorch `weights_only` serialization error:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Do not set that variable when loading untrusted checkpoints.

### 5.4 Install small project utilities

```bash
uv pip install huggingface_hub jiwer soundfile librosa orjson pyarrow
```

---

## 6. Download and pin the checkpoint

```bash
cd "$HOME/work/sl-nemotron"
mkdir -p models/nemotron35

hf download \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.nemo \
  --revision 3fc30f3e2ae5d78d462441f3ce89dda694f89bd7 \
  --local-dir models/nemotron35
```

Set paths:

```bash
export PROJECT_ROOT="$HOME/work/sl-nemotron"
export NEMO_ROOT="$PROJECT_ROOT/NeMo"
export MODEL="$PROJECT_ROOT/models/nemotron35/nemotron-3.5-asr-streaming-0.6b.nemo"
```

Record integrity:

```bash
sha256sum "$MODEL" | tee "$PROJECT_ROOT/models/nemotron35/SHA256SUMS"
git -C "$NEMO_ROOT" rev-parse HEAD \
  | tee "$PROJECT_ROOT/models/nemotron35/NEMO_COMMIT"
```

---

## 7. Inspect the actual checkpoint before training

Create `scripts/inspect_model.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
from nemo.collections.asr.models import ASRModel


MODEL = Path("models/nemotron35/nemotron-3.5-asr-streaming-0.6b.nemo")
OUT = Path("runs/contract")
OUT.mkdir(parents=True, exist_ok=True)

model = ASRModel.restore_from(str(MODEL), map_location="cpu")

prompt_dictionary = dict(model.cfg.model_defaults.prompt_dictionary)
assert prompt_dictionary["sl-SI"] == 62
assert prompt_dictionary["sl"] == 62

summary = {
    "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "tokenizer_vocab_size": model.tokenizer.vocab_size,
    "sl_si_prompt_index": prompt_dictionary["sl-SI"],
    "sample_rate": int(model.cfg.sample_rate),
    "encoder_layers": len(model.encoder.layers),
    "encoder_d_model": int(model.cfg.encoder.d_model),
    "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
    "prompt_kernel": repr(model.prompt_kernel),
    "streaming_cfg": str(model.encoder.streaming_cfg),
}

(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(OUT / "model_cfg.yaml").write_text(
    OmegaConf.to_yaml(model.cfg, resolve=True),
    encoding="utf-8",
)

for key, value in summary.items():
    print(f"{key}: {value}")
```

Run:

```bash
cd "$PROJECT_ROOT"
PYTHONPATH="$NEMO_ROOT:$PROJECT_ROOT" \
  "$NEMO_ROOT/.venv/bin/python" scripts/inspect_model.py
```

This runtime report, not a generic YAML, is the contract for all subsequent work.

---

## 8. Audit the tokenizer for Slovenian

Do not replace the tokenizer initially. The official fine-tuning script warns that changing to a different vocabulary size reinitializes decoder components, which would throw away useful transfer.

Add this to `inspect_model.py` or run separately:

```python
texts = [
    "abcčdefghijklmnoprsštuvzž",
    "ABCČDEFGHIJKLMNOPRSŠTUVZŽ",
    "Čez cesto švigne žaba.",
    "Ljubljana, 21. junij 2026.",
    "Zaženi Docker Compose in preveri GPU.",
    "Cena je 12,50 €, temperatura pa 23,7 °C.",
]

for text in texts:
    ids = model.tokenizer.text_to_ids(text)
    decoded = model.tokenizer.ids_to_text(ids)
    print({"text": text, "ids": ids, "decoded": decoded})
```

Required gate:

- `č`, `š`, and `ž` survive encode/decode;
- uppercase Slovenian survives;
- punctuation and apostrophes survive;
- technical code-switching is representable;
- token fertility is recorded.

If normalization changes spacing, define one canonical normalization function and use it for both labels and evaluation. Do not retrain the tokenizer merely because a word is split into several subwords.

---

# Part II — Inference

## 9. Prepare audio correctly

The model expects mono 16 kHz audio.

```bash
ffmpeg -y -i input_audio.ext \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  data/example_16k_mono.wav
```

Check:

```bash
ffprobe -v error \
  -show_entries stream=sample_rate,channels,codec_name \
  -of default=noprint_wrappers=1 \
  data/example_16k_mono.wav
```

---

## 10. Single-file streaming inference

Start at the balanced 320 ms setting:

```bash
cd "$NEMO_ROOT"

uv run python \
  examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path="$MODEL" \
  audio_file="$PROJECT_ROOT/data/example_16k_mono.wav" \
  target_lang=sl-SI \
  att_context_size="[56,3]" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  strip_lang_tags=true \
  compute_dtype=float32 \
  cuda=0 \
  debug_mode=true
```

The five released operating points are:

| `att_context_size` | Chunk/look-ahead operating point |
|---|---:|
| `[56,0]` | 80 ms |
| `[56,1]` | 160 ms |
| `[56,3]` | 320 ms |
| `[56,6]` | 560 ms |
| `[56,13]` | 1120 ms |

Use `[56,0]` for the most demanding low-latency evaluation and `[56,3]` as the initial engineering default.

The public inference script currently requires `compute_dtype=float32` for cache-aware inference. AMP may be tested separately:

```bash
amp=true amp_dtype=float16 compute_dtype=float32
```

First establish a correct FP32 baseline.

---

## 11. Manifest-based inference and WER output

Create `data/manifests/dev_sl.jsonl`:

```json
{"audio_filepath": "/absolute/path/a.wav", "duration": 3.1, "text": "Prvi slovenski stavek.", "lang": "sl-SI", "target_lang": "sl-SI"}
{"audio_filepath": "/absolute/path/b.wav", "duration": 5.4, "text": "Drugi slovenski stavek.", "lang": "sl-SI", "target_lang": "sl-SI"}
```

Run:

```bash
mkdir -p "$PROJECT_ROOT/runs/base_320ms"

cd "$NEMO_ROOT"
uv run python \
  examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path="$MODEL" \
  dataset_manifest="$PROJECT_ROOT/data/manifests/dev_sl.jsonl" \
  output_path="$PROJECT_ROOT/runs/base_320ms" \
  target_lang=sl-SI \
  att_context_size="[56,3]" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  strip_lang_tags=true \
  compute_dtype=float32 \
  batch_size=32 \
  cuda=0
```

The output JSONL contains:

```json
{
  "pred_text": "...",
  "text": "...",
  "wer": 23.4
}
```

Repeat at all five context sizes and retain per-utterance outputs.

---

## 12. Optional non-streaming Python smoke test

The authoritative streaming test is the script above. For quick full-utterance debugging:

```python
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)

model = ASRModel.restore_from(
    "models/nemotron35/nemotron-3.5-asr-streaming-0.6b.nemo"
).cuda().eval()

cfg = RNNTPromptTranscribeConfig(
    batch_size=1,
    target_lang="sl-SI",
)

result = model.transcribe(
    ["data/example_16k_mono.wav"],
    override_config=cfg,
)
print(result)
```

Use the cache-aware inference script for deployment-latency decisions.

---

# Part III — Data construction

All promotion-oriented data construction is now governed by
[`training-data-constitution.md`](training-data-constitution.md). A candidate
record is not training-eligible merely because it has valid JSON, unique IDs,
or no literal duplicates. Before TTS, candidate scoring, hard-example
selection, or model training, the corpus must pass the required structural
fingerprints, concentration analysis, cross-partition family checks, Slovenian
linguistic review, and privacy-safe acceptance-certificate process. Skipped,
blocked, unknown, or unrun quality checks prevent `TRAINING_ELIGIBLE` status.

The text-stage validator is implemented as
[`scripts/validate_training_corpus.py`](../scripts/validate_training_corpus.py)
with policy
[`configs/data_quality/training_text_v1.json`](../configs/data_quality/training_text_v1.json).
It may emit `TEXT_ACCEPTED`, `TEXT_REJECTED`, `DRAFT`,
`DIAGNOSTIC_ONLY`, or `RETIRED`; it cannot emit `TRAINING_ELIGIBLE`.
Promotion-oriented training still requires later acoustic validation and a
privacy-safe data acceptance certificate.

The first GaMS corpus-v2 candidate reservoir has passed an explicit whole-file
human review decision bound to the exact 415-row corpus hash and row count, and
the text validator reports `TEXT_ACCEPTED`. It has also been synthesized with
the external Piper boundary and waveform-validated as `AUDIO_ACCEPTED`.

This does not make the reservoir training-eligible. It remains a single-voice
candidate source pool with no independent synthetic holdout, no selected-
training partition, no partition-level data certificate, no ASR scoring, and no
authorization for hard-example selection or model training.

## 13. Text policy for the first adaptation

The base model is trained to emit punctuation and capitalization. Match that style.

Initially require:

```text
spoken_text == target_text
```

Examples:

```text
Dobro jutro, kako vam lahko pomagam?
Preveri stanje strežnika in nato zaženi novo opravilo.
Čez železniško progo pelje šest tovornjakov.
```

Do not combine lexical adaptation and written-form normalization in the first experiment. Introduce these later as separately tagged tasks:

```text
spoken: dvaindvajsetega junija ob devetih in trideset
target: 22. junija ob 9.30
```

---

## 14. GaMS candidate schema

GaMS should emit structured candidates:

```json
{
  "candidate_id": "r003-c00128",
  "spoken_text": "Čez železniško progo pelje šest tovornjakov.",
  "target_text": "Čez železniško progo pelje šest tovornjakov.",
  "phenomena": [
    "diacritic:č",
    "diacritic:š",
    "lexicon:transport"
  ],
  "source_error_clusters": [
    "c_to_caron",
    "s_to_caron"
  ],
  "minimal_pair_group": "mp-1021",
  "difficulty": 0.72,
  "seed": 314159
}
```

Validate before TTS:

- valid UTF-8 and Unicode normalization;
- Slovenian language check;
- exact and near-duplicate removal;
- no test-set sentence reuse;
- no malformed markup;
- no excessive repetition;
- bounded length;
- target/spoken consistency;
- phenomenon tags actually occur in the text.

---

## 15. TTS output contract

The TTS wrapper only needs to return:

```json
{
  "candidate_id": "r003-c00128",
  "audio_filepath": "/absolute/path/r003-c00128.wav",
  "duration": 4.22,
  "text": "Čez železniško progo pelje šest tovornjakov.",
  "lang": "sl-SI",
  "target_lang": "sl-SI",
  "tts_model": "old-sl-tts",
  "voice": "voice-1",
  "tts_parameters": {},
  "generation_seed": 314159
}
```

Word timing is not required.

Render to mono 16 kHz PCM WAV before training. Reject:

- empty files;
- clipped or NaN waveforms;
- implausible duration;
- silence-only output;
- synthesis failures;
- text/audio mismatches found by the current recognizer or a secondary check.

The M2 ingestion slice uses `OHF-Voice/piper1-gpl` revision
`b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6` and `rhasspy/piper-voices`
`sl_SI-artur-medium` revision `217ddc79818708b078d0d14a8fae9608b9d77141`.
Piper remains an external GPL executable in `.venv-piper`; voice artifacts,
native 22,050 Hz WAVs, resampled 16 kHz WAVs, logs, provenance, and manifests
remain ignored local artifacts. Future 2080 Ti training should use FP16 AMP
rather than BF16 unless a later work order changes that policy. The first
prompt-column micro-proof records an explicit FP32 fallback after FP16 AMP
produced loss-scale overflow events during the one-sample proof.

---

## 16. Manifest builder

Create `scripts/make_manifest.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.metadata.open("r", encoding="utf-8") as src, \
         args.output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            row = json.loads(line)
            audio_path = Path(row["audio_filepath"]).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"{audio_path} at line {line_number}")

            info = sf.info(str(audio_path))
            if info.channels != 1:
                raise ValueError(f"{audio_path}: expected mono, got {info.channels}")
            if info.samplerate != 16000:
                raise ValueError(
                    f"{audio_path}: expected 16000 Hz, got {info.samplerate}"
                )

            text = row["text"].strip()
            if not text:
                raise ValueError(f"Empty transcript at line {line_number}")

            manifest = {
                "audio_filepath": str(audio_path),
                "duration": info.frames / info.samplerate,
                "text": text,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
            }
            dst.write(json.dumps(manifest, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

---

# Part IV — Establish the official full-fine-tuning baseline

## 17. Why run one full baseline

NVIDIA's released recipe is a straightforward full fine-tune. It is useful as a ceiling/reference, even if the final production strategy trains fewer parameters.

Run it on a small controlled dataset first so that you can compare:

```text
prompt-column only
prompt kernel
prompt + decoder + joint
full fine-tune
```

using the same samples and step budget.

---

## 18. Official-style smoke fine-tune

Assume:

```bash
export TRAIN_MANIFEST="$PROJECT_ROOT/data/manifests/train_sl.jsonl"
export VAL_MANIFEST="$PROJECT_ROOT/data/manifests/dev_sl.jsonl"
```

Run a short smoke test:

```bash
cd "$NEMO_ROOT"

uv run python examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model="$MODEL" \
  ++model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
  ++model.validation_ds.manifest_filepath="$VAL_MANIFEST" \
  ++model.train_ds.default_prompt_mode=langID \
  ++model.train_ds.unified_auto_ratio=0.0 \
  ++model.validation_ds.default_prompt_mode=langID \
  ++model.validation_ds.unified_auto_ratio=0.0 \
  ++model.optim.sched.d_model=1024 \
  ++trainer.devices=1 \
  ++trainer.num_nodes=1 \
  ++trainer.max_epochs=-1 \
  ++trainer.max_steps=50 \
  ++trainer.limit_train_batches=1.0 \
  ++trainer.val_check_interval=25 \
  ++trainer.precision=bf16 \
  ++model.train_ds.batch_duration=200 \
  ++model.optim.name=adamw \
  ++model.optim.lr=0.1 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.warmup_steps=20 \
  ++exp_manager.version=smoke_full \
  ++exp_manager.use_datetime_version=false \
  ++exp_manager.exp_dir="$PROJECT_ROOT/runs"
```

The learning-rate value above follows NVIDIA's companion notebook style; it is a starting baseline, not a claim of Slovenian optimality.

Pass conditions:

- checkpoint restoration succeeds;
- `sl-SI` prompt is accepted;
- training loss is finite;
- validation WER runs;
- a `.nemo` checkpoint is written;
- the saved checkpoint can run the streaming inference script.

For a real experiment, use a fixed step budget and increase `max_steps`; do not rely on epochs for continuously regenerated iterable data.

---

## 19. Evaluate the resulting checkpoint

Expected output path resembles:

```text
runs/FastConformer-Transducer-BPE-Prompt-Streaming/
  smoke_full/checkpoints/
    FastConformer-Transducer-BPE-Prompt-Streaming.nemo
```

Evaluate:

```bash
export FT_MODEL="$PROJECT_ROOT/runs/FastConformer-Transducer-BPE-Prompt-Streaming/smoke_full/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo"

cd "$NEMO_ROOT"
uv run python \
  examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path="$FT_MODEL" \
  dataset_manifest="$VAL_MANIFEST" \
  output_path="$PROJECT_ROOT/runs/smoke_full_eval" \
  target_lang=sl-SI \
  att_context_size="[56,3]" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  strip_lang_tags=true \
  compute_dtype=float32 \
  batch_size=32 \
  cuda=0
```

---

# Part V — Preserve transferable knowledge with selective training

## 20. The Slovenian prompt pathway

The prompt mechanism is:

```text
encoder output: [B, T, 1024]
language prompt: [B, T, 128] one-hot
concatenate:    [B, T, 1152]
Linear:         1152 → 2048
ReLU
Linear:         2048 → 1024
RNNT decoder/joint
```

The Slovenian prompt index is obtained at runtime:

```python
sl_index = model.cfg.model_defaults.prompt_dictionary["sl-SI"]
```

The first prompt projection receives acoustic features in its first 1024 columns and language one-hot values in its last 128 columns.

Therefore, the Slovenian-specific input column is:

```python
column = encoder_hidden_size + sl_index
```

For the released configuration:

```text
1024 + 62 = 1086
```

Do not hardcode that number in production code; derive it from the checkpoint.

---

## 21. Stage S0: update only the Slovenian prompt column

This is the strongest preservation experiment.

Trainable:

```text
only W[:, encoder_hidden + sl_prompt_index]
inside prompt_kernel[0].weight
```

Frozen:

```text
preprocessor
FastConformer encoder
all acoustic columns of prompt projection
all other language-prompt columns
prompt-projection bias
second prompt-projection layer
RNNT decoder
RNNT joint
tokenizer
```

For every non-Slovenian prompt, this update is inactive because its one-hot input at the Slovenian column is zero.

Important:

> Set optimizer weight decay to zero in this stage. AdamW weight decay on the containing full matrix would otherwise alter columns whose gradients were masked to zero.

---

## 22. Selective fine-tuning patch

Copy the official script:

```bash
mkdir -p "$PROJECT_ROOT/scripts"
cp \
  "$NEMO_ROOT/examples/asr/speech_to_text_finetune.py" \
  "$PROJECT_ROOT/scripts/finetune_selective.py"
```

Add these imports:

```python
import types
import torch
```

Add the following helper before `main`:

```python
def _set_trainable(module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _trainable_only_param_groups(self) -> None:
    params = [p for p in self.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters remain after applying freeze policy")
    self._optimizer_param_groups = [{"params": params}]


def apply_selective_stage(
    model,
    stage: str,
    target_prompt: str = "sl-SI",
    encoder_layers_to_unfreeze: int = 0,
) -> None:
    # Freeze all pretrained parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if not hasattr(model, "prompt_kernel"):
        raise TypeError(
            f"{type(model).__name__} does not expose prompt_kernel; "
            "verify that the prompt-aware checkpoint was loaded."
        )

    prompt_dict = dict(model.cfg.model_defaults.prompt_dictionary)
    if target_prompt not in prompt_dict:
        raise KeyError(f"Unknown prompt {target_prompt!r}")

    if stage == "prompt_column":
        first_linear = model.prompt_kernel[0]
        weight = first_linear.weight
        enc_hidden = int(model.cfg.model_defaults.enc_hidden)
        prompt_index = int(prompt_dict[target_prompt])
        column = enc_hidden + prompt_index

        if column >= weight.shape[1]:
            raise RuntimeError(
                f"Prompt column {column} outside weight shape {tuple(weight.shape)}"
            )

        weight.requires_grad = True

        def keep_only_target_column(gradient, selected_column=column):
            masked = torch.zeros_like(gradient)
            masked[:, selected_column] = gradient[:, selected_column]
            return masked

        weight.register_hook(keep_only_target_column)

    elif stage == "prompt_kernel":
        _set_trainable(model.prompt_kernel, True)

    elif stage == "emission":
        _set_trainable(model.prompt_kernel, True)
        _set_trainable(model.decoder, True)
        _set_trainable(model.joint, True)

    elif stage == "top_encoder":
        _set_trainable(model.prompt_kernel, True)
        _set_trainable(model.decoder, True)
        _set_trainable(model.joint, True)

        if encoder_layers_to_unfreeze <= 0:
            raise ValueError("encoder_layers_to_unfreeze must be positive")
        layers = list(model.encoder.layers)
        if encoder_layers_to_unfreeze > len(layers):
            raise ValueError(
                f"Requested {encoder_layers_to_unfreeze} layers, "
                f"but encoder has {len(layers)}"
            )
        for layer in layers[-encoder_layers_to_unfreeze:]:
            _set_trainable(layer, True)

    elif stage == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True

    else:
        raise ValueError(f"Unknown selective stage: {stage}")

    # Ensure NeMo constructs an optimizer containing only trainable tensors.
    model.setup_optimizer_param_groups = types.MethodType(
        _trainable_only_param_groups,
        model,
    )

    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    total_trainable = sum(size for _, size in trainable)

    print(f"Selective stage: {stage}")
    print(f"Target prompt: {target_prompt} -> {prompt_dict[target_prompt]}")
    print(f"Trainable tensors: {len(trainable)}")
    print(f"Trainable parameters: {total_trainable:,}")
    for name, size in trainable:
        print(f"  {name}: {size:,}")
```

In `main`, insert this immediately after dataloader setup and before optimizer setup:

```python
stage = cfg.get("selective_stage", "full")
target_prompt = cfg.get("target_prompt", "sl-SI")
encoder_layers_to_unfreeze = int(
    cfg.get("encoder_layers_to_unfreeze", 0)
)

apply_selective_stage(
    asr_model,
    stage=stage,
    target_prompt=target_prompt,
    encoder_layers_to_unfreeze=encoder_layers_to_unfreeze,
)
```

The placement must be:

```python
asr_model = setup_dataloaders(asr_model, cfg)

apply_selective_stage(...)

asr_model.setup_optimization(cfg.model.optim)
```

---

## 23. Run prompt-column-only adaptation

```bash
cd "$NEMO_ROOT"

PYTHONPATH="$NEMO_ROOT:$PROJECT_ROOT" \
uv run python "$PROJECT_ROOT/scripts/finetune_selective.py" \
  --config-path="$NEMO_ROOT/examples/asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model="$MODEL" \
  +selective_stage=prompt_column \
  +target_prompt=sl-SI \
  ++model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
  ++model.validation_ds.manifest_filepath="$VAL_MANIFEST" \
  ++model.train_ds.default_prompt_mode=langID \
  ++model.train_ds.unified_auto_ratio=0.0 \
  ++model.validation_ds.default_prompt_mode=langID \
  ++model.validation_ds.unified_auto_ratio=0.0 \
  ++model.optim.sched.d_model=1024 \
  ++trainer.devices=1 \
  ++trainer.max_epochs=-1 \
  ++trainer.max_steps=250 \
  ++trainer.val_check_interval=50 \
  ++trainer.precision=bf16 \
  ++model.train_ds.batch_duration=200 \
  ++model.optim.name=adamw \
  ++model.optim.lr=0.03 \
  ++model.optim.weight_decay=0.0 \
  ++model.optim.sched.warmup_steps=25 \
  ++exp_manager.version=prompt_column_r000 \
  ++exp_manager.use_datetime_version=false \
  ++exp_manager.exp_dir="$PROJECT_ROOT/runs"
```

`0.03`, 250 steps, and 25 warm-up steps are project starting values, not NVIDIA-published Slovenian optima. With four A100s, compare learning rates independently rather than immediately using four-way DDP.

Mandatory post-run check:

```python
# Compare base and adapted state dictionaries.
# In prompt_column stage, exactly one full parameter tensor may differ:
# prompt_kernel.0.weight
#
# Within that tensor, numerical changes must occur only in the selected
# Slovenian prompt column.
```

---

## 24. Stage S1: prompt-kernel adaptation

Use when prompt-column-only and Slovenian-only residual adapters have been
shown insufficient, and a new work order explicitly permits shared prompt-kernel
adaptation. The residual-adapter proof indicates that more prompt-side capacity
alone can improve synthetic holdout while still regressing real gates.

```bash
+selective_stage=prompt_kernel
++model.optim.weight_decay=0.001
```

This trains the small prompt MLP but changes a shared component. Run multilingual regression gates after every checkpoint.

Suggested initial sweep:

```text
learning rate: 0.003, 0.01, 0.03
fixed steps: 250–1000
```

Treat these as experiment arms.

---

## 25. Stage S2: prompt kernel + RNNT decoder + joint

This is the preferred serious emission-adaptation stage.

```bash
+selective_stage=emission
```

Trainable:

```text
prompt_kernel
decoder
joint
```

Frozen:

```text
entire FastConformer acoustic encoder
```

This preserves the model's learned microphones, noise, reverberation, pauses, speakers, and streaming acoustic behavior while allowing the shared emission machinery to learn Slovenian.

Because decoder/joint weights are shared across languages:

- use a lower learning rate than an official full fine-tune;
- retain an accepted checkpoint parent;
- run multilingual regression evaluation;
- consider a small supported-language replay slice only after a measured regression appears.

Suggested initial sweep:

```text
learning rate: 1e-4, 3e-4, 1e-3
fixed steps: 500–2000
```

---

## 26. Stage S3: last encoder layers

Only use this stage when:

- synthetic Slovenian improves strongly;
- real Slovenian plateaus;
- remaining errors are systematic phonetic/acoustic confusions;
- prompt and RNNT emission stages have reached a stable ceiling.

Run:

```bash
+selective_stage=top_encoder \
+encoder_layers_to_unfreeze=2
```

Then compare 2 vs. 4 final encoder layers. Keep earlier layers frozen.

---

## 27. Full fine-tune

Use the unmodified official script and full model only as:

- a benchmark ceiling;
- a later production candidate if selective stages cannot meet accuracy;
- a route when substantial real Slovenian speech becomes available.

Do not let the active controller automatically cross into full fine-tuning.

---

# Part VI — Adaptive GaMS → TTS → train loop

## 28. Data partitions

### Controller-development set

May expose reference/hypothesis examples to error mining and GaMS.

### Immutable real-Slovenian gate

Used after every round for checkpoint acceptance. The current gates are the
complete FLEURS Slovenian test split as `fleurs-sl-si-test-full-v2` and
deterministic ARTUR-J public-speech gate. GaMS receives only aggregate error
categories, not raw sentences. Historical FLEURS v1 evidence is deprecated.

### Final blind test

Never used for prompt design, active selection, checkpoint choice, or hyperparameter tuning.

### Synthetic holdouts

Hold out:

- GaMS templates;
- lexical families;
- TTS voices/prosodies when available;
- difficult minimal-pair groups.

Synthetic holdouts are diagnostic synthetic-domain evidence. They must be
content-family-disjoint from training data and must not be used for training,
selection, early stopping, or steering. They do not establish real-speech
generalization.

### Multilingual regression set

A small evaluation-only set from several strong base-model languages. It is not required during prompt-column-only training because that update is Slovenian-specific, but it remains a valuable integrity check.

---

## 29. One active-learning round

```text
1. Load the last accepted `.nemo` checkpoint.
2. Evaluate real Slovenian controller-development audio.
3. Cluster substitutions, deletions, insertions, and latency failures.
4. Give GaMS a bounded machine-readable failure brief.
5. Generate a small candidate pool.
6. Validate and deduplicate the text.
7. Synthesize every candidate with the Slovenian TTS.
8. Build a standard NeMo manifest.
9. Run the current model before training at 80 and 320 ms.
10. Select actual failures, low-margin categories, and under-covered phenomena.
11. Mix selected examples with replay and fresh general controls.
12. Train a fixed small step budget.
13. Evaluate all acceptance gates.
14. Accept the challenger or roll it back.
15. Feed the updated error summary into the next GaMS round.
```

No large static synthetic corpus is created.

---

## 30. Initial round sizes

Starting point:

```text
GaMS candidate texts:        2048
TTS-rendered candidates:     2048
actively selected failures:   512
replay examples:               307
fresh general controls:        205
round training set:           1024
```

Training mixture:

```text
50% newest hard examples
30% replay from earlier difficult examples
20% fresh general Slovenian
```

Adjust based on measured coverage and forgetting.

These round sizes are planning targets only. A future work order must first
produce a `TRAINING_ELIGIBLE` corpus under the training-data constitution.
Hard-example selection must operate only on an accepted corpus; high ASR error
on malformed or structurally invalid text is not a valid selection reason.

---

## 31. Candidate selection

The official inference script provides per-example predictions and WER. Add CER and category scores.

Initial ranking:

```text
hard transcription failure
+ character error severity
+ rare Slovenian grapheme failure
+ morphology/suffix failure
+ underrepresented phenomenon
+ lexical novelty
+ disagreement between 80 and 320 ms
+ partial-stream instability
```

Always keep a random control slice to prevent the curriculum from becoming completely myopic.

Useful Slovenian error clusters:

```text
č ↔ c
š ↔ s
ž ↔ z
voiced/unvoiced consonants
vowel confusions
short function words
clitics
case endings
dual forms
gender/number endings
verb endings
proper names
municipalities and street names
numbers, dates, times, money
abbreviations
technical English inside Slovenian grammar
punctuation and capitalization
word insertion after silence
premature emission
late final token
```

---

## 32. GaMS failure brief

Example:

```json
{
  "round": 4,
  "accepted_checkpoint": "prompt_column_r003",
  "streaming_metrics": {
    "wer_80ms": 0.241,
    "wer_320ms": 0.196,
    "cer_320ms": 0.071
  },
  "clusters": [
    {
      "id": "caron_c",
      "type": "grapheme_substitution",
      "reference": "č",
      "hypothesis": "c",
      "count": 61,
      "contexts": [
        "word_initial",
        "before_r",
        "suffix"
      ]
    }
  ],
  "coverage_gaps": [
    "dual verb forms",
    "Slovenian institutional names",
    "technical commands with English nouns"
  ],
  "generation_request": {
    "count": 2048,
    "duration_seconds": [1.0, 12.0],
    "direct_failure_fraction": 0.45,
    "minimal_pair_fraction": 0.20,
    "morphology_fraction": 0.15,
    "long_context_fraction": 0.10,
    "general_control_fraction": 0.10
  }
}
```

Do not expose immutable-gate or final-test text to GaMS.

---

## 33. Active cycle pseudocode

```python
accepted_model = load_accepted_checkpoint()

for round_id in range(max_rounds):
    real_report = evaluate_real_controller_dev(accepted_model)
    error_brief = mine_and_cluster(real_report)

    candidates = gams_generate(
        error_brief,
        count=2048,
    )
    candidates = validate_and_deduplicate(candidates)

    rendered = tts_render(candidates)
    candidate_manifest = build_nemo_manifest(rendered)

    before = evaluate_streaming(
        accepted_model,
        candidate_manifest,
        target_lang="sl-SI",
        contexts=[[56, 0], [56, 3]],
    )

    selected = select_hard_examples(
        rendered,
        before,
        count=512,
    )

    round_manifest = mix_manifests(
        newest=selected,
        replay=replay_reservoir.sample(307),
        general=fresh_general_examples(205),
    )

    challenger = train_fixed_budget(
        parent=accepted_model,
        manifest=round_manifest,
        stage=current_stage,
    )

    gate = evaluate_acceptance_suite(
        parent=accepted_model,
        challenger=challenger,
    )

    if gate.accept:
        accepted_model = challenger
        replay_reservoir.update(selected)
    else:
        rollback(challenger)

    persist_every_artifact(round_id, error_brief, before, gate)
```

---

# Part VII — Evaluation and acceptance

## 34. Metrics

### Text

```text
raw WER
normalized WER
CER
č/š/ž error rate
proper-name WER
number/date/time accuracy
punctuation accuracy
capitalization accuracy
foreign-script leakage
hallucination after silence
```

### Streaming

Evaluate at every context:

```text
[56,0]
[56,1]
[56,3]
[56,6]
[56,13]
```

Track:

```text
final WER/CER
first-token latency
final-token latency
partial transcript churn
80-to-320ms disagreement
word insertions during silence
real-time factor
peak GPU memory
```

### Transfer integrity

```text
WER on selected supported languages
automatic-language-mode smoke test
checkpoint size and module diff
prompt dictionary unchanged
tokenizer hash unchanged
```

---

## 35. Initial acceptance gates

Project starting gates:

- targeted synthetic hard-set error improves by at least 10% relative;
- controller-development real Slovenian improves or stays statistically neutral while a targeted category improves materially;
- normalized FLEURS corpus WER does not regress by more than 1.0 absolute point;
- normalized ARTUR-J corpus WER does not regress by more than 1.0 absolute point;
- normalized FLEURS corpus CER does not regress by more than 1.5 absolute points;
- normalized ARTUR-J corpus CER does not regress by more than 1.5 absolute points;
- empty-hypothesis count does not increase on either real gate;
- supported-language macro WER does not regress by more than 0.5 absolute points after shared decoder/joint parameters are trained;
- 80 ms median final-token latency does not regress materially;
- no increased silence hallucination;
- no new foreign-script leakage;
- tokenizer and prompt dictionary are unchanged;
- the set of changed parameter tensors matches the declared selective stage.

For prompt-column-only runs, add the strict structural gate:

```text
Only prompt_kernel.0.weight may differ.
Inside it, only the Slovenian prompt column may differ.
```

Use paired bootstrap confidence intervals for sufficiently large evaluation sets.

---

# Part VIII — A100 execution strategy

## 36. One A100 80 GB

Run sequentially:

```text
GaMS generation
TTS synthesis
candidate streaming inference
selective fine-tune
acceptance evaluation
```

A 600M model fits comfortably. Start without model parallelism.

---

## 37. Two A100s

```text
GPU 0: accepted checkpoint inference, candidate scoring, gate evaluation
GPU 1: challenger fine-tuning
```

---

## 38. Four A100s

Use independent experiments before DDP:

```text
GPU 0: prompt-column LR arm A
GPU 1: prompt-column LR arm B
GPU 2: prompt-kernel experiment
GPU 3: full-fine-tune reference or streaming evaluation
```

After choosing a stage and optimizer:

```text
GPU 0: continuous accepted-model evaluation
GPU 1–2: DDP challenger training
GPU 3: GaMS/TTS service or alternative challenger
```

Independent ablations initially provide more information than four-way DDP.

---

## 39. Batch sizing

The official notebook uses:

```text
train_ds.batch_duration=200
```

Start with exactly one visible GPU. Historical smaller-platform runs used one
RTX 2080 Ti; current A100 experiments use physical GPU 1 exposed as logical
`cuda:0`. Do not use multi-GPU execution unless a later work order explicitly
permits it.

If out of memory:

```text
reduce batch_duration
reduce joint.fused_batch_size
reduce maximum utterance duration
use gradient accumulation
```

Bucket by audio duration, not merely utterance count.

---

# Part IX — Common failure modes

## 40. Wrong or random language prompt

Symptom:

```text
output is another language
training oscillates
validation differs between runs
```

Fix:

```text
target_lang: "sl-SI" in every record
train_ds.default_prompt_mode=langID
validation_ds.default_prompt_mode=langID
unified_auto_ratio=0.0
```

Only introduce `auto` training later if automatic language detection is a product requirement.

---

## 41. Accidentally replacing the tokenizer

Symptom:

```text
decoder or joint reinitialized
base accuracy collapses
```

Fix:

```text
do not set model.tokenizer.update_tokenizer=true
reuse the checkpoint tokenizer
audit Slovenian round-trip first
```

---

## 42. Treating the generic YAML as the released architecture

Symptom:

```text
freshly initialized 42-layer model
weights fail to load
unexpected parameter count
```

Fix:

```text
restore from the `.nemo` checkpoint
dump model.cfg
use speech_to_text_finetune.py
```

---

## 43. Cache-aware inference dtype error

The public script currently rejects non-FP32 `compute_dtype`.

Use:

```text
compute_dtype=float32
```

AMP is a separate autocast option and must be tested after the FP32 baseline.

---

## 44. Hydra quoting errors

Always quote list arguments:

```bash
att_context_size="[56,3]"
```

Use `+key=value` for a new Hydra key and `++key=value` when intentionally overriding or adding a nested field.

---

## 45. Prompt-column weight decay corrupts other columns

Use:

```text
model.optim.weight_decay=0.0
```

for `prompt_column`.

After training, compare the matrix against the base checkpoint and reject the run if any other column changed.

---

## 46. Excellent TTS score, no real-speech gain

Do not generate more of the same synthetic data.

Instead:

- inspect remaining errors;
- diversify text and available TTS voices/prosody;
- verify transcript style;
- move from prompt-column to emission stage only after the prompt stage plateaus;
- add a small amount of real supervised Slovenian if available;
- keep real speech as the checkpoint-selection authority.

---

## 47. Automatic-language detection degrades

The first plan intentionally trains forced `sl-SI`.

If `target_lang=auto` is required later:

1. create a separate experiment;
2. use `default_prompt_mode=unified`;
3. start with a small auto ratio, not 0.5 automatically;
4. evaluate forced and automatic modes separately;
5. retain language tags during evaluation when measuring detection.

---

# Part X — Reproducibility

## 48. Per-round artifacts

Persist:

```text
base model revision and SHA256
NeMo commit
environment lock
parent checkpoint hash
GaMS model/revision/prompt
all generated candidates
deduplication report
TTS model/revision/parameters
audio manifest
pre-training predictions at 80 and 320 ms
selected-example manifest
replay manifest
selective stage
complete trainable-parameter list
training config
optimizer state
all checkpoints
per-utterance gate predictions
accept/reject decision
rollback reason
```

---

## 49. Immediate execution checklist

### Day-zero contract

```text
[ ] NeMo pinned and installed
[ ] checkpoint downloaded and hashed
[ ] checkpoint class/config dumped
[ ] sl-SI prompt resolves to index 62
[ ] tokenizer round-trip passes
[ ] one real Slovenian file transcribes
[ ] all five streaming contexts run
```

### First training proof

```text
[ ] create 32–64 GaMS sentences
[ ] synthesize mono 16 kHz WAV
[ ] build manifest with target_lang=sl-SI
[ ] run base-model predictions
[ ] prompt-column run overfits the tiny set
[ ] only Slovenian prompt column changed
[ ] resulting `.nemo` restores and streams
```

### First active round

```text
[ ] generate 2048 candidates
[ ] pre-score all candidates
[ ] select 512 actual failures
[ ] mix replay and controls
[ ] train fixed step budget
[ ] evaluate real Slovenian and latency
[ ] accept or roll back
```

---

## 50. Recommended first three experiments

### Experiment A — prompt-column proof

```text
64–256 TTS utterances
prompt-column only
weight decay 0
50–250 steps
```

Question answered:

```text
Can the explicit Slovenian language pathway alone make the frozen recognizer emit materially better Slovenian?
```

### Experiment B — adaptive prompt-column loop

```text
2048 generated candidates per round
512 selected failures
5–10 rounds
```

Question answered:

```text
How far can targeted GaMS curriculum push the model without changing shared acoustic or decoder weights?
```

### Experiment C — emission adaptation

```text
same selected data
prompt kernel + RNNT decoder + joint
encoder fully frozen
```

Question answered:

```text
Does the remaining ceiling come from shared text-emission behavior rather than acoustic representation?
```

Only after those experiments should any encoder layer be unfrozen.

---

## 51. Bottom line

The new project should begin with the model's **actual Slovenian prompt pathway**, not with full retraining.

The fastest technically defensible route is:

```text
1. Install the pinned NeMo source revision.
2. Download and inspect the `.nemo` checkpoint.
3. Force `target_lang=sl-SI`.
4. Reuse the checkpoint tokenizer.
5. Establish 80/320/1120 ms baselines.
6. Generate a tiny GaMS batch and synthesize it.
7. Train only the Slovenian prompt column.
8. Verify that no other parameter region changed.
9. Run failure-directed GaMS rounds.
10. Escalate to prompt kernel + RNNT decoder/joint only after a measured plateau.
11. Touch encoder layers only when real speech proves the acoustic representation is the bottleneck.
```

This preserves the transferable cache-aware acoustic recognizer and uses the old TTS for its ideal role: teaching the existing multilingual model what Slovenian text to emit.
