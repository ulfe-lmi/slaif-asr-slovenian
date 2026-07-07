from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from slaif_asr.batched_streaming import StreamingRecord, load_gate_records, metrics_for, resolve_manifest_audio_path
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_jsonl, sha256_file
from slaif_asr.real_eval import normalize_sl_asr_text
from slaif_asr.scale8000_corpus import local_run_path
from slaif_asr.scale8000_clean_training import (
    BASE_DIRECTIONAL_METRICS,
    SCALE2000_DIRECTIONAL_METRICS,
    assert_public_report_safe,
    metric_row,
    read_json,
    read_jsonl,
    scale8000_supertonic_heldout_records,
    scale8000_synthetic_holdout_records,
)


ARM_NAME = "text_only_decoder_lm_adapter_v1"
SCALE8000_CLEAN_DIRECTIONAL_METRICS = {
    "piper_synthetic_holdout": {"wer": 62.267, "cer": 21.979, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 28.882, "cer": 8.004, "empty": 0},
    "fleurs_v2": {"wer": 51.268, "cer": 16.259, "empty": 0},
    "artur_j": {"wer": 60.856, "cer": 20.708, "empty": 0},
}
PUBLIC_FORBIDDEN_KEYS = {"text", "spoken_text", "target_text", "tokens", "token_ids", "sample_id", "reference", "hypothesis", "audio_filepath"}
PUBLIC_FORBIDDEN_MARKERS = ("gamsv", "fleurs-sl-si-test-occ-", "artur-j-public-", ".wav", "/data-nvme/", "/home/", "/tmp/")


@dataclass(frozen=True)
class TextRow:
    row_id: str
    normalized_text: str
    source_family_id: str
    utterance_family_id: str


@dataclass(frozen=True)
class TokenizedRow:
    row_id: str
    token_ids: list[int]
    split: str


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_config(path: str | Path = "configs/experiments/text_only_decoder_lm_adapter_v1.json") -> dict[str, Any]:
    config = read_json(REPO_ROOT / path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0029":
        raise ValueError("work_order_id must be 0029")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    data = config["data"]
    if data.get("corpus_id") != "sl-corpus-v5-scale8000-training-v1":
        raise ValueError("unexpected text corpus ID")
    if data.get("text_rows") != 64000 or data.get("train_rows") != 60800 or data.get("validation_rows") != 3200:
        raise ValueError("text-only split counts are invalid")
    training = config["training"]
    required = {
        "epochs": 20,
        "effective_batch_size": 128,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "scheduler": "linear_warmup_5_percent_then_cosine",
        "gradient_clipping": 1.0,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("microbatch_candidates") != [128, 64, 32, 16, 8]:
        raise ValueError("microbatch candidates must be [128,64,32,16,8]")


def local_path(path_text: str | Path) -> Path:
    return local_run_path(path_text)


def run_dir(config: dict[str, Any]) -> Path:
    return local_path(config["run_dir"])


def accepted_text_path(config: dict[str, Any]) -> Path:
    return local_path(config["data"]["fixed_text"])


def load_accepted_text_rows(config: dict[str, Any]) -> list[TextRow]:
    path = accepted_text_path(config)
    actual = sha256_file(path)
    if actual != config["data"]["text_sha256"]:
        raise RuntimeError(f"accepted scale-8000 text SHA mismatch: {actual}")
    rows = read_jsonl(path)
    if len(rows) != int(config["data"]["text_rows"]):
        raise RuntimeError("accepted scale-8000 text row count mismatch")
    output = []
    for row in rows:
        text = str(row.get("target_text") or row.get("spoken_text") or "")
        if not text.strip():
            raise RuntimeError("empty accepted text row")
        output.append(
            TextRow(
                row_id=str(row["candidate_id"]),
                normalized_text=normalize_sl_asr_text(text),
                source_family_id=str(row.get("source_family_id", "")),
                utterance_family_id=str(row.get("utterance_family_id", "")),
            )
        )
    return output


def deterministic_text_split(rows: Sequence[TextRow], corpus_id: str) -> dict[str, list[TextRow]]:
    ordered = sorted(rows, key=lambda row: stable_sha256(corpus_id + row.row_id + row.normalized_text))
    train = list(ordered[:60800])
    validation = list(ordered[60800:])
    if len(train) != 60800 or len(validation) != 3200:
        raise RuntimeError("text-only split count mismatch")
    if set(row.row_id for row in train) & set(row.row_id for row in validation):
        raise RuntimeError("text-only train/validation split overlaps")
    return {"train": train, "validation": validation}


def tokenizer_text_to_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "text_to_ids"):
        return list(tokenizer.text_to_ids(text))
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "text_to_ids"):
        return list(tokenizer.tokenizer.text_to_ids(text))
    if callable(tokenizer):
        return list(tokenizer(text))
    raise TypeError("unsupported tokenizer interface")


def tokenizer_vocab_size(tokenizer: Any) -> int:
    if hasattr(tokenizer, "vocab_size"):
        value = tokenizer.vocab_size
        return int(value() if callable(value) else value)
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "vocab_size"):
        value = tokenizer.tokenizer.vocab_size
        return int(value() if callable(value) else value)
    if hasattr(tokenizer, "__len__"):
        return int(len(tokenizer))
    raise TypeError("could not determine tokenizer vocabulary size")


def tokenizer_special_id(tokenizer: Any, names: Sequence[str], default: int) -> int:
    for name in names:
        if hasattr(tokenizer, name):
            value = getattr(tokenizer, name)
            if value is not None:
                value = value() if callable(value) else value
                if int(value) >= 0:
                    return int(value)
        if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, name):
            value = getattr(tokenizer.tokenizer, name)
            if value is not None:
                value = value() if callable(value) else value
                if int(value) >= 0:
                    return int(value)
    return int(default)


def tokenize_split(split: dict[str, list[TextRow]], tokenizer: Any) -> tuple[dict[str, list[TokenizedRow]], dict[str, Any]]:
    tokenized: dict[str, list[TokenizedRow]] = {"train": [], "validation": []}
    empty_rows = []
    lengths = []
    for split_name, rows in split.items():
        for row in rows:
            ids = tokenizer_text_to_ids(tokenizer, row.normalized_text)
            if not ids:
                empty_rows.append(split_name)
                continue
            lengths.append(len(ids))
            tokenized[split_name].append(TokenizedRow(row_id=row.row_id, token_ids=ids, split=split_name))
    stats = {
        "vocabulary_size": tokenizer_vocab_size(tokenizer),
        "empty_token_rows": len(empty_rows),
        "rows_rejected_by_tokenization": len(empty_rows),
        "tokenized_rows": {name: len(rows) for name, rows in tokenized.items()},
        "token_count": sum(lengths),
        "token_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 6) if lengths else 0,
        },
        "oov_events": 0,
    }
    return tokenized, stats


def make_lm_batch(rows: Sequence[TokenizedRow], *, bos_id: int, eos_id: int, pad_id: int, device: str | torch.device) -> dict[str, torch.Tensor]:
    sequences = [[bos_id, *row.token_ids] for row in rows]
    targets = [[*row.token_ids, eos_id] for row in rows]
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    labels = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    lengths = torch.zeros((len(rows),), dtype=torch.long, device=device)
    for index, (seq, tgt) in enumerate(zip(sequences, targets)):
        length = len(seq)
        input_ids[index, :length] = torch.tensor(seq, dtype=torch.long, device=device)
        labels[index, :length] = torch.tensor(tgt, dtype=torch.long, device=device)
        mask[index, :length] = True
        lengths[index] = length
    return {"input_ids": input_ids, "labels": labels, "mask": mask, "lengths": lengths}


def decoder_lm_forward_loss(model: Any, lm_head: torch.nn.Module, batch: dict[str, torch.Tensor], *, pad_id: int) -> torch.Tensor:
    decoder, _target_length, _state = model.decoder(targets=batch["input_ids"], target_length=batch["lengths"])
    hidden = decoder.transpose(1, 2)
    logits = lm_head(hidden)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=int(pad_id))


def perplexity(loss: float) -> float:
    return round(math.exp(min(float(loss), 50.0)), 6)


def batch_order(rows: Sequence[TokenizedRow], *, epoch: int, seed: int, batch_size: int) -> list[list[TokenizedRow]]:
    ordered = sorted(rows, key=lambda row: (len(row.token_ids), stable_sha256(row.row_id)))
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    keyed = sorted(enumerate(batches), key=lambda item: stable_sha256(f"{seed}:{epoch}:{item[0]}"))
    return [batch for _index, batch in keyed]


def split_token_counts(tokenized: dict[str, list[TokenizedRow]]) -> dict[str, int]:
    return {name: sum(len(row.token_ids) for row in rows) for name, rows in tokenized.items()}


def real_regression_burden(metrics: dict[str, dict[str, Any]], base_metrics: dict[str, dict[str, Any]] = BASE_DIRECTIONAL_METRICS) -> float:
    value = 0.0
    for split in ("fleurs_v2", "artur_j"):
        value += max(0.0, float(metrics[split]["wer"]) - float(base_metrics[split]["wer"]))
        value += max(0.0, float(metrics[split]["cer"]) - float(base_metrics[split]["cer"]))
    return round(value, 6)


def classify_text_only(metrics: dict[str, dict[str, Any]], *, text_validation_improved: bool) -> dict[str, Any]:
    burden = real_regression_burden(metrics)
    real_improves = any(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in ("fleurs_v2", "artur_j")
        for metric in ("wer", "cer")
    )
    real_not_worse_quarter = all(
        float(metrics[split][metric]) - float(BASE_DIRECTIONAL_METRICS[split][metric]) <= 0.25
        for split in ("fleurs_v2", "artur_j")
        for metric in ("wer", "cer")
    )
    empty_ok = all(int(metrics[split]["empty"]) <= int(BASE_DIRECTIONAL_METRICS[split]["empty"]) for split in ("fleurs_v2", "artur_j"))
    synthetic_gain = any(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout")
        for metric in ("wer", "cer")
    )
    catastrophic = any(
        float(metrics[split][metric]) - float(BASE_DIRECTIONAL_METRICS[split][metric]) > 1.0
        for split in ("fleurs_v2", "artur_j")
        for metric in ("wer", "cer")
    ) or any(int(metrics[split]["empty"]) > int(BASE_DIRECTIONAL_METRICS[split]["empty"]) for split in ("fleurs_v2", "artur_j"))
    if burden == 0.0 and real_improves and real_not_worse_quarter and empty_ok:
        classification = "TEXT_ONLY_DECODER_LM_REAL_GAIN_DIRECTIONAL"
    elif catastrophic:
        classification = "TEXT_ONLY_DECODER_LM_DEGRADES_ASR"
    elif synthetic_gain and burden > 0.0:
        classification = "TEXT_ONLY_DECODER_LM_HELPS_SYNTHETIC_ONLY"
    elif text_validation_improved:
        classification = "TEXT_ONLY_DECODER_LM_NO_ASR_GAIN"
    else:
        classification = "TEXT_ONLY_DECODER_LM_DEGRADES_ASR"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "real_regression_burden": burden,
        "real_gate_improves_vs_base": real_improves,
        "real_not_worse_than_base_by_0_25": real_not_worse_quarter,
        "empty_hypotheses_not_increased_vs_base": empty_ok,
        "synthetic_holdout_gain_vs_base": synthetic_gain,
        "text_validation_loss_improved": text_validation_improved,
    }


def directional_suite(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]]]:
    split_records = {
        "piper_synthetic_holdout": scale8000_synthetic_holdout_records(config),
        "supertonic_heldout_voice_holdout": scale8000_supertonic_heldout_records(config),
        "fleurs_v2": load_gate_records(
            local_path(config["data"]["fleurs_v2_manifest"]),
            expected_sha256=config["data"]["fleurs_v2_manifest_sha256"],
            expected_rows=int(config["data"]["fleurs_v2_rows"]),
            gate_id="fleurs-sl-si-test-full-v2",
        ),
        "artur_j": load_gate_records(
            local_path(config["data"]["artur_j_manifest"]),
            expected_sha256=config["data"]["artur_j_manifest_sha256"],
            expected_rows=int(config["data"]["artur_j_rows"]),
            gate_id="artur-j-public-gate-v1",
        ),
    }
    suite = []
    for split, records in split_records.items():
        for index, record in enumerate(records):
            suite.append(
                StreamingRecord(
                    sample_id=f"{split}:{index:04d}",
                    audio_filepath=record.audio_filepath,
                    duration=record.duration,
                    reference=record.reference,
                    original_index=len(suite),
                    row={"split": split, "source_order": index},
                )
            )
    if len(suite) != 1378:
        raise RuntimeError(f"expected 1378 directional rows, found {len(suite)}")
    return suite, split_records


def metrics_from_predictions(suite: Sequence[StreamingRecord], split_records: dict[str, list[StreamingRecord]], predictions: dict[str, str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected = {row.sample_id for row in suite}
    if set(predictions) != expected:
        raise RuntimeError("directional prediction set mismatch")
    split_summaries = {}
    metric_table = {}
    for split in split_records:
        rows = [row for row in suite if row.row["split"] == split]
        split_predictions = {row.sample_id: predictions[row.sample_id] for row in rows}
        summary = {"rows": len(rows), "audio_duration_seconds": round(sum(row.duration for row in rows), 6), "metrics": metrics_for(rows, split_predictions)}
        split_summaries[split] = summary
        metric_table[split] = metric_row(summary)
    return split_summaries, metric_table


def assert_text_only_public_report_safe(payload: Any) -> None:
    assert_public_report_safe(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in PUBLIC_FORBIDDEN_KEYS:
                    raise ValueError(f"public report contains forbidden key: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(payload)
    for marker in PUBLIC_FORBIDDEN_MARKERS:
        if marker in serialized:
            raise ValueError(f"public report contains forbidden marker: {marker}")
