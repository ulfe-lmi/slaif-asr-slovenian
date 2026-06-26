from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_sha256,
    character_shingles,
    entity_masked_form,
    fingerprint_hash,
    jaccard,
    load_json,
    load_jsonl,
    load_protected_index,
    metadata_leakage,
    number_masked_form,
    sha256_file,
    surface_form,
    token_shingles,
    validate_text_record,
)


TEXT_RECORD_SCHEMA_VERSION = "2.0"
SCALE200_TEXT_VERSION = "gams-corpus-v3-1600-v1"
PUBLIC_REPORT_SCHEMA_VERSION = "1.0"
SELECTION_MAX_BLOCKING_SHINGLE_ROWS = 160
ALLOWED_WHOLE_FILE_OUTCOMES = {
    "ACCEPT",
    "REJECT_GRAMMAR",
    "REJECT_SEMANTICS",
    "REJECT_UNNATURAL",
    "REJECT_TEMPLATE",
    "REJECT_METADATA_LEAK",
    "REJECT_DUPLICATE",
    "REJECT_DOMAIN",
    "REJECT_TRANSCRIPTION",
    "REVISE_AND_REREVIEW",
}
LINE_NUMBER_OR_BULLET = re.compile(r"^\s*(?:\d{1,4}[\).:\-]\s+|[-*•]\s+)")
JSON_OR_MARKUP_LINE = re.compile(r"^\s*(?:[\[{}`]|#+\s+)")
NOISY_HEADING = re.compile(
    r"^\s*(?:stavki|odgovor|output|sentences?|kandidati|candidate|json|slovenian|slovenščina)\s*:?\s*$",
    re.IGNORECASE,
)
METADATA_WORDS = {
    "candidate",
    "row",
    "group",
    "batch",
    "sample",
    "station",
    "skupina",
    "vrstica",
    "kandidat",
    "vzorec",
}
TRAINING_VIEWS = (
    "piper-sl_SI-artur-medium",
    "supertonic-M1",
    "supertonic-M2",
    "supertonic-M3",
    "supertonic-M4",
    "supertonic-F1",
    "supertonic-F2",
    "supertonic-F3",
    "supertonic-F4",
)


@dataclass(frozen=True)
class ParsedLine:
    text: str
    output_ordinal: int


@dataclass(frozen=True)
class Rejection:
    reason: str
    cell_id: str | None = None
    attempt_id: str | None = None
    output_ordinal: int | None = None
    candidate_id: str | None = None
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": self.reason}
        if self.cell_id is not None:
            payload["cell_id"] = self.cell_id
        if self.attempt_id is not None:
            payload["attempt_id"] = self.attempt_id
        if self.output_ordinal is not None:
            payload["output_ordinal"] = self.output_ordinal
        if self.candidate_id is not None:
            payload["candidate_id"] = self.candidate_id
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_sha256(serialized)


def load_generation_config(path: Path) -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_generation_config(config)
    return config


def load_augmentation_config(path: Path) -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_augmentation_config(config)
    return config


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = load_json(repo_path(path))
    if config.get("work_order_id") != "0025":
        raise ValueError("scale-200 experiment config must belong to work order 0025")
    training = config.get("training", {})
    required = {
        "batch_size": 8,
        "exposure_rounds": 20,
        "semantic_rows": 1600,
        "sample_exposures": 32000,
        "optimizer_steps": 4000,
        "optimizer": "AdamW",
        "scheduler": "none",
        "gradient_accumulation": "none",
        "gradient_clipping": "none",
        "seed": 1234,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if float(training.get("learning_rate", -1.0)) != 0.001:
        raise ValueError("training.learning_rate must be 0.001")
    if float(training.get("weight_decay", -1.0)) != 0.0:
        raise ValueError("training.weight_decay must be 0.0")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    return config


def validate_generation_config(config: dict[str, Any]) -> None:
    if config.get("corpus_id") != "sl-corpus-v3-gams-1600-training-v1":
        raise ValueError("unexpected corpus_id")
    if config.get("partition_role") != "selected_training":
        raise ValueError("partition_role must be selected_training")
    if config.get("target_requested_rows") != 2400:
        raise ValueError("target_requested_rows must be 2400")
    if config.get("minimum_admissible_rows") != 1800:
        raise ValueError("minimum_admissible_rows must be 1800")
    if config.get("final_rows") != 1600 or config.get("final_rows_per_cell") != 40:
        raise ValueError("final selection must be 1600 rows, 40 per cell")
    model = config.get("model", {})
    if model.get("repository") != "cjvt/GaMS3-12B-Instruct":
        raise ValueError("GaMS repository mismatch")
    if model.get("revision") != "1d0b27af5748784482600d24779409e7e1dc9adc":
        raise ValueError("GaMS revision mismatch")
    quant = config.get("quantization", {})
    if quant.get("load_in_4bit") is not True or quant.get("quant_type") != "nf4":
        raise ValueError("GaMS must use 4-bit NF4")
    if quant.get("double_quantization") is not True or quant.get("compute_dtype") != "bfloat16":
        raise ValueError("GaMS must use double quantization and BF16 compute")
    generation = config.get("generation", {})
    if generation.get("prompt_batch_size") != 8 or generation.get("oom_fallback_batch_size") != 4:
        raise ValueError("prompt batching must be fixed at 8 with OOM fallback to 4")
    cells = config.get("prompt_cells", [])
    if len(cells) != 40:
        raise ValueError("exactly forty prompt cells are required")
    seen: set[str] = set()
    total = 0
    for cell in cells:
        cell_id = str(cell.get("cell_id", ""))
        if not re.fullmatch(r"cell\d{2}", cell_id):
            raise ValueError(f"unsafe cell_id {cell_id!r}")
        if cell_id in seen:
            raise ValueError(f"duplicate cell_id {cell_id}")
        seen.add(cell_id)
        for key in ("domain", "register", "length_target", "phenomena", "source_family_id", "prompt_revision", "seed_sequence"):
            if key not in cell:
                raise ValueError(f"{cell_id}: missing {key}")
        if int(cell.get("requested_rows", 0)) != 60:
            raise ValueError(f"{cell_id}: requested_rows must be 60")
        maximum_retries = int(cell.get("maximum_retries", -1))
        if maximum_retries < 2:
            raise ValueError(f"{cell_id}: maximum_retries must be at least 2")
        if len(cell.get("seed_sequence", [])) < 3:
            raise ValueError(f"{cell_id}: seed_sequence must include initial attempt plus two retry seeds")
        total += int(cell["requested_rows"])
    if total != int(config["target_requested_rows"]):
        raise ValueError("prompt-cell requested rows must total 2400")


def validate_augmentation_config(config: dict[str, Any]) -> None:
    if config.get("policy_id") != "scale200-transcript-preserving-v1":
        raise ValueError("unexpected augmentation policy")
    voices = config.get("clean_voices", [])
    if tuple(voices) != TRAINING_VIEWS:
        raise ValueError("clean voices must be Piper plus Supertonic M1-M4/F1-F4")
    profiles = config.get("augmentation_profiles", [])
    if len(profiles) != 11:
        raise ValueError("exactly eleven augmentation profiles are required")
    rounds = [int(profile.get("view_round", 0)) for profile in profiles]
    if rounds != list(range(10, 21)):
        raise ValueError("augmentation view rounds must be 10 through 20")
    profile_ids = [str(profile.get("profile_id", "")) for profile in profiles]
    if len(set(profile_ids)) != 11 or any(not item for item in profile_ids):
        raise ValueError("augmentation profile IDs must be unique and non-empty")
    spec = config.get("spec_augment", {})
    if spec.get("enabled") is not True or spec.get("applies_to_evaluation") is not False:
        raise ValueError("SpecAugment policy must be fixed and evaluation-disabled")


def run_dir(config: dict[str, Any]) -> Path:
    return repo_path(config["run_directory"])


def raw_generation_dir(config: dict[str, Any]) -> Path:
    return run_dir(config) / "raw-generation"


def generated_all_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generated-all.local.jsonl"


def fixed_text_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "fixed-training-text.local.jsonl"


def rejected_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "rejected.local.jsonl"


def review_capsule_tsv_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "review-capsule.local.tsv"


def review_capsule_md_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "review-capsule.local.md"


def whole_file_command_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "whole-file-decision-command.local.txt"


def validation_report_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "text-validation.local.json"


def validator_review_output_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "text-validation-review.local.jsonl"


def holdout_review_for_validation_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "holdout-review-for-validation.local.jsonl"


def accepted_review_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "accepted-training-review.local.jsonl"


def text_decisions_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "text-review-decisions.local.jsonl"


def text_certificate_path() -> Path:
    return REPO_ROOT / "docs/data-certificates/sl-corpus-v3-gams-1600-text-v1.json"


def text_report_json_path(config: dict[str, Any]) -> Path:
    return repo_path(config["public_reports"]["text_json"])


def text_report_markdown_path(config: dict[str, Any]) -> Path:
    return repo_path(config["public_reports"]["text_markdown"])


def prompt_cell_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in config["prompt_cells"]}


def generation_seed(cell: dict[str, Any], attempt_index: int) -> int:
    seeds = [int(seed) for seed in cell["seed_sequence"]]
    if attempt_index < len(seeds):
        return seeds[attempt_index]
    return seeds[-1] + (attempt_index - len(seeds) + 1)


def build_prompt(cell: dict[str, Any], *, requested_rows: int, avoid_openings: Sequence[str] | None = None) -> str:
    phenomena = ", ".join(str(item).replace("_", " ") for item in cell["phenomena"])
    requirements = [
        "Naravne samostojne slovenske povedi, ki jih lahko človek verjetno izgovori.",
        "Slovnično pravilna morfologija, ujemanje, skloni in predlogi.",
        "Naravni pomen, primeren register in raznolika skladnja.",
        "Ena poved na vrstico.",
        "Brez oštevilčenja, alinej, JSON-a, oznak, kategorij, razlag ali komentarjev.",
        "Brez umetnih ovojev, ponovljenih repov in zapolnjevanja kvote z istim okvirom.",
        "Brez številk vrstic, skupin, kandidatov, serij, postaj, vzorcev, datotek ali korpusnih oznak v povedih.",
        "Brez kopiranja evalvacijskih ali zaščitenih besedil.",
    ]
    if avoid_openings:
        requirements.append(
            "Za dodatno raznolikost se izogibaj pogostim začetkom: "
            + "; ".join(str(item) for item in avoid_openings)
            + "."
        )
    return "\n".join(
        [
            "Naloga: predlagaj slovenske povedi za notranji diagnostični ASR učni korpus.",
            f"Število povedi: {requested_rows}",
            f"Področje: {cell['domain']}",
            f"Register: {cell['register']}",
            f"Ciljna dolžina: {cell['length_target']}",
            f"Naravno zastopani pojavi: {phenomena}",
            "",
            "Zahteve:",
            *[f"- {item}" for item in requirements],
            "",
            "Odgovor naj vsebuje samo povedi, po eno v vsaki vrstici.",
        ]
    )


def prompt_contains_forbidden_identifier(prompt: str) -> bool:
    lowered = prompt.casefold()
    forbidden = ("candidate_id", "source_family_id", "gamsv3-", "batch 1", "sample 1", "cell01")
    return any(item in lowered for item in forbidden)


def clean_line(value: str) -> str:
    value = value.strip().strip("\"'“”„`")
    value = re.sub(r"\s+", " ", value)
    return unicodedata.normalize("NFC", value).strip()


def extract_utterance_lines(raw_text: str, *, cell_id: str, attempt_id: str) -> tuple[list[ParsedLine], list[Rejection]]:
    lines: list[ParsedLine] = []
    rejected: list[Rejection] = []
    in_fence = False
    for ordinal, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            rejected.append(Rejection("parser_markdown_fence", cell_id, attempt_id, ordinal))
            continue
        if in_fence:
            rejected.append(Rejection("parser_markdown_content", cell_id, attempt_id, ordinal))
            continue
        if not stripped:
            continue
        if NOISY_HEADING.fullmatch(stripped):
            rejected.append(Rejection("parser_heading", cell_id, attempt_id, ordinal))
            continue
        if LINE_NUMBER_OR_BULLET.search(stripped):
            stripped = LINE_NUMBER_OR_BULLET.sub("", stripped)
        if JSON_OR_MARKUP_LINE.search(stripped):
            rejected.append(Rejection("parser_json_or_markup", cell_id, attempt_id, ordinal))
            continue
        line = clean_line(stripped)
        if not line:
            rejected.append(Rejection("parser_empty_text", cell_id, attempt_id, ordinal))
            continue
        lines.append(ParsedLine(text=line, output_ordinal=ordinal))
    if not lines:
        rejected.append(Rejection("parser_no_usable_lines", cell_id, attempt_id))
    return lines, rejected


def attempt_id(cell_id: str, attempt_index: int) -> str:
    return f"{cell_id}-attempt-{attempt_index:02d}"


def candidate_id(cell_id: str, attempt_index: int, output_ordinal: int) -> str:
    return f"gamsv3-{cell_id}-a{attempt_index:02d}-o{output_ordinal:03d}"


def source_id(cell_id: str, attempt_index: int, output_ordinal: int) -> str:
    return f"source-{candidate_id(cell_id, attempt_index, output_ordinal)}"


def build_record(
    *,
    config: dict[str, Any],
    cell: dict[str, Any],
    attempt_index: int,
    output_ordinal: int,
    text: str,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    cid = candidate_id(cell_id, attempt_index, output_ordinal)
    return {
        "schema_version": TEXT_RECORD_SCHEMA_VERSION,
        "candidate_id": cid,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": config["partition_role"],
        "source_type": config["source_type"],
        "source_id": source_id(cell_id, attempt_index, output_ordinal),
        "source_family_id": cell["source_family_id"],
        "template_family_id": None,
        "utterance_family_id": cid,
        "phenomena": list(cell["phenomena"]),
        "domain": cell["domain"],
        "license": config["model"]["license"],
        "generation": {
            "system": "project-generated",
            "method": "gams-local-text-proposal",
            "corpus_id": config["corpus_id"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "prompt_revision": cell["prompt_revision"],
            "corpus_prompt_revision": config["prompt_revision"],
            "seed": generation_seed(cell, attempt_index),
            "prompt_cell": cell_id,
            "generation_attempt": attempt_id(cell_id, attempt_index),
            "extraction_mode": "line",
            "quantization_policy": config["quantization"]["policy"],
        },
        "entities": [],
        "minimal_pair": None,
    }


def metadata_word_in_text(text: str) -> bool:
    words = {part.casefold() for part in re.findall(r"[\wčšžČŠŽ]+", text)}
    return bool(words & METADATA_WORDS)


def protected_indexes(config: dict[str, Any]) -> list[Any]:
    indexes = []
    for path_text in config.get("protected_indexes", []):
        path = repo_path(path_text)
        if path.exists():
            indexes.append(load_protected_index(path))
    return indexes


def load_existing_holdout(config: dict[str, Any]) -> list[dict[str, Any]]:
    holdout = config["existing_holdout"]
    path = repo_path(holdout["text"])
    if sha256_file(path) != holdout["sha256"]:
        raise RuntimeError("existing holdout SHA256 mismatch")
    rows = load_jsonl(path)
    if len(rows) != int(holdout["rows"]):
        raise RuntimeError("existing holdout row count mismatch")
    return rows


def _compile_prohibited_patterns() -> list[re.Pattern[str]]:
    data_config = load_json(REPO_ROOT / "configs/data_quality/training_text_v1.json")
    return [re.compile(pattern, re.IGNORECASE) for pattern in data_config["carrier_detection"].get("prohibited_patterns", [])]


def filter_records(
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    existing_rejections: Sequence[Rejection | dict[str, Any]] = (),
    protected: Sequence[Any] = (),
    holdout_rows: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[Rejection], dict[str, Any]]:
    rejected: list[Rejection] = []
    for item in existing_rejections:
        if isinstance(item, Rejection):
            rejected.append(item)
        else:
            rejected.append(
                Rejection(
                    reason=str(item.get("reason", "unknown")),
                    cell_id=item.get("cell_id"),
                    attempt_id=item.get("attempt_id"),
                    output_ordinal=item.get("output_ordinal"),
                    candidate_id=item.get("candidate_id"),
                    detail=item.get("detail"),
                )
            )
    retained: list[dict[str, Any]] = []
    surface_seen: dict[str, str] = {}
    number_seen: dict[str, str] = {}
    entity_seen: dict[str, str] = {}
    protected_surface = set()
    protected_number = set()
    for index in protected:
        protected_surface.update(index.surface_hashes)
        protected_number.update(index.number_masked_hashes)
    holdout_surface = {fingerprint_hash(surface_form(str(row["target_text"]))) for row in holdout_rows}
    holdout_number = {fingerprint_hash(number_masked_form(str(row["target_text"]))) for row in holdout_rows}
    holdout_entity = {fingerprint_hash(entity_masked_form(str(row["target_text"]), ())) for row in holdout_rows}
    prohibited = _compile_prohibited_patterns()
    reason_counts = Counter(item.reason for item in rejected)
    per_cell: dict[str, Counter[str]] = defaultdict(Counter)
    data_config = load_json(REPO_ROOT / "configs/data_quality/training_text_v1.json")
    for row in sorted(records, key=lambda item: str(item.get("candidate_id", ""))):
        cid = str(row.get("candidate_id", ""))
        generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
        cell_id = str(generation.get("prompt_cell", "unknown"))
        attempt = str(generation.get("generation_attempt", "unknown"))
        text = str(row.get("target_text", ""))
        reject_reason: str | None = None
        detail: str | None = None
        try:
            validate_text_record(row, expected_role=config["partition_role"], config=data_config)
        except Exception as exc:
            reject_reason = "schema_invalid"
            detail = exc.__class__.__name__
        surface = fingerprint_hash(surface_form(text))
        number = fingerprint_hash(number_masked_form(text))
        entity = fingerprint_hash(entity_masked_form(text, ()))
        masked = number_masked_form(text)
        if reject_reason is None and (metadata_leakage(text) or metadata_word_in_text(text)):
            reject_reason = "metadata_leak"
        if reject_reason is None and surface in surface_seen:
            reject_reason = "surface_duplicate"
            detail = surface_seen[surface]
        if reject_reason is None and number in number_seen:
            reject_reason = "number_masked_collision"
            detail = number_seen[number]
        if reject_reason is None and entity in entity_seen:
            reject_reason = "entity_masked_collision"
            detail = entity_seen[entity]
        if reject_reason is None and any(pattern.search(masked) for pattern in prohibited):
            reject_reason = "prohibited_carrier"
        if reject_reason is None and surface in protected_surface:
            reject_reason = "protected_surface_overlap"
        if reject_reason is None and number in protected_number:
            reject_reason = "protected_number_masked_overlap"
        if reject_reason is None and surface in holdout_surface:
            reject_reason = "holdout_surface_overlap"
        if reject_reason is None and number in holdout_number:
            reject_reason = "holdout_number_masked_overlap"
        if reject_reason is None and entity in holdout_entity:
            reject_reason = "holdout_entity_masked_overlap"
        if reject_reason:
            rejected.append(Rejection(reject_reason, cell_id, attempt, candidate_id=cid, detail=detail))
            reason_counts[reject_reason] += 1
            per_cell[cell_id][reject_reason] += 1
            continue
        surface_seen[surface] = cid
        number_seen[number] = cid
        entity_seen[entity] = cid
        retained.append(row)
        per_cell[cell_id]["retained"] += 1
    summary = {
        "retained": len(retained),
        "rejected": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_cell": {cell: dict(sorted(counter.items())) for cell, counter in sorted(per_cell.items())},
    }
    return retained, rejected, summary


def select_fixed_rows(
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        generation = row.get("generation", {})
        rows_by_cell[str(generation.get("prompt_cell", "unknown"))].append(row)
    shortfalls: dict[str, int] = {}
    per_cell_counts: dict[str, int] = {}
    needed = int(config["final_rows_per_cell"])
    for cell in sorted(prompt_cell_by_id(config)):
        rows = rows_by_cell.get(cell, [])
        per_cell_counts[cell] = len(rows)
        if len(rows) < needed:
            shortfalls[cell] = needed - len(rows)
    if shortfalls:
        raise RuntimeError(f"cell shortfall before fixed selection: {shortfalls}")

    data_config = load_json(REPO_ROOT / "configs/data_quality/training_text_v1.json")
    similarity = data_config["similarity"]
    token_threshold = float(similarity["token_jaccard_review_threshold"])
    char_threshold = float(similarity["character_jaccard_review_threshold"])
    row_features = _selection_features(records)
    feature_counts = Counter(feature for features in row_features.values() for feature in features)
    hot_features = {
        feature
        for feature, count in feature_counts.items()
        if count > needed * 2 and feature.startswith(("p2:", "i2:"))
    }
    risky_pairs = _selection_similarity_graph(
        records,
        token_threshold=token_threshold,
        char_threshold=char_threshold,
    )
    risky_by_id: dict[str, set[str]] = defaultdict(set)
    for left, right in risky_pairs:
        risky_by_id[left].add(right)
        risky_by_id[right].add(left)

    rows_by_id = {str(row["candidate_id"]): row for row in records}
    row_token_lengths = {cid: len(_selection_tokens(str(row["target_text"]))) for cid, row in rows_by_id.items()}
    selected_ids: set[str] = set()
    selected_by_cell: dict[str, list[dict[str, Any]]] = {}
    selected_feature_counts: Counter[str] = Counter()

    def row_score(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, str]:
        cid = str(row["candidate_id"])
        features = row_features[cid]
        conflicts = len(risky_by_id[cid] & selected_ids)
        over_limit = sum(1 for feature in features if selected_feature_counts[feature] + 1 > needed * 2 - 2)
        hot_over_limit = sum(
            1
            for feature in features
            if feature in hot_features and selected_feature_counts[feature] + 1 > needed * 2 - 2
        )
        hot_selected_density = sum(selected_feature_counts[feature] for feature in features if feature in hot_features)
        hot_global_density = sum(feature_counts[feature] for feature in features if feature in hot_features)
        global_bigram_density = sum(feature_counts[feature] for feature in features if feature[1:2] == "2")
        selected_density = sum(selected_feature_counts[feature] for feature in features)
        return (
            conflicts * 10_000_000_000,
            hot_over_limit * 1_000_000_000,
            over_limit * 100_000_000,
            hot_selected_density * 1_000_000,
            hot_global_density * 10_000,
            global_bigram_density * 100,
            selected_density * 10 - row_token_lengths[cid],
            stable_sha256(cid),
        )

    forced_hot_counts: dict[str, int] = {}
    for cell, rows in rows_by_cell.items():
        cell_forced = 0
        row_feature_sets = [row_features[str(row["candidate_id"])] for row in rows]
        for feature in hot_features:
            rows_without_feature = sum(1 for features in row_feature_sets if feature not in features)
            cell_forced += max(0, needed - rows_without_feature)
        forced_hot_counts[cell] = cell_forced

    cell_order = sorted(
        prompt_cell_by_id(config),
        key=lambda cell: (-forced_hot_counts.get(cell, 0), len(rows_by_cell.get(cell, ())), cell),
    )
    for cell in cell_order:
        available = list(rows_by_cell[cell])
        chosen: list[dict[str, Any]] = []
        while len(chosen) < needed:
            if not available:
                raise RuntimeError(f"cell shortfall during diversity selection: {cell}")
            best = min(available, key=row_score)
            available.remove(best)
            chosen.append(best)
            cid = str(best["candidate_id"])
            selected_ids.add(cid)
            selected_feature_counts.update(row_features[cid])
        selected_by_cell[cell] = chosen

    selected = []
    for cell in sorted(prompt_cell_by_id(config)):
        selected.extend(selected_by_cell[cell])
    selected = sorted(selected, key=lambda row: (str(row["generation"]["prompt_cell"]), stable_sha256(str(row["candidate_id"]))))
    if len(selected) != int(config["final_rows"]):
        raise RuntimeError(f"expected {config['final_rows']} fixed rows, saw {len(selected)}")
    selected_id_set = {str(row["candidate_id"]) for row in selected}
    selected_risky_pairs = sum(1 for left, right in risky_pairs if left in selected_id_set and right in selected_id_set)
    public_hot_features = {
        stable_sha256(feature): selected_feature_counts[feature]
        for feature in sorted(hot_features)
        if selected_feature_counts[feature]
    }
    summary = {
        "selector": "diversity-aware-greedy-v1",
        "fixed_rows": len(selected),
        "fixed_per_cell": {cell: needed for cell in sorted(prompt_cell_by_id(config))},
        "admissible_per_cell": per_cell_counts,
        "risky_pairs_in_pool": len(risky_pairs),
        "risky_pairs_selected": selected_risky_pairs,
        "hot_feature_count": len(hot_features),
        "hot_feature_selected_counts_sha256": public_hot_features,
    }
    return selected, summary


def _selection_tokens(text: str) -> list[str]:
    return surface_form(text).split()


def _selection_features(records: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    features_by_id: dict[str, set[str]] = {}
    for row in records:
        cid = str(row["candidate_id"])
        entities = row.get("entities", ())
        entities = entities if isinstance(entities, list) else ()
        tokens = entity_masked_form(str(row["target_text"]), entities).split()
        features: set[str] = set()
        for width in range(2, 7):
            if len(tokens) < width:
                continue
            features.add(f"p{width}:{' '.join(tokens[:width])}")
            features.add(f"s{width}:{' '.join(tokens[-width:])}")
            for index in range(len(tokens) - width + 1):
                features.add(f"i{width}:{' '.join(tokens[index:index + width])}")
            if len(tokens) >= width * 2:
                features.add(f"f{width}:{' '.join(tokens[:width])}||{' '.join(tokens[-width:])}")
        features_by_id[cid] = features
    return features_by_id


def _selection_similarity_graph(
    records: Sequence[dict[str, Any]],
    *,
    token_threshold: float,
    char_threshold: float,
) -> set[tuple[str, str]]:
    token_views: dict[tuple[str, int], dict[str, set[str]]] = defaultdict(dict)
    char_views: dict[str, set[str]] = {}
    forms_by_id: dict[str, dict[str, str]] = {}
    for row in records:
        cid = str(row["candidate_id"])
        text = str(row["target_text"])
        entities = row.get("entities", ())
        entities = entities if isinstance(entities, list) else ()
        forms = {
            "surface": surface_form(text),
            "number": number_masked_form(text),
            "entity": entity_masked_form(text, entities),
            "carrier": _carrier_stripped_form(text),
        }
        forms_by_id[cid] = forms
        for view_name, form in forms.items():
            for width in range(2, 6):
                token_views[(view_name, width)][cid] = token_shingles(form, width)
        char_views[cid] = character_shingles(forms["surface"], 5)

    pairs: set[tuple[str, str]] = set()
    for (_view_name, _width), shingle_sets in token_views.items():
        for left, right in _candidate_pairs_from_shingles_limited(shingle_sets):
            if jaccard(shingle_sets[left], shingle_sets[right]) >= token_threshold:
                pairs.add((left, right) if left < right else (right, left))
    for left, right in _candidate_pairs_from_shingles_limited(char_views):
        if jaccard(char_views[left], char_views[right]) >= char_threshold:
            pairs.add((left, right) if left < right else (right, left))
    return pairs


def _candidate_pairs_from_shingles_limited(items: dict[str, set[str]]) -> set[tuple[str, str]]:
    inverted: dict[str, list[str]] = defaultdict(list)
    for item_id, shingles in items.items():
        for shingle in shingles:
            inverted[shingle].append(item_id)
    pairs: set[tuple[str, str]] = set()
    for ids in inverted.values():
        unique_ids = sorted(set(ids))
        if len(unique_ids) < 2 or len(unique_ids) > SELECTION_MAX_BLOCKING_SHINGLE_ROWS:
            continue
        for left_index, left in enumerate(unique_ids):
            for right in unique_ids[left_index + 1 :]:
                pairs.add((left, right))
    return pairs


def _carrier_stripped_form(text: str) -> str:
    form = number_masked_form(text)
    for pattern in _compile_prohibited_patterns():
        form = pattern.sub(" ", form)
    return surface_form(form)


def parse_generated_outputs(
    *,
    config: dict[str, Any],
    outputs: Sequence[tuple[dict[str, Any], str]],
) -> tuple[list[dict[str, Any]], list[Rejection]]:
    records: list[dict[str, Any]] = []
    rejections: list[Rejection] = []
    cells = prompt_cell_by_id(config)
    for prompt_meta, raw in outputs:
        cell_id = str(prompt_meta["cell_id"])
        attempt_index = int(prompt_meta["attempt_index"])
        attempt = attempt_id(cell_id, attempt_index)
        lines, parser_rejections = extract_utterance_lines(raw, cell_id=cell_id, attempt_id=attempt)
        rejections.extend(parser_rejections)
        cell = cells[cell_id]
        for line in lines:
            records.append(
                build_record(
                    config=config,
                    cell=cell,
                    attempt_index=attempt_index,
                    output_ordinal=line.output_ordinal,
                    text=line.text,
                )
            )
    return records, rejections


def planned_prompts(config: dict[str, Any], retained_by_cell: dict[str, int], attempt_index: int) -> list[dict[str, Any]]:
    prompts = []
    cell_count = len(config["prompt_cells"])
    minimum_per_cell = (int(config["minimum_admissible_rows"]) + cell_count - 1) // cell_count
    needed = max(int(config["final_rows_per_cell"]), minimum_per_cell)
    diversity = config.get("diversity_retry", {})
    diversity_cells = {str(item) for item in diversity.get("cell_ids", [])}
    diversity_max_attempt = int(diversity.get("maximum_retry_attempt", -1))
    avoid_openings = [str(item) for item in diversity.get("avoid_openings", [])]
    for cell in config["prompt_cells"]:
        cell_id = str(cell["cell_id"])
        maximum_attempt = int(cell["maximum_retries"])
        if cell_id in diversity_cells:
            maximum_attempt = max(maximum_attempt, diversity_max_attempt)
        if attempt_index > maximum_attempt:
            continue
        is_diversity_retry = cell_id in diversity_cells and attempt_index > int(cell["maximum_retries"])
        if attempt_index > 0 and retained_by_cell.get(cell_id, 0) >= needed and not is_diversity_retry:
            continue
        prompts.append(
            {
                "cell_id": cell_id,
                "attempt_index": attempt_index,
                "seed": generation_seed(cell, attempt_index),
                "prompt": build_prompt(
                    cell,
                    requested_rows=int(cell["requested_rows"]),
                    avoid_openings=avoid_openings if is_diversity_retry else None,
                ),
            }
        )
    return prompts


def write_rejections(path: Path, rejections: Sequence[Rejection | dict[str, Any]]) -> None:
    rows = [item.to_json() if isinstance(item, Rejection) else item for item in rejections]
    atomic_write_jsonl(path, rows)


def read_rejections(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def write_holdout_review_for_validation(config: dict[str, Any]) -> Path:
    source = repo_path(config["existing_holdout"]["linguistic_review"])
    rows = load_jsonl(source)
    atomic_write_jsonl(holdout_review_for_validation_path(config), rows)
    return holdout_review_for_validation_path(config)


def run_text_validator(config: dict[str, Any], *, review_path: Path | None, require_status: str) -> dict[str, Any]:
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/validate_training_corpus.py"),
        "--config",
        str(REPO_ROOT / "configs/data_quality/training_text_v1.json"),
        "--corpus-id",
        f"{config['corpus_id']}-with-existing-holdout",
        "--partition",
        f"selected_training={fixed_text_path(config)}",
        "--partition",
        f"synthetic_holdout={repo_path(config['existing_holdout']['text'])}",
        "--protected-index",
        str(REPO_ROOT / "runs/data-quality/protected/fleurs-v2.hash-index.json"),
        "--protected-index",
        str(REPO_ROOT / "runs/data-quality/protected/artur-j.hash-index.json"),
        "--output-report",
        str(validation_report_path(config)),
        "--local-review-output",
        str(validator_review_output_path(config)),
        "--require-status",
        require_status,
    ]
    if review_path is not None:
        command.extend(["--linguistic-review", str(review_path)])
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if not validation_report_path(config).exists():
        raise RuntimeError(completed.stdout.strip() or "text validator did not write a report")
    report = load_json(validation_report_path(config))
    if completed.returncode != 0:
        report["_command_output"] = completed.stdout.strip()
    return report


def validate_and_select_text(config: dict[str, Any]) -> dict[str, Any]:
    if not generated_all_path(config).exists():
        raise FileNotFoundError(generated_all_path(config))
    generated = load_jsonl(generated_all_path(config))
    parser_rejections = read_rejections(rejected_path(config))
    holdout = load_existing_holdout(config)
    retained, rejected, filter_summary = filter_records(
        generated,
        config=config,
        existing_rejections=parser_rejections,
        protected=protected_indexes(config),
        holdout_rows=holdout,
    )
    if len(retained) < int(config["minimum_admissible_rows"]):
        write_rejections(rejected_path(config), rejected)
        raise RuntimeError(f"minimum admissible shortfall: {len(retained)} < {config['minimum_admissible_rows']}")
    fixed, selection = select_fixed_rows(retained, config=config)
    atomic_write_jsonl(fixed_text_path(config), fixed)
    write_rejections(rejected_path(config), rejected)
    review_path = write_holdout_review_for_validation(config)
    report = run_text_validator(config, review_path=review_path, require_status="DRAFT")
    status = report.get("final_text_status")
    if status == "TEXT_REJECTED":
        raise RuntimeError(f"text validator rejected fixed corpus: {report.get('decision_reasons')}")
    summary = {
        "status": status,
        "decision_reasons": report.get("decision_reasons", []),
        "generated_rows": len(generated),
        "admissible_rows": len(retained),
        "fixed_rows": len(fixed),
        "fixed_text_sha256": sha256_file(fixed_text_path(config)),
        "filter_summary": filter_summary,
        "selection": selection,
        "validator_report_sha256": sha256_file(validation_report_path(config)),
    }
    atomic_write_json(run_dir(config) / "text-selection-summary.local.json", summary)
    return summary


def write_review_capsule(config: dict[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(fixed_text_path(config))
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["generation"]["prompt_cell"])].append(row)
    review_capsule_tsv_path(config).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=review_capsule_tsv_path(config).parent, delete=False) as fp:
        writer = csv.DictWriter(
            fp,
            delimiter="\t",
            fieldnames=["prompt_cell", "domain", "spoken_text", "target_text", "outcome", "review_revision", "reason_codes"],
        )
        writer.writeheader()
        for cell_id in sorted(by_cell):
            for row in sorted(by_cell[cell_id], key=lambda item: stable_sha256(str(item["candidate_id"]))):
                writer.writerow(
                    {
                        "prompt_cell": cell_id,
                        "domain": row["domain"],
                        "spoken_text": row["spoken_text"],
                        "target_text": row["target_text"],
                        "outcome": "",
                        "review_revision": "",
                        "reason_codes": "",
                    }
                )
        temp_name = fp.name
    os.replace(temp_name, review_capsule_tsv_path(config))
    md_lines = [
        "# Corpus v3 1600 Whole-File Review Capsule",
        "",
        "Review every generated Slovenian sentence. Do not approve if quality is mixed.",
        "",
        "Use the exact whole-file command only if the complete file is uniformly acceptable or uniformly rejectable.",
        "",
    ]
    for cell_id in sorted(by_cell):
        md_lines.extend([f"## {cell_id}", ""])
        for row in sorted(by_cell[cell_id], key=lambda item: stable_sha256(str(item["candidate_id"]))):
            md_lines.append(f"- {row['spoken_text']}")
        md_lines.append("")
    atomic_write_text(review_capsule_md_path(config), "\n".join(md_lines))
    corpus_hash = sha256_file(fixed_text_path(config))
    command = " ".join(
        [
            ".venv/bin/python",
            "scripts/run_scale200_synthetic_diagnostic.py",
            "--stage admit-text",
            "--whole-file-outcome <ACCEPT_OR_REJECT>",
            "--review-revision human-scale-review-v1",
            "--decision-id human-scale-decision-v1",
            f"--expected-corpus-sha256 {corpus_hash}",
            "--expected-rows 1600",
        ]
    )
    atomic_write_text(whole_file_command_path(config), command + "\n")
    return {
        "fixed_rows": len(rows),
        "fixed_text_sha256": corpus_hash,
        "review_capsule_tsv_sha256": sha256_file(review_capsule_tsv_path(config)),
        "review_capsule_markdown_sha256": sha256_file(review_capsule_md_path(config)),
        "whole_file_decision_command": command,
    }


def expand_whole_file_decision(
    config: dict[str, Any],
    *,
    outcome: str,
    review_revision: str,
    decision_id: str,
    expected_corpus_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    if outcome not in ALLOWED_WHOLE_FILE_OUTCOMES:
        raise ValueError(f"unsupported whole-file outcome: {outcome}")
    if not review_revision.strip() or not decision_id.strip():
        raise ValueError("review revision and decision ID are required")
    actual = sha256_file(fixed_text_path(config))
    if actual != expected_corpus_sha256:
        raise RuntimeError(f"fixed text SHA mismatch: {actual}")
    rows = load_jsonl(fixed_text_path(config))
    if len(rows) != expected_rows:
        raise RuntimeError(f"fixed text row mismatch: {len(rows)}")
    decisions = [
        {
            "candidate_id": row["candidate_id"],
            "outcome": outcome,
            "review_revision": review_revision,
            "reviewer_approval": decision_id,
            "reason_codes": [] if outcome == "ACCEPT" else [outcome],
            "minimal_pair_approved": False,
        }
        for row in rows
    ]
    atomic_write_jsonl(text_decisions_path(config), decisions)
    if outcome == "ACCEPT":
        accepted_reviews = [
            {
                "candidate_id": row["candidate_id"],
                "outcome": "ACCEPT",
                "review_revision": review_revision,
                "reason_codes": [],
                "minimal_pair_approved": False,
            }
            for row in rows
        ]
        holdout_reviews = load_jsonl(repo_path(config["existing_holdout"]["linguistic_review"]))
        atomic_write_jsonl(accepted_review_path(config), [*accepted_reviews, *holdout_reviews])
        report = run_text_validator(config, review_path=accepted_review_path(config), require_status="TEXT_ACCEPTED")
        if report.get("final_text_status") != "TEXT_ACCEPTED":
            raise RuntimeError(f"accepted text did not reach TEXT_ACCEPTED: {report.get('decision_reasons')}")
        status = "TEXT_ACCEPTED"
    else:
        status = "TEXT_REJECTED" if outcome.startswith("REJECT_") else "DRAFT"
        report = {"final_text_status": status, "decision_reasons": [outcome]}
    certificate = build_text_certificate(
        config,
        status=status,
        outcome=outcome,
        review_revision=review_revision,
        decision_id=decision_id,
        validator_report=report,
    )
    atomic_write_json(text_certificate_path(), certificate)
    write_text_public_reports(config, certificate, validator_report=report)
    return {
        "status": status,
        "whole_file_outcome": outcome,
        "decision_id": decision_id,
        "fixed_text_sha256": actual,
        "rows": len(rows),
        "text_certificate_sha256": sha256_file(text_certificate_path()),
        "validator_status": report.get("final_text_status"),
    }


def fingerprint_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "surface": len({fingerprint_hash(surface_form(row["target_text"])) for row in rows}),
        "number_masked": len({fingerprint_hash(number_masked_form(row["target_text"])) for row in rows}),
        "entity_masked": len({fingerprint_hash(entity_masked_form(row["target_text"], ())) for row in rows}),
        "carrier_stripped": len({fingerprint_hash(entity_masked_form(row["target_text"], ())) for row in rows}),
    }


def family_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    discovered = Counter(fingerprint_hash(entity_masked_form(str(row["target_text"]), ())) for row in rows)
    source = Counter(str(row["source_family_id"]) for row in rows)
    largest = max(discovered.values(), default=0)
    total = max(1, len(rows))
    return {
        "source_family_count": len(source),
        "discovered_family_count": len(discovered),
        "largest_family_size": largest,
        "largest_family_fraction": round(largest / total, 6),
    }


def rejection_counts(config: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("reason", "unknown")) for row in read_rejections(rejected_path(config))).items()))


def build_text_certificate(
    config: dict[str, Any],
    *,
    status: str,
    outcome: str,
    review_revision: str,
    decision_id: str,
    validator_report: dict[str, Any],
) -> dict[str, Any]:
    rows = load_jsonl(fixed_text_path(config))
    return {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v3-gams-1600-text-v1",
        "corpus_id": config["corpus_id"],
        "status": status,
        "decision_date": date.today().isoformat(),
        "work_order_id": "0025",
        "partition_role": config["partition_role"],
        "row_count": len(rows),
        "fixed_text_sha256": sha256_file(fixed_text_path(config)),
        "whole_file_decision": {
            "outcome": outcome,
            "decision_id": decision_id,
            "review_revision": review_revision,
            "row_count": len(rows),
            "corpus_sha256": sha256_file(fixed_text_path(config)),
        },
        "generator": {
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "prompt_revision": config["prompt_revision"],
        },
        "config_sha256": canonical_json_sha256(config),
        "fingerprint_unique_counts": fingerprint_counts(rows),
        "family_summary": family_summary(rows),
        "protected_gate_overlap_counts": validator_report.get("protected_overlap_counts", {}),
        "cross_partition_overlap_counts": validator_report.get("cross_partition_overlap_counts", {}),
        "linguistic_review": {
            "mode": "whole_file",
            "coverage": len(rows),
            "accepted": len(rows) if outcome == "ACCEPT" else 0,
            "rejected": 0 if outcome == "ACCEPT" else len(rows),
        },
        "validator": {
            "status": validator_report.get("final_text_status"),
            "algorithm_version": validator_report.get("validator_algorithm_version"),
            "report_sha256": sha256_file(validation_report_path(config)) if validation_report_path(config).exists() else None,
        },
        "limitations": [
            "Text admission does not prove acoustic suitability.",
            "This corpus is DIAGNOSTIC_ONLY until later audio and experiment certificates are issued.",
            "No TRAINING_ELIGIBLE decision is issued.",
        ],
    }


def write_text_public_reports(config: dict[str, Any], certificate: dict[str, Any], *, validator_report: dict[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(fixed_text_path(config)) if fixed_text_path(config).exists() else []
    certificate_validator = certificate.get("validator", {}) if isinstance(certificate.get("validator"), dict) else {}
    validator_status = (
        certificate_validator.get("status")
        or validator_report.get("final_text_status")
        or validator_report.get("status")
    )
    payload = {
        "schema_version": PUBLIC_REPORT_SCHEMA_VERSION,
        "report_id": "0011-gams1600-text-admission",
        "corpus_id": config["corpus_id"],
        "status": certificate.get("status", "DRAFT"),
        "row_count": len(rows),
        "fixed_text_sha256": sha256_file(fixed_text_path(config)) if fixed_text_path(config).exists() else None,
        "generated_rows": len(load_jsonl(generated_all_path(config))) if generated_all_path(config).exists() else 0,
        "rejection_counts": rejection_counts(config) if rejected_path(config).exists() else {},
        "per_cell_fixed_counts": fixed_counts_by_cell(rows),
        "fingerprint_unique_counts": fingerprint_counts(rows) if rows else {},
        "family_summary": family_summary(rows) if rows else {},
        "validator_status": validator_status,
        "validator_decision_reasons": [] if validator_status == "TEXT_ACCEPTED" else validator_report.get("decision_reasons", []),
        "configuration_sha256": canonical_json_sha256(config),
        "review": certificate.get("whole_file_decision"),
        "limitations": certificate.get("limitations", []),
    }
    atomic_write_json(text_report_json_path(config), payload)
    lines = [
        "# GaMS 1600 Text Admission",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Fixed rows: `{payload['row_count']}`",
        f"- Fixed text SHA256: `{payload['fixed_text_sha256']}`",
        f"- Validator status: `{payload['validator_status']}`",
        f"- Generated rows: `{payload['generated_rows']}`",
        "",
        "This report is aggregate-only and contains no generated sentences or candidate IDs.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    atomic_write_text(text_report_markdown_path(config), "\n".join(lines))
    return payload


def fixed_counts_by_cell(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("generation", {}).get("prompt_cell", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def build_exposure_schedule(
    text_rows: Sequence[dict[str, Any]],
    augmentation_config: dict[str, Any],
    *,
    batch_size: int = 8,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_augmentation_config(augmentation_config)
    ordered = sorted(text_rows, key=lambda row: stable_sha256(str(row["candidate_id"])))
    if len(ordered) != 1600:
        raise ValueError(f"expected 1600 semantic rows, got {len(ordered)}")
    clean_voices = list(augmentation_config["clean_voices"])
    profiles = list(augmentation_config["augmentation_profiles"])
    schedule: list[dict[str, Any]] = []
    for round_index, voice in enumerate(clean_voices, start=1):
        for position, row in enumerate(ordered):
            schedule.append(
                {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": row["candidate_id"],
                    "view_type": "clean",
                    "voice": voice,
                    "profile_id": "clean",
                    "spec_augment": False,
                    "batch_order_seed": stable_sha256(f"{seed}:{round_index}:{position}"),
                }
            )
    for profile_index, profile in enumerate(profiles):
        round_index = int(profile["view_round"])
        profile_id = str(profile["profile_id"])
        for position, row in enumerate(ordered):
            voice = clean_voices[(position + profile_index) % len(clean_voices)]
            spec_augment = ((position + profile_index) % 2) == 0
            schedule.append(
                {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": row["candidate_id"],
                    "view_type": "augmented",
                    "voice": voice,
                    "profile_id": profile_id,
                    "spec_augment": spec_augment,
                    "batch_order_seed": stable_sha256(f"{seed}:{round_index}:{position}"),
                }
            )
    summary = validate_exposure_schedule(schedule, augmentation_config, batch_size=batch_size)
    return schedule, summary


def validate_exposure_schedule(
    schedule: Sequence[dict[str, Any]],
    augmentation_config: dict[str, Any],
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    if len(schedule) != 32000:
        raise ValueError("scale-200 schedule must contain exactly 32000 exposures")
    if len(schedule) % batch_size != 0:
        raise ValueError("schedule must divide evenly into optimizer steps")
    clean_voices = list(augmentation_config["clean_voices"])
    profiles = [str(row["profile_id"]) for row in augmentation_config["augmentation_profiles"]]
    by_round: dict[int, Counter[str]] = defaultdict(Counter)
    semantic_by_round: dict[int, set[str]] = defaultdict(set)
    voice_counts = Counter(str(row["voice"]) for row in schedule)
    profile_counts = Counter(str(row["profile_id"]) for row in schedule if row["view_type"] == "augmented")
    for row in schedule:
        round_index = int(row["round"])
        key = str(row["semantic_key"])
        by_round[round_index][str(row["profile_id"])] += 1
        if key in semantic_by_round[round_index]:
            raise ValueError(f"duplicate semantic item in round {round_index}")
        semantic_by_round[round_index].add(key)
    issues: list[str] = []
    for round_index in range(1, 21):
        if len(semantic_by_round[round_index]) != 1600:
            issues.append(f"round_{round_index}_semantic_count")
    for voice in clean_voices:
        if voice_counts[voice] == 0:
            issues.append(f"voice_{voice}_missing")
    for profile_id in profiles:
        if profile_counts[profile_id] != 1600:
            issues.append(f"profile_{profile_id}_count")
    for held_out in ("supertonic-M5", "supertonic-F5", "M5", "F5"):
        if voice_counts.get(held_out, 0):
            issues.append(f"heldout_voice_{held_out}_leakage")
    if issues:
        raise ValueError(f"invalid exposure schedule: {issues}")
    return {
        "status": "PASSED",
        "exposures": len(schedule),
        "rounds": 20,
        "optimizer_steps": len(schedule) // batch_size,
        "batch_size": batch_size,
        "clean_voice_counts": {voice: voice_counts[voice] for voice in clean_voices},
        "augmentation_profile_counts": {profile_id: profile_counts[profile_id] for profile_id in profiles},
        "heldout_voice_exposures": {voice: voice_counts.get(voice, 0) for voice in ("supertonic-M5", "supertonic-F5")},
    }


def multiplier_table() -> dict[str, Any]:
    return {
        "reference_semantic_items": 160,
        "new_semantic_items": 1600,
        "semantic_text_multiplier": 10,
        "clean_voice_realizations_per_text": 9,
        "augmentation_views_per_text": 11,
        "total_view_records": 32000,
        "exposure_multiplier": 200,
        "interpretation": "200x refers to deterministic exposure count, not independent linguistic information.",
    }


def verify_directional_reference(report_path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(repo_path(report_path))
    if actual != expected_sha256:
        raise RuntimeError(f"Experiment 0012 report SHA mismatch: {actual}")
    payload = load_json(repo_path(report_path))
    decision = payload["directional_evaluation"]["decision"]
    if round(float(decision["replay_supertonic_burden"]), 3) != 9.536:
        raise RuntimeError("Experiment 0012 replay burden mismatch")
    return {
        "sha256": actual,
        "classification": decision["classification"],
        "replay_supertonic_burden": decision["replay_supertonic_burden"],
    }
