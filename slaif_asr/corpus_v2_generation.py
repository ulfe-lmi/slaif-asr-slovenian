from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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


CORPUS_V2_GENERATOR_VERSION = "gams-corpus-v2-reservoir-v1"
TEXT_RECORD_SCHEMA_VERSION = "2.0"
PUBLIC_REPORT_SCHEMA_VERSION = "1.0"
REVIEW_REVISION_PLACEHOLDER = ""
LINE_NUMBER_OR_BULLET = re.compile(r"^\s*(?:\d{1,4}[\).:\-]\s+|[-*•]\s+)")
JSON_OR_MARKUP_LINE = re.compile(r"^\s*(?:[\[{}`]|#+\s+)")
NOISY_HEADING = re.compile(
    r"^\s*(?:stavki|odgovor|output|sentences?|kandidati|candidate|json|slovenian)\s*:?\s*$",
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root() / path


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("corpus_id") != "sl-corpus-v2-gams-candidate-reservoir-v1":
        raise ValueError("unexpected corpus_id")
    model = config.get("model", {})
    if model.get("repository") != "cjvt/GaMS3-12B-Instruct":
        raise ValueError("GaMS repository must be cjvt/GaMS3-12B-Instruct")
    revision = model.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("GaMS revision must be a 40-character commit SHA")
    quant = config.get("quantization", {})
    if quant.get("load_in_4bit") is not True or quant.get("quant_type") != "nf4":
        raise ValueError("GaMS must use 4-bit NF4")
    if quant.get("double_quantization") is not True or quant.get("compute_dtype") != "bfloat16":
        raise ValueError("GaMS must use double quantization and BF16 compute")
    device = config.get("device_policy", {})
    if device.get("visible_gpu_count") != 1 or device.get("cpu_offload") is not False or device.get("disk_offload") is not False:
        raise ValueError("GaMS device policy must forbid offload and expose one GPU")
    generation = config.get("generation", {})
    allowed = set(generation.get("allowed_prompt_batch_sizes", []))
    if generation.get("prompt_batch_size") not in allowed or allowed != {1, 2, 4, 8}:
        raise ValueError("prompt batching must allow exactly 1, 2, 4, 8 with configured default")
    cells = config.get("prompt_cells", [])
    if len(cells) != 12:
        raise ValueError("exactly twelve prompt cells are required")
    total = 0
    seen_ids: set[str] = set()
    for cell in cells:
        cell_id = str(cell.get("cell_id", ""))
        if not re.fullmatch(r"cell\d{2}", cell_id):
            raise ValueError(f"unsafe cell_id {cell_id!r}")
        if cell_id in seen_ids:
            raise ValueError(f"duplicate cell_id {cell_id}")
        seen_ids.add(cell_id)
        for key in ("domain", "register", "length_target", "phenomena", "source_family_id", "prompt_revision", "seed_sequence"):
            if key not in cell:
                raise ValueError(f"{cell_id}: missing {key}")
        if int(cell.get("requested_rows", 0)) != 40:
            raise ValueError(f"{cell_id}: requested_rows must be 40")
        if int(cell.get("maximum_retries", -1)) != 2:
            raise ValueError(f"{cell_id}: maximum_retries must be 2")
        if len(cell.get("seed_sequence", [])) != 3:
            raise ValueError(f"{cell_id}: seed_sequence must include initial attempt plus two retries")
        total += int(cell["requested_rows"])
    if total != int(config.get("target_generated_rows", 0)):
        raise ValueError("target_generated_rows must equal prompt-cell requested rows")


def config_sha256(config: dict[str, Any]) -> str:
    return canonical_json_sha256(config)


def run_dir(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["run_directory"])


def raw_generation_dir(config: dict[str, Any]) -> Path:
    return run_dir(config) / "raw-generation"


def generated_all_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generated-all.local.jsonl"


def pre_review_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "pre-review-candidates.local.jsonl"


def rejected_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "rejected.local.jsonl"


def review_template_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "linguistic-review-template.local.jsonl"


def review_sheet_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "linguistic-review-sheet.local.tsv"


def validation_report_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "validation-report.local.json"


def local_review_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "review.local.jsonl"


def gpu_monitor_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "gpu-monitor.local.csv"


def public_json_report_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["public_reports"]["json"])


def public_markdown_report_path(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["public_reports"]["markdown"])


def prompt_cell_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in config["prompt_cells"]}


def build_prompt(cell: dict[str, Any], *, requested_rows: int) -> str:
    phenomena = ", ".join(str(item).replace("_", " ") for item in cell["phenomena"])
    requirements = [
        "Napiši naravne samostojne slovenske povedi.",
        "Vsaka poved mora biti slovnično pravilna, z naravnim sklonom, predlogi, ujemanjem in pomenom.",
        "Vsaka poved mora biti verjetna v govorjeni rabi in primerna za navedeno področje.",
        "Uporabi leksikalno in skladenjsko raznolikost.",
        "Vsako poved napiši v svojo vrstico.",
        "Ne uporabljaj oštevilčenja, alinej, JSON-a, oznak, razlag ali prevodov.",
        "Ne uporabljaj ponavljajočega se ovoja, skupnega uvoda ali umetnega repa.",
        "V poved ne vstavljaj številk vrstic, skupin, kandidatov, serij, postaj, vzorcev ali datotečnih oznak.",
        "Ne kopiraj besedila iz evalvacijskih zbirk.",
        "Kvote ne zapolni s ponavljanjem istega stavčnega okvira.",
    ]
    return "\n".join(
        [
            "Naloga: predlagaj slovenske povedi za notranji vir kandidatov za ASR.",
            f"Število povedi: {requested_rows}",
            f"Področje: {cell['domain']}",
            f"Register: {cell['register']}",
            f"Ciljna dolžina: {cell['length_target']}",
            f"Pojavi, ki naj bodo naravno zastopani: {phenomena}",
            "",
            "Zahteve:",
            *[f"- {item}" for item in requirements],
            "",
            "Odgovor naj vsebuje samo povedi, po eno v vsaki vrstici.",
        ]
    )


def prompt_contains_forbidden_identifier(prompt: str) -> bool:
    lowered = prompt.casefold()
    forbidden = ("cell01", "candidate_id", "source_family_id", "batch 1", "sample 1")
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
    return f"gamsv2-{cell_id}-a{attempt_index:02d}-o{output_ordinal:03d}"


def source_id(cell_id: str, attempt_index: int, output_ordinal: int) -> str:
    return f"source-{candidate_id(cell_id, attempt_index, output_ordinal)}"


def build_record(
    *,
    config: dict[str, Any],
    cell: dict[str, Any],
    attempt_index: int,
    output_ordinal: int,
    text: str,
    extraction_mode: str,
) -> dict[str, Any]:
    seed = int(cell["seed_sequence"][attempt_index])
    cid = candidate_id(str(cell["cell_id"]), attempt_index, output_ordinal)
    return {
        "schema_version": TEXT_RECORD_SCHEMA_VERSION,
        "candidate_id": cid,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": config["partition_role"],
        "source_type": config["source_type"],
        "source_id": source_id(str(cell["cell_id"]), attempt_index, output_ordinal),
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
            "seed": seed,
            "prompt_cell": cell["cell_id"],
            "generation_attempt": attempt_id(str(cell["cell_id"]), attempt_index),
            "extraction_mode": extraction_mode,
            "quantization_policy": config["quantization"]["policy"],
        },
        "entities": [],
        "minimal_pair": None,
    }


def metadata_word_in_text(text: str) -> bool:
    words = {part.casefold() for part in re.findall(r"[\wčšžČŠŽ]+", text)}
    return bool(words & METADATA_WORDS)


def _compile_prohibited_patterns(config: dict[str, Any]) -> list[re.Pattern[str]]:
    data_config = load_json(resolve_repo_path("configs/data_quality/training_text_v1.json"))
    return [re.compile(pattern, re.IGNORECASE) for pattern in data_config["carrier_detection"].get("prohibited_patterns", [])]


def load_protected_indexes(config: dict[str, Any]) -> list[Any]:
    indexes = []
    for path_text in config.get("protected_indexes", []):
        path = resolve_repo_path(path_text)
        if path.exists():
            indexes.append(load_protected_index(path))
    return indexes


def filter_records(
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    existing_rejections: Sequence[Rejection] | Sequence[dict[str, Any]] = (),
    protected_indexes: Sequence[Any] = (),
) -> tuple[list[dict[str, Any]], list[Rejection], dict[str, Any]]:
    retained: list[dict[str, Any]] = []
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
    surface_seen: dict[str, str] = {}
    number_seen: dict[str, str] = {}
    entity_seen: dict[str, str] = {}
    prohibited = _compile_prohibited_patterns(config)
    protected_surface_hashes: set[str] = set()
    for index in protected_indexes:
        protected_surface_hashes.update(index.surface_hashes)
    filtering = config.get("filtering", {})
    max_ngram_count = int(filtering.get("max_repeated_token_ngram_count", 0))
    token_threshold = float(filtering.get("token_jaccard_reject_threshold", 1.1))
    char_threshold = float(filtering.get("character_jaccard_reject_threshold", 1.1))
    ngram_counts: Counter[str] = Counter()
    retained_similarity_views: list[tuple[set[str], set[str], set[str], set[str]]] = []

    reason_counts: Counter[str] = Counter(item.reason for item in rejected)
    per_cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        cid = str(row.get("candidate_id", ""))
        generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
        cell_id = str(generation.get("prompt_cell", "unknown"))
        attempt = str(generation.get("generation_attempt", "unknown"))
        text = str(row.get("spoken_text", ""))
        reject_reason: str | None = None
        detail: str | None = None
        try:
            validate_text_record(row, expected_role=config["partition_role"], config=load_json(resolve_repo_path("configs/data_quality/training_text_v1.json")))
        except Exception as exc:
            reject_reason = "schema_invalid"
            detail = exc.__class__.__name__
        if reject_reason is None and (metadata_leakage(text) or metadata_word_in_text(text)):
            reject_reason = "metadata_leak"
        surface = fingerprint_hash(surface_form(text))
        number = fingerprint_hash(number_masked_form(text))
        entity = fingerprint_hash(entity_masked_form(text, ()))
        if reject_reason is None and surface in surface_seen:
            reject_reason = "surface_duplicate"
            detail = surface_seen[surface]
        if reject_reason is None and number in number_seen:
            reject_reason = "number_masked_collision"
            detail = number_seen[number]
        if reject_reason is None and entity in entity_seen:
            reject_reason = "entity_masked_collision"
            detail = entity_seen[entity]
        if reject_reason is None and any(pattern.search(number_masked_form(text)) for pattern in prohibited):
            reject_reason = "prohibited_carrier"
        if reject_reason is None and protected_surface_hashes and surface in protected_surface_hashes:
            reject_reason = "protected_surface_overlap"
        entity_form = entity_masked_form(text, ())
        tokens = entity_form.split()
        ngrams: list[str] = []
        for ngram_size in range(2, 7):
            for index in range(0, max(0, len(tokens) - ngram_size + 1)):
                ngrams.append(" ".join(tokens[index : index + ngram_size]))
        if reject_reason is None and max_ngram_count > 0 and any(ngram_counts[item] >= max_ngram_count for item in ngrams):
            reject_reason = "token_ngram_concentration"
        if reject_reason is None:
            surface_tokens: set[str] = set()
            number_tokens: set[str] = set()
            entity_tokens: set[str] = set()
            for ngram_size in (2, 3, 4, 5):
                surface_tokens.update(token_shingles(surface_form(text), ngram_size))
                number_tokens.update(token_shingles(number_masked_form(text), ngram_size))
                entity_tokens.update(token_shingles(entity_form, ngram_size))
            char_tokens = character_shingles(surface_form(text), 5)
            for old_surface, old_number, old_entity, old_chars in retained_similarity_views:
                if (
                    jaccard(surface_tokens, old_surface) >= token_threshold
                    or jaccard(number_tokens, old_number) >= token_threshold
                    or jaccard(entity_tokens, old_entity) >= token_threshold
                    or jaccard(char_tokens, old_chars) >= char_threshold
                ):
                    reject_reason = "fuzzy_similarity_candidate"
                    break
        if reject_reason is not None:
            rejected.append(Rejection(reject_reason, cell_id, attempt, candidate_id=cid, detail=detail))
            reason_counts[reject_reason] += 1
            per_cell_counts[cell_id][reject_reason] += 1
            continue
        surface_seen[surface] = cid
        number_seen[number] = cid
        entity_seen[entity] = cid
        for item in ngrams:
            ngram_counts[item] += 1
        retained_similarity_views.append((surface_tokens, number_tokens, entity_tokens, char_tokens))
        retained.append(row)
        per_cell_counts[cell_id]["retained"] += 1

    summary = {
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_cell": {cell: dict(sorted(counter.items())) for cell, counter in sorted(per_cell_counts.items())},
    }
    return retained, rejected, summary


def review_template_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(records, key=lambda item: item["candidate_id"]):
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "spoken_text": row["spoken_text"],
                "target_text": row["target_text"],
                "domain": row["domain"],
                "phenomena": row["phenomena"],
                "source_family_id": row["source_family_id"],
                "outcome": REVIEW_REVISION_PLACEHOLDER,
                "review_revision": REVIEW_REVISION_PLACEHOLDER,
                "reason_codes": [],
                "minimal_pair_approved": False,
            }
        )
    return rows


def write_review_outputs(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> None:
    template = review_template_rows(records)
    atomic_write_jsonl(review_template_path(config), template)
    review_sheet_path(config).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=review_sheet_path(config).parent, delete=False) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "candidate_id",
                "spoken_text",
                "target_text",
                "domain",
                "phenomena",
                "source_family_id",
                "outcome",
                "review_revision",
                "reason_codes",
                "minimal_pair_approved",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in template:
            writer.writerow({**row, "phenomena": ",".join(row["phenomena"]), "reason_codes": ""})
        temp_name = fp.name
    os.replace(temp_name, review_sheet_path(config))
    atomic_write_text(
        run_dir(config) / "linguistic-review-instructions.local.md",
        "\n".join(
            [
                "# Slovenian Corpus v2 Review Instructions",
                "",
                "Review every row before any TTS, ASR scoring, selection, or training.",
                "",
                "Judge grammar, agreement, case government, prepositions, entity inflection, semantic plausibility, spoken naturalness, register/domain fit, category correctness, pronunciation plausibility, and transcription correctness.",
                "",
                "Allowed outcomes: ACCEPT, REJECT_GRAMMAR, REJECT_SEMANTICS, REJECT_UNNATURAL, REJECT_TEMPLATE, REJECT_METADATA_LEAK, REJECT_DUPLICATE, REJECT_DOMAIN, REJECT_TRANSCRIPTION, REVISE_AND_REREVIEW.",
                "",
                "Do not preapprove rows. Empty outcome means review is outstanding.",
                "",
            ]
        ),
    )


def fingerprint_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "surface": len({fingerprint_hash(surface_form(row["spoken_text"])) for row in records}),
        "number_masked": len({fingerprint_hash(number_masked_form(row["spoken_text"])) for row in records}),
        "entity_masked": len({fingerprint_hash(entity_masked_form(row["spoken_text"], ())) for row in records}),
        "carrier_stripped": len({fingerprint_hash(entity_masked_form(row["spoken_text"], ())) for row in records}),
    }


def family_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_families = Counter(str(row["source_family_id"]) for row in records)
    discovered = Counter(fingerprint_hash(entity_masked_form(str(row["spoken_text"]), ())) for row in records)
    largest = max(discovered.values(), default=0)
    total = max(1, len(records))
    return {
        "source_family_count": len(source_families),
        "discovered_family_count": len(discovered),
        "largest_family_size": largest,
        "largest_family_fraction": round(largest / total, 6),
    }


def rejection_counts(rejections: Sequence[Rejection] | Sequence[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in rejections:
        if isinstance(item, Rejection):
            counter[item.reason] += 1
        else:
            counter[str(item.get("reason", "unknown"))] += 1
    return dict(sorted(counter.items()))


def read_rejections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def write_rejections(path: Path, rejections: Sequence[Rejection] | Sequence[dict[str, Any]]) -> None:
    rows = [item.to_json() if isinstance(item, Rejection) else item for item in rejections]
    atomic_write_jsonl(path, rows)


def build_summary_payload(config: dict[str, Any]) -> dict[str, Any]:
    generated = load_jsonl(generated_all_path(config)) if generated_all_path(config).exists() else []
    retained = load_jsonl(pre_review_path(config)) if pre_review_path(config).exists() else []
    rejected = read_rejections(rejected_path(config))
    validation = load_json(validation_report_path(config)) if validation_report_path(config).exists() else None
    monitor = summarize_gpu_monitor(gpu_monitor_path(config)) if gpu_monitor_path(config).exists() else {}
    generated_hash = sha256_file(generated_all_path(config)) if generated_all_path(config).exists() else None
    retained_hash = sha256_file(pre_review_path(config)) if pre_review_path(config).exists() else None
    review_hash = sha256_file(review_template_path(config)) if review_template_path(config).exists() else None
    protected_identities = validation.get("protected_indexes", []) if isinstance(validation, dict) else []
    protected_counts = validation.get("protected_overlap_counts", {}) if isinstance(validation, dict) else {}
    candidate_counts_by_cell: dict[str, int] = defaultdict(int)
    retained_counts_by_cell: dict[str, int] = defaultdict(int)
    for row in generated:
        generation = row.get("generation", {})
        candidate_counts_by_cell[str(generation.get("prompt_cell", "unknown"))] += 1
    for row in retained:
        generation = row.get("generation", {})
        retained_counts_by_cell[str(generation.get("prompt_cell", "unknown"))] += 1
    payload = {
        "schema_version": PUBLIC_REPORT_SCHEMA_VERSION,
        "status": "DRAFT — awaiting native-speaker linguistic review",
        "corpus_id": config["corpus_id"],
        "generator_version": CORPUS_V2_GENERATOR_VERSION,
        "model": {
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "license": config["model"]["license"],
        },
        "configuration_sha256": config_sha256(config),
        "prompt_revision": config["prompt_revision"],
        "prompt_cells": len(config["prompt_cells"]),
        "requested_rows": int(config["target_generated_rows"]),
        "minimum_structurally_admissible_rows": int(config["minimum_structurally_admissible_rows"]),
        "generated_count": len(generated),
        "retained_pre_review_count": len(retained),
        "shortfall": max(0, int(config["minimum_structurally_admissible_rows"]) - len(retained)),
        "candidate_file_hashes": {
            "generated_all_sha256": generated_hash,
            "pre_review_candidates_sha256": retained_hash,
            "review_template_sha256": review_hash,
        },
        "rejection_counts_by_reason": rejection_counts(rejected),
        "per_cell_counts": {
            "generated": dict(sorted(candidate_counts_by_cell.items())),
            "retained": dict(sorted(retained_counts_by_cell.items())),
        },
        "fingerprint_unique_counts": fingerprint_counts(retained) if retained else {},
        "family_summary": family_summary(retained) if retained else {},
        "protected_indexes": protected_identities,
        "protected_overlap_counts": protected_counts,
        "validator": {
            "status": validation.get("final_text_status") if isinstance(validation, dict) else "NOT_RUN",
            "decision_reasons": validation.get("decision_reasons", []) if isinstance(validation, dict) else [],
            "checks": validation.get("checks", {}) if isinstance(validation, dict) else {},
            "fuzzy_review_pairs": validation.get("fuzzy_review_pair_counts", {}).get("pairs_requiring_review") if isinstance(validation, dict) else None,
        },
        "review_pack": {
            "rows": len(retained),
            "outcomes_prefilled": False,
        },
        "gpu_monitor": monitor,
        "limitations": [
            "Native-speaker linguistic review is outstanding.",
            "No synthetic holdout exists in this work order.",
            "No TTS synthesis, ASR scoring, acoustic validation, data certificate, or training was performed.",
            "The reservoir is DRAFT and is not TRAINING_ELIGIBLE.",
        ],
    }
    assert_public_summary_safe(payload)
    return payload


def assert_public_summary_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden_patterns = (
        re.compile(r"\bspoken_text\b|\btarget_text\b"),
        re.compile(r"\bgamsv2-cell\d{2}-a\d{2}-o\d{3}\b"),
        re.compile(r"/(?:home|mnt/data|tmp)/"),
    )
    for pattern in forbidden_patterns:
        if pattern.search(serialized):
            raise ValueError(f"public report contains forbidden content matching {pattern.pattern}")


def write_public_reports(config: dict[str, Any]) -> dict[str, Any]:
    payload = build_summary_payload(config)
    atomic_write_json(public_json_report_path(config), payload)
    lines = [
        "# GaMS Corpus-v2 Candidate Reservoir",
        "",
        "Status: DRAFT — awaiting native-speaker linguistic review.",
        "",
        "This report is privacy-safe. It does not include raw generated sentences, candidate IDs, protected references, local paths, raw GaMS output, audio paths, or hypotheses.",
        "",
        "## Identity",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Model: `{payload['model']['repository']}`",
        f"- Revision: `{payload['model']['revision']}`",
        f"- Configuration SHA256: `{payload['configuration_sha256']}`",
        "",
        "## Funnel",
        "",
        f"- Requested rows: {payload['requested_rows']}",
        f"- Raw extracted rows: {payload['generated_count']}",
        f"- Retained pre-review rows: {payload['retained_pre_review_count']}",
        f"- Minimum structurally admissible target: {payload['minimum_structurally_admissible_rows']}",
        f"- Shortfall: {payload['shortfall']}",
        "",
        "## Validation",
        "",
        f"- Validator status: `{payload['validator']['status']}`",
        f"- Decision reasons: `{', '.join(payload['validator']['decision_reasons']) if payload['validator']['decision_reasons'] else 'not run'}`",
        f"- Fuzzy review pairs: {payload['validator']['fuzzy_review_pairs']}",
        "",
        "## Review Pack",
        "",
        f"- Rows: {payload['review_pack']['rows']}",
        "- Review outcomes prefilled: no",
        "- Human linguistic review is required before any TTS, scoring, selection, or training.",
        "",
        "## GPU Measurement",
        "",
        f"- Monitor samples: {payload['gpu_monitor'].get('sample_count', 0)}",
        f"- Mean utilization: {payload['gpu_monitor'].get('mean_utilization_percent', 'not recorded')}",
        f"- Median utilization: {payload['gpu_monitor'].get('median_utilization_percent', 'not recorded')}",
        f"- P95 utilization: {payload['gpu_monitor'].get('p95_utilization_percent', 'not recorded')}",
        f"- Fraction >=80%: {payload['gpu_monitor'].get('fraction_at_or_above_80_percent', 'not recorded')}",
        f"- Peak memory MiB: {payload['gpu_monitor'].get('peak_memory_mib', 'not recorded')}",
        "",
        "## Limitations",
        "",
        "- Native-speaker review remains outstanding.",
        "- No synthetic holdout exists.",
        "- Acoustic suitability remains untested.",
        "- No data is authorized for training.",
        "",
    ]
    atomic_write_text(public_markdown_report_path(config), "\n".join(lines))
    return payload


class GpuMonitor:
    def __init__(self, *, physical_selector: str, output_path: Path, interval_seconds: float) -> None:
        self.physical_selector = physical_selector
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMonitor":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("timestamp,utilization_gpu_percent,memory_used_mib,power_watts\n", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    self.physical_selector,
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                first = completed.stdout.strip().splitlines()[0]
                parts = [part.strip() for part in first.split(",")]
                if len(parts) >= 3:
                    line = f"{time.time():.3f},{parts[0]},{parts[1]},{parts[2]}\n"
                    with self.output_path.open("a", encoding="utf-8") as fp:
                        fp.write(line)
            self._stop.wait(self.interval_seconds)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def summarize_gpu_monitor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    util: list[float] = []
    mem: list[float] = []
    power: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                util.append(float(row["utilization_gpu_percent"]))
                mem.append(float(row["memory_used_mib"]))
                power.append(float(row["power_watts"]))
            except (KeyError, ValueError):
                continue
    if not util:
        return {"sample_count": 0}
    return {
        "sample_count": len(util),
        "mean_utilization_percent": round(statistics.fmean(util), 3),
        "median_utilization_percent": round(statistics.median(util), 3),
        "p95_utilization_percent": round(float(_percentile(util, 0.95) or 0.0), 3),
        "fraction_at_or_above_80_percent": round(sum(1 for value in util if value >= 80.0) / len(util), 6),
        "peak_memory_mib": round(max(mem), 3) if mem else None,
        "mean_power_watts": round(statistics.fmean(power), 3) if power else None,
        "p95_power_watts": round(float(_percentile(power, 0.95) or 0.0), 3) if power else None,
    }


def local_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "run_dir": run_dir(config),
        "raw_generation": raw_generation_dir(config),
        "generated_all": generated_all_path(config),
        "pre_review": pre_review_path(config),
        "rejected": rejected_path(config),
        "review_template": review_template_path(config),
        "review_sheet": review_sheet_path(config),
        "validation_report": validation_report_path(config),
        "local_review": local_review_path(config),
        "gpu_monitor": gpu_monitor_path(config),
    }


def output_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
