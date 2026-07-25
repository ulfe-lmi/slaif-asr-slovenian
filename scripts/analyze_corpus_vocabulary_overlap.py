#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json, atomic_write_text, normalize_sl_asr_text


WORK_ORDER_ID = "0048"
SCALE8000_DATASET_ID = "sl-corpus-v5-scale8000-training-v1"
FLEURS_V2_DATASET_ID = "fleurs-sl-si-test-full-v2"
ARTUR_J_DATASET_ID = "artur-j-public-gate-v1"
EXPECTED_SCALE8000_ROWS = 64_000
EXPECTED_FLEURS_V2_ROWS = 834
EXPECTED_ARTUR_J_ROWS = 256
EXPECTED_SCALE8000_SHA256 = "e76e55ffd12cfa0000a27579566f0a0604a49376a993027663c082cbefd1aadd"
LOW_GATE_COVERAGE_WARNING_THRESHOLD = 0.80

CLASSIFICATION_COMPLETE = "VOCAB_COVERAGE_ANALYSIS_COMPLETE"
CLASSIFICATION_LOW = "VOCAB_COVERAGE_ANALYSIS_COMPLETE_LOW_GATE_COVERAGE"
CLASSIFICATION_BLOCKED = "VOCAB_COVERAGE_ANALYSIS_BLOCKED_INPUT_MISSING"
CLASSIFICATION_LEAKAGE = "VOCAB_COVERAGE_ANALYSIS_INVALID_REAL_GATE_LEAKAGE"
CLASSIFICATION_INVALID = "EXPERIMENT_INVALID"

BLOCKED_SCALE8000 = "BLOCKED_SCALE8000_TEXT_UNAVAILABLE"
BLOCKED_FLEURS = "BLOCKED_FLEURS_V2_REFERENCES_UNAVAILABLE"
BLOCKED_ARTUR = "BLOCKED_ARTUR_J_REFERENCES_UNAVAILABLE"

PRIMARY_MODE = "normalized_word_forms"
SECONDARY_MODE = "normalized_alnum_tokens"
PRIMARY_TOKEN_POLICY = "unicode_letter_tokens_casefold_nfc_no_lemmatization"
SECONDARY_TOKEN_POLICY = "unicode_alnum_tokens_casefold_nfc_numeric_only_included_no_lemmatization"

DEFAULT_SCALE8000_PATH = Path(
    "runs/data-quality/sl-corpus-v5-scale8000-training-v1/fixed-combined-training-text.local.jsonl"
)
DEFAULT_FLEURS_PATH = Path("runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl")
DEFAULT_ARTUR_PATH = Path("runs/evaluation-gates/artur-j-public-gate-v1/manifest.jsonl")
DEFAULT_JSON_OUTPUT = Path("docs/analysis/scale8000-gate-vocabulary-overlap.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/analysis/scale8000-gate-vocabulary-overlap.md")

ALLOWED_CLASSIFICATIONS = {
    CLASSIFICATION_COMPLETE,
    CLASSIFICATION_LOW,
    CLASSIFICATION_BLOCKED,
    CLASSIFICATION_LEAKAGE,
    CLASSIFICATION_INVALID,
}
REQUIRED_REPORT_TOP_LEVEL_KEYS = {
    "schema_version",
    "work_order_id",
    "classification",
    "analysis_role",
    "normalization",
    "datasets",
    "overlap",
    "missing_aggregate",
    "secondary_normalized_alnum_tokens",
    "safety",
}
FORBIDDEN_REPORT_KEYS = {
    "audio_filepath",
    "full_gate_vocabularies",
    "full_gate_vocabulary",
    "full_gate_wordlists",
    "gate_vocabularies",
    "gate_vocabulary",
    "input_path",
    "input_paths",
    "local_path",
    "local_paths",
    "missing_word_forms",
    "missing_words",
    "raw_reference",
    "raw_references",
    "reference",
    "references",
    "source_path",
    "source_paths",
    "spoken_text",
    "target_text",
    "text",
    "texts",
    "tokens",
    "word_forms",
    "wordlist",
    "wordlists",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
EXPECTED_SAFETY_FLAGS = {
    "raw_references_committed",
    "full_gate_wordlists_committed",
    "missing_gate_wordlists_committed",
    "local_paths_committed",
    "used_for_training_selection",
    "used_for_generation_prompting",
    "training_run",
    "asr_evaluation_run",
    "new_text_generated",
}


class InputUnavailableError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class RealGateLeakageError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetBinding:
    key: str
    dataset_id: str
    source_role: str
    source_sha256: str


@dataclass(frozen=True)
class VocabularySummary:
    row_count: int
    token_counts: Counter[str]
    row_token_counts: tuple[int, ...]


@dataclass(frozen=True)
class DatasetVocabulary:
    binding: DatasetBinding
    primary: VocabularySummary
    secondary: VocabularySummary


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def extract_normalized_tokens(text: str, *, include_numeric_only: bool = False) -> list[str]:
    normalized = unicodedata.normalize("NFC", normalize_sl_asr_text(text).casefold())
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        token = "".join(current)
        if include_numeric_only or any(_is_letter(character) for character in token):
            tokens.append(token)
        current.clear()

    for character in normalized:
        if _is_letter(character) or _is_number(character):
            current.append(character)
        else:
            flush()
    flush()
    return tokens


def summarize_texts(texts: Iterable[str], *, include_numeric_only: bool) -> VocabularySummary:
    token_counts: Counter[str] = Counter()
    row_token_counts: list[int] = []
    for text in texts:
        tokens = extract_normalized_tokens(text, include_numeric_only=include_numeric_only)
        token_counts.update(tokens)
        row_token_counts.append(len(tokens))
    return VocabularySummary(
        row_count=len(row_token_counts),
        token_counts=token_counts,
        row_token_counts=tuple(row_token_counts),
    )


def _nearest_rank(values: tuple[int, ...], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def dataset_metrics(summary: VocabularySummary) -> dict[str, int | float]:
    total_tokens = sum(summary.token_counts.values())
    unique_forms = len(summary.token_counts)
    return {
        "row_count": summary.row_count,
        "total_word_tokens": total_tokens,
        "unique_word_forms": unique_forms,
        "type_token_ratio": _safe_ratio(unique_forms, total_tokens),
        "mean_word_tokens_per_row": _safe_ratio(total_tokens, summary.row_count),
        "p50_word_tokens_per_row": _nearest_rank(summary.row_token_counts, 0.50),
        "p95_word_tokens_per_row": _nearest_rank(summary.row_token_counts, 0.95),
    }


def overlap_metrics(left: VocabularySummary, right: VocabularySummary) -> dict[str, int | float]:
    left_forms = set(left.token_counts)
    right_forms = set(right.token_counts)
    shared = left_forms & right_forms
    right_total_tokens = sum(right.token_counts.values())
    right_tokens_covered = sum(right.token_counts[form] for form in shared)
    right_unique_absent = len(right_forms - left_forms)
    right_tokens_absent = right_total_tokens - right_tokens_covered
    return {
        "shared_unique_word_forms": len(shared),
        "right_unique_word_forms": len(right_forms),
        "right_word_tokens": right_total_tokens,
        "coverage_of_right_unique_vocab": _safe_ratio(len(shared), len(right_forms)),
        "right_tokens_covered": right_tokens_covered,
        "token_mass_coverage_of_right_tokens": _safe_ratio(right_tokens_covered, right_total_tokens),
        "right_unique_forms_absent": right_unique_absent,
        "absent_form_rate": _safe_ratio(right_unique_absent, len(right_forms)),
        "right_tokens_using_absent_forms": right_tokens_absent,
        "absent_token_mass_rate": _safe_ratio(right_tokens_absent, right_total_tokens),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_texts(
    path: Path,
    *,
    dataset_id: str,
    text_field: str,
    expected_rows: int,
    expected_dataset_field: str | None,
) -> Iterator[str]:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{dataset_id}: blank JSONL row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{dataset_id}: invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{dataset_id}: row {line_number} must be an object")
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{dataset_id}: row {line_number} has no usable {text_field}")
            if expected_dataset_field is not None and row.get("dataset") != expected_dataset_field:
                raise ValueError(f"{dataset_id}: row {line_number} has an unexpected dataset identity")
            rows += 1
            yield text
    if rows != expected_rows:
        raise ValueError(f"{dataset_id}: expected {expected_rows} rows, observed {rows}")


def load_dataset_vocabulary(
    path: Path,
    *,
    key: str,
    dataset_id: str,
    source_role: str,
    text_field: str,
    expected_rows: int,
    expected_dataset_field: str | None = None,
    expected_sha256: str | None = None,
) -> DatasetVocabulary:
    source_sha256 = sha256_file(path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(f"{dataset_id}: source SHA256 does not match the work-order binding")
    texts = list(
        _jsonl_texts(
            path,
            dataset_id=dataset_id,
            text_field=text_field,
            expected_rows=expected_rows,
            expected_dataset_field=expected_dataset_field,
        )
    )
    binding = DatasetBinding(
        key=key,
        dataset_id=dataset_id,
        source_role=source_role,
        source_sha256=source_sha256,
    )
    return DatasetVocabulary(
        binding=binding,
        primary=summarize_texts(texts, include_numeric_only=False),
        secondary=summarize_texts(texts, include_numeric_only=True),
    )


def _dataset_report(dataset: DatasetVocabulary, *, secondary: bool = False) -> dict[str, str | int | float]:
    summary = dataset.secondary if secondary else dataset.primary
    return {
        "dataset_id": dataset.binding.dataset_id,
        "source_role": dataset.binding.source_role,
        "source_sha256": dataset.binding.source_sha256,
        **dataset_metrics(summary),
    }


def _mode_overlap_report(
    scale8000: VocabularySummary,
    fleurs_v2: VocabularySummary,
    artur_j: VocabularySummary,
) -> dict[str, dict[str, int | float] | int]:
    return {
        "scale8000_to_fleurs_v2": overlap_metrics(scale8000, fleurs_v2),
        "scale8000_to_artur_j": overlap_metrics(scale8000, artur_j),
        "fleurs_v2_to_artur_j": overlap_metrics(fleurs_v2, artur_j),
        "all_three_shared_unique_word_forms": len(
            set(scale8000.token_counts) & set(fleurs_v2.token_counts) & set(artur_j.token_counts)
        ),
    }


def build_aggregate_report(
    scale8000: DatasetVocabulary,
    fleurs_v2: DatasetVocabulary,
    artur_j: DatasetVocabulary,
) -> dict[str, object]:
    primary_overlap = _mode_overlap_report(scale8000.primary, fleurs_v2.primary, artur_j.primary)
    fleurs_coverage = primary_overlap["scale8000_to_fleurs_v2"]
    artur_coverage = primary_overlap["scale8000_to_artur_j"]
    if not isinstance(fleurs_coverage, dict) or not isinstance(artur_coverage, dict):
        raise AssertionError("overlap construction failed")
    classification = (
        CLASSIFICATION_LOW
        if (
            float(fleurs_coverage["coverage_of_right_unique_vocab"]) < LOW_GATE_COVERAGE_WARNING_THRESHOLD
            or float(artur_coverage["coverage_of_right_unique_vocab"]) < LOW_GATE_COVERAGE_WARNING_THRESHOLD
        )
        else CLASSIFICATION_COMPLETE
    )
    datasets = {
        scale8000.binding.key: _dataset_report(scale8000),
        fleurs_v2.binding.key: _dataset_report(fleurs_v2),
        artur_j.binding.key: _dataset_report(artur_j),
    }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "work_order_id": WORK_ORDER_ID,
        "classification": classification,
        "analysis_role": "read_only_aggregate_corpus_diagnostic",
        "normalization": {
            "mode": PRIMARY_MODE,
            "normalizer": NORMALIZER_VERSION,
            "token_policy": PRIMARY_TOKEN_POLICY,
            "secondary_mode": SECONDARY_MODE,
            "secondary_token_policy": SECONDARY_TOKEN_POLICY,
            "percentile_method": "nearest_rank",
        },
        "datasets": datasets,
        "overlap": primary_overlap,
        "missing_aggregate": {
            "fleurs_v2": {
                key: fleurs_coverage[key]
                for key in (
                    "right_unique_forms_absent",
                    "absent_form_rate",
                    "right_tokens_using_absent_forms",
                    "absent_token_mass_rate",
                )
            },
            "artur_j": {
                key: artur_coverage[key]
                for key in (
                    "right_unique_forms_absent",
                    "absent_form_rate",
                    "right_tokens_using_absent_forms",
                    "absent_token_mass_rate",
                )
            },
        },
        "secondary_normalized_alnum_tokens": {
            "datasets": {
                scale8000.binding.key: _dataset_report(scale8000, secondary=True),
                fleurs_v2.binding.key: _dataset_report(fleurs_v2, secondary=True),
                artur_j.binding.key: _dataset_report(artur_j, secondary=True),
            },
            "overlap": _mode_overlap_report(scale8000.secondary, fleurs_v2.secondary, artur_j.secondary),
        },
        "safety": {
            "raw_references_committed": False,
            "full_gate_wordlists_committed": False,
            "missing_gate_wordlists_committed": False,
            "local_paths_committed": False,
            "used_for_training_selection": False,
            "used_for_generation_prompting": False,
            "training_run": False,
            "asr_evaluation_run": False,
            "new_text_generated": False,
        },
    }
    assert_aggregate_only_report(report)
    return report


def assert_aggregate_only_report(report: Mapping[str, object]) -> None:
    def walk(value: object, *, key: str = "") -> None:
        if key in FORBIDDEN_REPORT_KEYS:
            raise RealGateLeakageError(f"aggregate report contains forbidden field: {key}")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if not isinstance(child_key, str):
                    raise ValueError("aggregate report keys must be strings")
                walk(child_value, key=child_key)
            return
        if isinstance(value, (list, tuple, set)):
            raise RealGateLeakageError("aggregate report must not contain arrays or word lists")
        if isinstance(value, str):
            if value.startswith(("/", "file://")) or WINDOWS_ABSOLUTE_PATH.match(value):
                raise RealGateLeakageError("aggregate report contains a local absolute path")
            if "\n" in value or "\r" in value:
                raise ValueError("aggregate report strings must be single-line metadata")
            return
        if not isinstance(value, (bool, int, float)) and value is not None:
            raise ValueError("aggregate report contains an unsupported value type")

    walk(report)
    if set(report) != REQUIRED_REPORT_TOP_LEVEL_KEYS:
        raise ValueError("aggregate report has unexpected top-level fields")
    if report.get("classification") not in ALLOWED_CLASSIFICATIONS:
        raise ValueError("aggregate report has an invalid classification")
    safety = report.get("safety")
    if not isinstance(safety, Mapping) or set(safety) != EXPECTED_SAFETY_FLAGS:
        raise ValueError("aggregate report has invalid safety fields")
    leakage_flags = {
        "raw_references_committed",
        "full_gate_wordlists_committed",
        "missing_gate_wordlists_committed",
        "local_paths_committed",
    }
    if any(safety.get(flag) is not False for flag in leakage_flags):
        raise RealGateLeakageError("aggregate report declares real-gate or local-path leakage")
    if any(safety.get(flag) is not False for flag in EXPECTED_SAFETY_FLAGS - leakage_flags):
        raise ValueError("aggregate report violates the read-only diagnostic boundary")


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def _decimal(value: object, places: int = 4) -> str:
    return f"{float(value):.{places}f}"


def render_markdown(report: Mapping[str, object]) -> str:
    assert_aggregate_only_report(report)
    datasets = report["datasets"]
    overlap = report["overlap"]
    missing = report["missing_aggregate"]
    secondary = report["secondary_normalized_alnum_tokens"]
    if not all(isinstance(item, Mapping) for item in (datasets, overlap, missing, secondary)):
        raise ValueError("aggregate report sections are malformed")

    scale = datasets["scale8000"]
    fleurs = datasets["fleurs_v2"]
    artur = datasets["artur_j"]
    scale_fleurs = overlap["scale8000_to_fleurs_v2"]
    scale_artur = overlap["scale8000_to_artur_j"]
    fleurs_artur = overlap["fleurs_v2_to_artur_j"]
    fleurs_missing = missing["fleurs_v2"]
    artur_missing = missing["artur_j"]
    secondary_datasets = secondary["datasets"]
    secondary_overlap = secondary["overlap"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            scale,
            fleurs,
            artur,
            scale_fleurs,
            scale_artur,
            fleurs_artur,
            fleurs_missing,
            artur_missing,
            secondary_datasets,
            secondary_overlap,
        )
    ):
        raise ValueError("aggregate report values are malformed")

    def dataset_row(label: str, item: Mapping[str, object]) -> str:
        return (
            f"| {label} | {item['row_count']} | {item['total_word_tokens']} | "
            f"{item['unique_word_forms']} | {_decimal(item['type_token_ratio'], 6)} | "
            f"{_decimal(item['mean_word_tokens_per_row'], 2)} | "
            f"{item['p50_word_tokens_per_row']} | {item['p95_word_tokens_per_row']} |"
        )

    def overlap_row(label: str, item: Mapping[str, object]) -> str:
        return (
            f"| {label} | {item['shared_unique_word_forms']} | "
            f"{_percent(item['coverage_of_right_unique_vocab'])} | "
            f"{_percent(item['token_mass_coverage_of_right_tokens'])} |"
        )

    def missing_row(label: str, item: Mapping[str, object]) -> str:
        return (
            f"| {label} | {item['right_unique_forms_absent']} | "
            f"{_percent(item['absent_form_rate'])} | "
            f"{item['right_tokens_using_absent_forms']} | "
            f"{_percent(item['absent_token_mass_rate'])} |"
        )

    def lexical_interpretation(label: str, item: Mapping[str, object]) -> str:
        coverage = float(item["coverage_of_right_unique_vocab"])
        threshold_result = "clears" if coverage >= LOW_GATE_COVERAGE_WARNING_THRESHOLD else "falls below"
        breadth_result = "broad enough" if coverage >= LOW_GATE_COVERAGE_WARNING_THRESHOLD else "not broad enough"
        return (
            f"Scale-8000 covers {_percent(coverage)} of {label} unique normalized forms, which "
            f"{threshold_result} the precommitted 80% warning threshold. It is {breadth_result} relative to "
            f"{label} under this narrow surface-form test."
        )
    token_mass_floor = min(
        float(scale_fleurs["token_mass_coverage_of_right_tokens"]),
        float(scale_artur["token_mass_coverage_of_right_tokens"]),
    )
    missing_interpretation = (
        "The high covered token mass does not identify missing surface forms as a dominant remaining error source; "
        "this vocabulary-only analysis cannot attribute ASR errors."
        if token_mass_floor >= 0.90
        else "The uncovered token mass is large enough to remain a plausible lexical limitation, but this "
        "vocabulary-only analysis cannot attribute ASR errors."
    )

    secondary_scale = secondary_datasets["scale8000"]
    secondary_fleurs = secondary_datasets["fleurs_v2"]
    secondary_artur = secondary_datasets["artur_j"]
    secondary_scale_fleurs = secondary_overlap["scale8000_to_fleurs_v2"]
    secondary_scale_artur = secondary_overlap["scale8000_to_artur_j"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            secondary_scale,
            secondary_fleurs,
            secondary_artur,
            secondary_scale_fleurs,
            secondary_scale_artur,
        )
    ):
        raise ValueError("secondary aggregate report values are malformed")

    lines = [
        "# Scale-8000 vs FLEURS/ARTUR-J Vocabulary Coverage",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "This Work Order 0048 result is a read-only aggregate corpus diagnostic. "
        "It did not train a model, evaluate ASR, generate text, select examples, or create a curriculum. "
        "No real-gate reference, vocabulary, missing-word list, local manifest, or local path is committed.",
        "",
        "## Input bindings",
        "",
        "| Dataset | Role | Rows | Source SHA256 |",
        "|---|---|---:|---|",
        f"| scale-8000 | synthetic training text | {scale['row_count']} | `{scale['source_sha256']}` |",
        f"| FLEURS-v2 | immutable real gate | {fleurs['row_count']} | `{fleurs['source_sha256']}` |",
        f"| ARTUR-J | immutable real gate | {artur['row_count']} | `{artur['source_sha256']}` |",
        "",
        "## Normalization and tokenization",
        "",
        f"- Normalizer: `{NORMALIZER_VERSION}`.",
        "- Primary mode: `normalized_word_forms`.",
        "- Token policy: NFC, repository normalization, casefolding, punctuation splitting, "
        "at least one Unicode letter, Slovenian diacritics preserved, and numeric-only tokens excluded.",
        "- No lemmatization or stemming.",
        "- P50 and P95 use the nearest-rank method.",
        "",
        "## Dataset sizes",
        "",
        "| Dataset | Rows | Total word tokens | Unique word forms | Type-token ratio | Mean words/row | P50 words/row | P95 words/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        dataset_row("scale-8000", scale),
        dataset_row("FLEURS-v2", fleurs),
        dataset_row("ARTUR-J", artur),
        "",
        "## Vocabulary overlap",
        "",
        "| Comparison | Shared unique word forms | Coverage of right-side unique vocab | Token-mass coverage of right-side tokens |",
        "|---|---:|---:|---:|",
        overlap_row("scale-8000 → FLEURS-v2", scale_fleurs),
        overlap_row("scale-8000 → ARTUR-J", scale_artur),
        overlap_row("FLEURS-v2 ↔ ARTUR-J", fleurs_artur),
        "",
        "For the first two rows, the right side is the real gate. In the third row, ARTUR-J is the "
        "right-side denominator. The three-way intersection contains "
        f"{overlap['all_three_shared_unique_word_forms']} unique normalized word forms.",
        "",
        "## Missing aggregate only",
        "",
        "| Gate | Unique forms absent from scale-8000 | Absent-form rate | Gate tokens using absent forms | Absent token-mass rate |",
        "|---|---:|---:|---:|---:|",
        missing_row("FLEURS-v2", fleurs_missing),
        missing_row("ARTUR-J", artur_missing),
        "",
        "No missing form is included in this report.",
        "",
        "## Secondary normalized alphanumeric mode",
        "",
        "This optional mode includes numeric-only tokens and remains aggregate-only.",
        "",
        "| Dataset | Total alphanumeric tokens | Unique alphanumeric forms |",
        "|---|---:|---:|",
        f"| scale-8000 | {secondary_scale['total_word_tokens']} | {secondary_scale['unique_word_forms']} |",
        f"| FLEURS-v2 | {secondary_fleurs['total_word_tokens']} | {secondary_fleurs['unique_word_forms']} |",
        f"| ARTUR-J | {secondary_artur['total_word_tokens']} | {secondary_artur['unique_word_forms']} |",
        "",
        "| Comparison | Unique-form coverage | Token-mass coverage |",
        "|---|---:|---:|",
        f"| scale-8000 → FLEURS-v2 | {_percent(secondary_scale_fleurs['coverage_of_right_unique_vocab'])} | "
        f"{_percent(secondary_scale_fleurs['token_mass_coverage_of_right_tokens'])} |",
        f"| scale-8000 → ARTUR-J | {_percent(secondary_scale_artur['coverage_of_right_unique_vocab'])} | "
        f"{_percent(secondary_scale_artur['token_mass_coverage_of_right_tokens'])} |",
        "",
        "## Interpretation",
        "",
        f"- {lexical_interpretation('FLEURS-v2', scale_fleurs)}",
        f"- {lexical_interpretation('ARTUR-J', scale_artur)}",
        f"- {missing_interpretation}",
        "- These results measure only normalized surface vocabulary. They do not measure meaning, pronunciation, "
        "acoustics, or ASR quality, and distinct Slovenian inflections remain distinct forms.",
        "- No corpus change or generation steering is authorized by this result. A safe follow-up would use "
        "independently sourced or independently authored Slovenian material to test lexical and morphological "
        "breadth, without transferring real-gate words, sentences, or missing-form lists into prompts or selection.",
        "",
        "## Safety",
        "",
        "- Raw FLEURS/ARTUR-J references committed: no.",
        "- Full or missing real-gate word lists committed: no.",
        "- Local manifests or absolute paths committed: no.",
        "- Used for training selection or curriculum construction: no.",
        "- Used for GaMS prompting or new text generation: no.",
        "- Training or ASR evaluation run: no.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aggregate-only vocabulary coverage for Work Order 0048.")
    parser.add_argument("--scale8000-text", type=Path, default=DEFAULT_SCALE8000_PATH)
    parser.add_argument("--fleurs-v2-manifest", type=Path, default=DEFAULT_FLEURS_PATH)
    parser.add_argument("--artur-j-manifest", type=Path, default=DEFAULT_ARTUR_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def _require_input(path: Path, reason_code: str) -> None:
    if not path.is_file():
        raise InputUnavailableError(reason_code)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    _require_input(args.scale8000_text, BLOCKED_SCALE8000)
    _require_input(args.fleurs_v2_manifest, BLOCKED_FLEURS)
    _require_input(args.artur_j_manifest, BLOCKED_ARTUR)
    scale8000 = load_dataset_vocabulary(
        args.scale8000_text,
        key="scale8000",
        dataset_id=SCALE8000_DATASET_ID,
        source_role="synthetic_training_text",
        text_field="target_text",
        expected_rows=EXPECTED_SCALE8000_ROWS,
        expected_sha256=EXPECTED_SCALE8000_SHA256,
    )
    fleurs_v2 = load_dataset_vocabulary(
        args.fleurs_v2_manifest,
        key="fleurs_v2",
        dataset_id=FLEURS_V2_DATASET_ID,
        source_role="immutable_real_gate",
        text_field="text",
        expected_rows=EXPECTED_FLEURS_V2_ROWS,
        expected_dataset_field=FLEURS_V2_DATASET_ID,
    )
    artur_j = load_dataset_vocabulary(
        args.artur_j_manifest,
        key="artur_j",
        dataset_id=ARTUR_J_DATASET_ID,
        source_role="immutable_real_gate",
        text_field="text",
        expected_rows=EXPECTED_ARTUR_J_ROWS,
        expected_dataset_field=ARTUR_J_DATASET_ID,
    )
    report = build_aggregate_report(scale8000, fleurs_v2, artur_j)
    atomic_write_json(args.output_json, report)
    atomic_write_text(args.output_markdown, render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_analysis(args)
    except InputUnavailableError as error:
        print(f"{CLASSIFICATION_BLOCKED}: {error.reason_code}", file=sys.stderr)
        return 2
    except RealGateLeakageError as error:
        print(f"{CLASSIFICATION_LEAKAGE}: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"{CLASSIFICATION_INVALID}: {error}", file=sys.stderr)
        return 1
    print(report["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
