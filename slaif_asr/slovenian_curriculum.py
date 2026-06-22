from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slaif_asr.metrics import CorpusMetricSummary, corpus_metric_summary, raw_character_edit_counts, raw_word_edit_counts
from slaif_asr.real_eval import normalize_sl_asr_text, stable_text_hash
from slaif_asr.tts import Candidate


SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
WHITESPACE_PATTERN = re.compile(r"\s+")
SUPPORTED_SCHEMA_VERSION = "1.0"
ALLOWED_PARTITION_ROLES = {"synthetic_holdout", "synthetic_candidate"}
ALLOWED_CATEGORIES = {
    "ordinary",
    "questions_requests",
    "commands",
    "cz_sz_z_coverage",
    "morphology_inflection",
    "dual",
    "function_words_clitics",
    "names_places_institutions",
    "dates_numbers_quantities",
    "technical_code_switching",
}
UNSUPPORTED_SYMBOLS = {"€", "°"}


@dataclass(frozen=True)
class CurriculumRecord:
    schema_version: str
    candidate_id: str
    spoken_text: str
    target_text: str
    language: str
    partition_role: str
    phenomena: tuple[str, ...]
    generation: dict[str, Any]

    @property
    def primary_category(self) -> str:
        return self.phenomena[0]


@dataclass(frozen=True)
class ValidationSummary:
    role: str
    requested_count: int
    valid_count: int
    exact_duplicates: int
    near_duplicates: int
    protected_gate_overlaps: int
    unsupported_symbols: int
    repeated_prefix_violations: int
    repeated_suffix_violations: int
    category_counts: dict[str, int]
    corpus_sha256: str


@dataclass(frozen=True)
class RoundDecision:
    decision: str
    reasons: tuple[str, ...]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        fp.write(text)
        temp_name = fp.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_generated_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value)).strip()


def validate_record(row: dict[str, Any], *, expected_role: str, config: dict[str, Any]) -> CurriculumRecord:
    if row.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    candidate_id = str(row.get("candidate_id", ""))
    if not SAFE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError(f"{candidate_id or '<missing>'}: unsafe candidate_id")
    if row.get("language") != "sl-SI":
        raise ValueError(f"{candidate_id}: language must be sl-SI")
    if row.get("partition_role") != expected_role or expected_role not in ALLOWED_PARTITION_ROLES:
        raise ValueError(f"{candidate_id}: partition_role must be {expected_role}")
    spoken_text = normalize_generated_text(str(row.get("spoken_text", "")))
    target_text = normalize_generated_text(str(row.get("target_text", "")))
    if spoken_text != row.get("spoken_text") or target_text != row.get("target_text"):
        raise ValueError(f"{candidate_id}: text must be NFC and whitespace-normalized")
    if not spoken_text:
        raise ValueError(f"{candidate_id}: empty text")
    if spoken_text != target_text:
        raise ValueError(f"{candidate_id}: spoken_text must equal target_text")
    words = spoken_text.split()
    limits = config["validation"]
    if not int(limits["min_words"]) <= len(words) <= int(limits["max_words"]):
        raise ValueError(f"{candidate_id}: word count outside bounds")
    if not int(limits["min_characters"]) <= len(spoken_text) <= int(limits["max_characters"]):
        raise ValueError(f"{candidate_id}: character count outside bounds")
    lowered = spoken_text.lower()
    for forbidden in limits.get("forbidden_substrings", []):
        if str(forbidden).lower() in lowered:
            raise ValueError(f"{candidate_id}: forbidden substring {forbidden!r}")
    if any(symbol in spoken_text for symbol in UNSUPPORTED_SYMBOLS):
        raise ValueError(f"{candidate_id}: unsupported symbol")
    phenomena = row.get("phenomena")
    if not isinstance(phenomena, list) or not phenomena:
        raise ValueError(f"{candidate_id}: phenomena must be non-empty")
    normalized_phenomena = tuple(str(item) for item in phenomena)
    unknown = sorted(set(normalized_phenomena) - ALLOWED_CATEGORIES)
    if unknown:
        raise ValueError(f"{candidate_id}: unsupported phenomena {unknown}")
    generation = row.get("generation")
    if not isinstance(generation, dict) or generation.get("system") != "project-generated":
        raise ValueError(f"{candidate_id}: generation.system must be project-generated")
    if generation.get("method") != "direct-language-generation":
        raise ValueError(f"{candidate_id}: generation.method must be direct-language-generation")
    return CurriculumRecord(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        candidate_id=candidate_id,
        spoken_text=spoken_text,
        target_text=target_text,
        language="sl-SI",
        partition_role=expected_role,
        phenomena=normalized_phenomena,
        generation=dict(generation),
    )


def load_records(path: Path, *, expected_role: str, config: dict[str, Any]) -> list[CurriculumRecord]:
    records: list[CurriculumRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                records.append(validate_record(json.loads(line), expected_role=expected_role, config=config))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def records_to_rows(records: list[CurriculumRecord]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": record.schema_version,
            "candidate_id": record.candidate_id,
            "spoken_text": record.spoken_text,
            "target_text": record.target_text,
            "language": record.language,
            "partition_role": record.partition_role,
            "phenomena": list(record.phenomena),
            "generation": record.generation,
        }
        for record in records
    ]


def to_tts_candidates(records: list[CurriculumRecord]) -> list[Candidate]:
    return [
        Candidate(
            schema_version=record.schema_version,
            candidate_id=record.candidate_id,
            spoken_text=record.spoken_text,
            target_text=record.target_text,
            language=record.language,
            partition_role=record.partition_role,
            phenomena=record.phenomena,
            generation=dict(record.generation),
        )
        for record in records
    ]


def text_ngrams(text: str, n: int) -> set[str]:
    compact = normalize_sl_asr_text(text).replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def near_duplicate(left: str, right: str, *, n: int, threshold: float) -> bool:
    return jaccard(text_ngrams(left, n), text_ngrams(right, n)) >= threshold


def first_two_words(text: str) -> str:
    return " ".join(normalize_sl_asr_text(text).split()[:2])


def final_three_words(text: str) -> str:
    words = normalize_sl_asr_text(text).split()
    return " ".join(words[-3:])


def protected_hashes_from_metadata(metadata_paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in metadata_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("row_hashes", "segment_hashes", "selected_hashes", "sample_hashes"):
            values = data.get(key)
            if isinstance(values, list):
                hashes.update(str(item) for item in values)
        for item in data.get("items", []) if isinstance(data.get("items"), list) else []:
            if isinstance(item, dict):
                for key in ("row_hash", "segment_hash", "reference_hash", "text_hash"):
                    if key in item:
                        hashes.add(str(item[key]))
    return hashes


def validate_collection(
    records: list[CurriculumRecord],
    *,
    expected_count: int,
    config: dict[str, Any],
    protected_hashes: set[str] | None = None,
    disjoint_records: list[CurriculumRecord] | None = None,
) -> ValidationSummary:
    protected_hashes = protected_hashes or set()
    disjoint_records = disjoint_records or []
    exact_duplicates = 0
    near_duplicates = 0
    protected_overlaps = 0
    unsupported_symbols = 0
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    previous_texts: list[str] = [record.target_text for record in disjoint_records]
    ngram = int(config["validation"]["near_duplicate_char_ngram"])
    threshold = float(config["validation"]["near_duplicate_jaccard_threshold"])
    prefix_counts: dict[str, int] = {}
    suffix_counts: dict[str, int] = {}
    category_counts = {category: 0 for category in ALLOWED_CATEGORIES}

    for record in records:
        if record.candidate_id in seen_ids or normalize_sl_asr_text(record.target_text) in seen_texts:
            exact_duplicates += 1
        seen_ids.add(record.candidate_id)
        seen_texts.add(normalize_sl_asr_text(record.target_text))
        if any(symbol in record.target_text for symbol in UNSUPPORTED_SYMBOLS):
            unsupported_symbols += 1
        if stable_text_hash(record.target_text) in protected_hashes:
            protected_overlaps += 1
        if any(near_duplicate(record.target_text, text, n=ngram, threshold=threshold) for text in previous_texts):
            near_duplicates += 1
        previous_texts.append(record.target_text)
        prefix_counts[first_two_words(record.target_text)] = prefix_counts.get(first_two_words(record.target_text), 0) + 1
        suffix_counts[final_three_words(record.target_text)] = suffix_counts.get(final_three_words(record.target_text), 0) + 1
        for phenomenon in record.phenomena:
            category_counts[phenomenon] = category_counts.get(phenomenon, 0) + 1

    max_prefix = int(config["validation"]["max_same_first_two_words"])
    max_suffix = int(config["validation"]["max_same_final_three_words"])
    repeated_prefix = sum(1 for count in prefix_counts.values() if count > max_prefix)
    repeated_suffix = sum(1 for count in suffix_counts.values() if count > max_suffix)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records_to_rows(records))
    summary = ValidationSummary(
        role=records[0].partition_role if records else "",
        requested_count=expected_count,
        valid_count=len(records),
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        protected_gate_overlaps=protected_overlaps,
        unsupported_symbols=unsupported_symbols,
        repeated_prefix_violations=repeated_prefix,
        repeated_suffix_violations=repeated_suffix,
        category_counts=dict(sorted(category_counts.items())),
        corpus_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, saw {len(records)}")
    if any(
        [
            exact_duplicates,
            near_duplicates,
            protected_overlaps,
            unsupported_symbols,
            repeated_prefix,
            repeated_suffix,
        ]
    ):
        raise ValueError(f"invalid {summary.role} collection: {summary}")
    return summary


def category_counts(records: list[CurriculumRecord]) -> dict[str, int]:
    counts = {category: 0 for category in ALLOWED_CATEGORIES}
    for record in records:
        for phenomenon in record.phenomena:
            counts[phenomenon] = counts.get(phenomenon, 0) + 1
    return dict(sorted(counts.items()))


def assert_quota(counts: dict[str, int], quota: dict[str, int]) -> None:
    for category, expected in quota.items():
        if counts.get(category, 0) != expected:
            raise ValueError(f"{category}: expected {expected}, saw {counts.get(category, 0)}")


def read_scored_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def hard_ranking_key(row: dict[str, Any], *, category_counts_selected: dict[str, int]) -> tuple[Any, ...]:
    phenomena = tuple(row.get("phenomena", []))
    underrepresented = min(category_counts_selected.get(item, 0) for item in phenomena) if phenomena else 0
    return (
        0 if row.get("empty_hypothesis") else 1,
        -float(row.get("normalized_cer", row.get("raw_cer", 0.0))),
        -float(row.get("normalized_wer", row.get("raw_wer", 0.0))),
        -int(row.get("word_deletions", 0)),
        underrepresented,
        str(row["candidate_id"]),
    )


def select_hard_examples(scored: list[dict[str, Any]], *, count: int, category_cap: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_selected = {category: 0 for category in ALLOWED_CATEGORIES}
    while len(selected) < count:
        progress = False
        for row in sorted(scored, key=lambda item: hard_ranking_key(item, category_counts_selected=category_selected)):
            if row["candidate_id"] in selected_ids:
                continue
            primary = str(row.get("phenomena", ["ordinary"])[0])
            if category_selected.get(primary, 0) >= category_cap:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            category_selected[primary] = category_selected.get(primary, 0) + 1
            progress = True
            break
        if not progress:
            for row in sorted(scored, key=lambda item: hard_ranking_key(item, category_counts_selected=category_selected)):
                if row["candidate_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["candidate_id"])
                    break
        if len(selected_ids) == len(scored) and len(selected) < count:
            raise ValueError("not enough scored candidates for hard selection")
    return selected


def select_controls(scored: list[dict[str, Any]], *, exclude_ids: set[str], count: int, seed: int) -> list[dict[str, Any]]:
    pool = [row for row in scored if row["candidate_id"] not in exclude_ids]
    ordered = sorted(
        pool,
        key=lambda row: hashlib.sha256(f"{seed}:{row['candidate_id']}:{row.get('normalized_wer', 0)}".encode("utf-8")).hexdigest(),
    )
    controls = ordered[:count]
    if len(controls) != count:
        raise ValueError(f"expected {count} controls, saw {len(controls)}")
    return controls


def assert_training_disjoint(
    training_ids: set[str],
    *,
    holdout_ids: set[str],
    real_gate_ids: set[str] | None = None,
) -> None:
    holdout_overlap = sorted(training_ids & holdout_ids)
    if holdout_overlap:
        raise ValueError(f"synthetic holdout IDs cannot enter training: {holdout_overlap}")
    real_overlap = sorted(training_ids & (real_gate_ids or set()))
    if real_overlap:
        raise ValueError(f"real gate IDs cannot enter training: {real_overlap}")


def relative_improvement(base: float, challenger: float) -> float:
    return 0.0 if base == 0 else (base - challenger) / base * 100.0


def classify_round1(
    *,
    integrity_passed: bool,
    synthetic_holdout_base: CorpusMetricSummary,
    synthetic_holdout_challenger: CorpusMetricSummary,
    fleurs_base: CorpusMetricSummary,
    fleurs_challenger: CorpusMetricSummary,
    artur_base: CorpusMetricSummary,
    artur_challenger: CorpusMetricSummary,
    thresholds: dict[str, Any],
) -> RoundDecision:
    reasons: list[str] = []
    if not integrity_passed:
        return RoundDecision("EXPERIMENT_INVALID", ("parameter integrity failed",))
    synthetic_wer_improvement = relative_improvement(synthetic_holdout_base.corpus_wer, synthetic_holdout_challenger.corpus_wer)
    synthetic_cer_improvement = relative_improvement(synthetic_holdout_base.corpus_cer, synthetic_holdout_challenger.corpus_cer)
    synthetic_ok = synthetic_wer_improvement >= float(
        thresholds["synthetic_holdout_relative_wer_or_cer_improvement_percent"]
    ) or synthetic_cer_improvement >= float(thresholds["synthetic_holdout_relative_wer_or_cer_improvement_percent"])
    if not synthetic_ok:
        reasons.append("synthetic holdout improvement was below 15% relative WER/CER")
    real_regressed = False
    if fleurs_challenger.corpus_wer - fleurs_base.corpus_wer > float(thresholds["fleurs_max_absolute_wer_regression"]):
        real_regressed = True
        reasons.append("FLEURS normalized corpus WER regressed beyond threshold")
    if artur_challenger.corpus_wer - artur_base.corpus_wer > float(thresholds["artur_j_max_absolute_wer_regression"]):
        real_regressed = True
        reasons.append("ARTUR-J normalized corpus WER regressed beyond threshold")
    if fleurs_challenger.corpus_cer - fleurs_base.corpus_cer > float(thresholds["fleurs_max_absolute_cer_regression"]):
        real_regressed = True
        reasons.append("FLEURS normalized corpus CER regressed beyond threshold")
    if artur_challenger.corpus_cer - artur_base.corpus_cer > float(thresholds["artur_j_max_absolute_cer_regression"]):
        real_regressed = True
        reasons.append("ARTUR-J normalized corpus CER regressed beyond threshold")
    if bool(thresholds["empty_hypotheses_must_not_increase"]):
        if fleurs_challenger.empty_hypothesis_count > fleurs_base.empty_hypothesis_count:
            real_regressed = True
            reasons.append("FLEURS empty-hypothesis count increased")
        if artur_challenger.empty_hypothesis_count > artur_base.empty_hypothesis_count:
            real_regressed = True
            reasons.append("ARTUR-J empty-hypothesis count increased")
    if real_regressed:
        return RoundDecision("ROUND1_REJECTED", tuple(reasons))
    if not synthetic_ok:
        return RoundDecision("ROUND1_REJECTED", tuple(reasons))
    real_improved = (
        fleurs_base.corpus_wer - fleurs_challenger.corpus_wer >= float(thresholds["real_improvement_wer_abs"])
        or artur_base.corpus_wer - artur_challenger.corpus_wer >= float(thresholds["real_improvement_wer_abs"])
        or fleurs_base.corpus_cer - fleurs_challenger.corpus_cer >= float(thresholds["real_improvement_cer_abs"])
        or artur_base.corpus_cer - artur_challenger.corpus_cer >= float(thresholds["real_improvement_cer_abs"])
    )
    if real_improved:
        return RoundDecision("ROUND1_ACCEPTED_REAL_GENERALIZATION", ("synthetic holdout improved and real gate improved",))
    return RoundDecision("ROUND1_SYNTHETIC_ONLY", ("synthetic holdout improved but no real promotion improvement occurred",))


def summary_dict(summary: ValidationSummary) -> dict[str, Any]:
    return asdict(summary)
