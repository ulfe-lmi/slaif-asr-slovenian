from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slaif_asr.real_eval import NORMALIZER_VERSION, normalize_sl_asr_text


ALGORITHM_VERSION = "training-text-validator-v1"
REPORT_SCHEMA_VERSION = "1.0"
TEXT_RECORD_SCHEMA_VERSION = "2.0"
PROTECTED_INDEX_SCHEMA_VERSION = "1.0"

STATUS_DRAFT = "DRAFT"
STATUS_TEXT_REJECTED = "TEXT_REJECTED"
STATUS_TEXT_ACCEPTED = "TEXT_ACCEPTED"
STATUS_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
STATUS_RETIRED = "RETIRED"

EMITTABLE_STATUSES = {
    STATUS_DRAFT,
    STATUS_TEXT_REJECTED,
    STATUS_TEXT_ACCEPTED,
    STATUS_DIAGNOSTIC_ONLY,
    STATUS_RETIRED,
}
IMPOSSIBLE_TEXT_VALIDATOR_STATUSES = {
    "AUDIO_REJECTED",
    "AUDIO_ACCEPTED",
    "TRAINING_ELIGIBLE",
}

SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_OR_MARKUP_PATTERN = re.compile(r"https?://|www\.|```|<[^>]+>")
DIGIT_SEQUENCE_PATTERN = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}[:.]\d{2}\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b")
QUANTITY_PATTERN = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:kg|g|km|m|cm|mm|l|ml|%)\b", re.IGNORECASE)
METADATA_TOKEN_PATTERNS = (
    re.compile(
        r"\b(?:candidate|row|group|batch|sample|skupina|vrstica|kandidat|vzorec)\s*[-_.:#]*\s*[a-z]*\d+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bid\s*[-_.:#]*\s*\d+\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class TextRecord:
    schema_version: str
    candidate_id: str
    language: str
    spoken_text: str
    target_text: str
    partition_role: str
    source_type: str
    source_id: str
    source_family_id: str
    template_family_id: str | None
    utterance_family_id: str
    phenomena: tuple[str, ...]
    domain: str
    license: str
    generation: dict[str, Any] | None
    entities: tuple[dict[str, str], ...]
    minimal_pair: dict[str, str] | None
    optional_metadata: dict[str, Any]


@dataclass(frozen=True)
class LinguisticReview:
    candidate_id: str
    outcome: str
    review_revision: str
    reason_codes: tuple[str, ...]
    minimal_pair_approved: bool


@dataclass(frozen=True)
class Fingerprints:
    surface: str
    number_masked: str
    entity_masked: str
    carrier_stripped: str
    discovered_template_family: str


@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    severity: str
    check: str
    partition_role: str | None = None
    candidate_id: str | None = None
    related_candidate_id: str | None = None
    detail: str | None = None

    def public(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ProtectedTextIndex:
    gate_id: str
    manifest_sha256: str
    metadata_manifest_sha256: str
    reference_manifest_sha256: str | None
    metadata_reference_manifest_sha256: str | None
    row_count: int
    normalizer: str
    algorithm_version: str
    surface_hashes: set[str]
    number_masked_hashes: set[str]

    def public_identity(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "manifest_sha256": self.manifest_sha256,
            "metadata_manifest_sha256": self.metadata_manifest_sha256,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "metadata_reference_manifest_sha256": self.metadata_reference_manifest_sha256,
            "row_count": self.row_count,
            "normalizer": self.normalizer,
            "algorithm_version": self.algorithm_version,
            "surface_hash_count": len(self.surface_hashes),
            "number_masked_hash_count": len(self.number_masked_hashes),
        }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        fp.write(text)
        temp_name = fp.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: column {exc.colno}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def normalize_record_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value)).strip()


def surface_form(text: str) -> str:
    return normalize_sl_asr_text(text)


def number_masked_form(text: str) -> str:
    value = surface_form(text)
    value = DATE_PATTERN.sub("<DATE>", value)
    value = TIME_PATTERN.sub("<TIME>", value)
    value = QUANTITY_PATTERN.sub("<QUANTITY>", value)
    value = DIGIT_SEQUENCE_PATTERN.sub("<NUM>", value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _replace_longest(value: str, surface: str, marker: str) -> str:
    escaped = re.escape(surface_form(surface))
    return re.sub(rf"\b{escaped}\b", marker, value, flags=re.IGNORECASE)


def entity_masked_form(text: str, entities: tuple[dict[str, str], ...] = ()) -> str:
    value = number_masked_form(text)
    normalized_entities = sorted(
        entities,
        key=lambda item: len(surface_form(str(item.get("surface", "")))),
        reverse=True,
    )
    for entity in normalized_entities:
        surface = str(entity.get("surface", "")).strip()
        entity_type = str(entity.get("type", "ENTITY")).strip().upper() or "ENTITY"
        if surface:
            value = _replace_longest(value, surface, f"<{entity_type}>")
    value = DATE_PATTERN.sub("<DATE>", value)
    value = TIME_PATTERN.sub("<TIME>", value)
    value = QUANTITY_PATTERN.sub("<QUANTITY>", value)
    value = DIGIT_SEQUENCE_PATTERN.sub("<NUM>", value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def fingerprint_hash(value: str) -> str:
    return sha256_text(value)


def token_shingles(text: str, n: int) -> set[str]:
    tokens = text.split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def character_shingles(text: str, n: int = 5) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def assert_safe_id(value: str, field: str, candidate_id: str) -> None:
    if not value or not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{candidate_id}: unsafe {field}: {value!r}")


def validate_text_record(row: dict[str, Any], *, expected_role: str, config: dict[str, Any]) -> TextRecord:
    candidate_id = str(row.get("candidate_id", ""))
    if row.get("schema_version") != TEXT_RECORD_SCHEMA_VERSION:
        raise ValueError(f"{candidate_id or '<missing>'}: schema_version must be {TEXT_RECORD_SCHEMA_VERSION}")
    assert_safe_id(candidate_id, "candidate_id", candidate_id or "<missing>")
    if row.get("language") != "sl-SI":
        raise ValueError(f"{candidate_id}: language must be sl-SI")
    allowed_roles = set(config["record_contract"]["partition_roles"])
    if expected_role not in allowed_roles or row.get("partition_role") != expected_role:
        raise ValueError(f"{candidate_id}: partition_role must be {expected_role}")
    source_type = str(row.get("source_type", ""))
    if source_type not in set(config["record_contract"]["source_types"]):
        raise ValueError(f"{candidate_id}: unsupported source_type {source_type!r}")

    spoken_text = normalize_record_text(str(row.get("spoken_text", "")))
    target_text = normalize_record_text(str(row.get("target_text", "")))
    if spoken_text != row.get("spoken_text") or target_text != row.get("target_text"):
        raise ValueError(f"{candidate_id}: text must be NFC and whitespace-normalized")
    if not spoken_text or not target_text:
        raise ValueError(f"{candidate_id}: spoken_text and target_text are required")
    if config["record_contract"].get("spoken_text_must_equal_target_text", True) and spoken_text != target_text:
        raise ValueError(f"{candidate_id}: spoken_text must equal target_text")

    limits = config["record_contract"]["length_limits"]
    words = spoken_text.split()
    if not int(limits["min_words"]) <= len(words) <= int(limits["max_words"]):
        raise ValueError(f"{candidate_id}: word count outside limits")
    if not int(limits["min_characters"]) <= len(spoken_text) <= int(limits["max_characters"]):
        raise ValueError(f"{candidate_id}: character count outside limits")
    if URL_OR_MARKUP_PATTERN.search(spoken_text):
        raise ValueError(f"{candidate_id}: URLs or markup are forbidden")
    for symbol in config["record_contract"].get("forbidden_symbols", []):
        if str(symbol) in spoken_text:
            raise ValueError(f"{candidate_id}: unsupported symbol {symbol!r}")

    source_id = str(row.get("source_id", ""))
    source_family_id = str(row.get("source_family_id", ""))
    utterance_family_id = str(row.get("utterance_family_id", ""))
    assert_safe_id(source_id, "source_id", candidate_id)
    assert_safe_id(source_family_id, "source_family_id", candidate_id)
    assert_safe_id(utterance_family_id, "utterance_family_id", candidate_id)
    template_family_id = row.get("template_family_id")
    if template_family_id is not None:
        template_family_id = str(template_family_id)
        assert_safe_id(template_family_id, "template_family_id", candidate_id)

    phenomena = row.get("phenomena")
    if not isinstance(phenomena, list):
        raise ValueError(f"{candidate_id}: phenomena must be a list")
    normalized_phenomena = tuple(str(item) for item in phenomena)
    domain = str(row.get("domain", "")).strip()
    license_name = str(row.get("license", "")).strip()
    if not domain or not license_name:
        raise ValueError(f"{candidate_id}: domain and license are required")

    generation = row.get("generation")
    if source_type == "generated_text":
        if not isinstance(generation, dict):
            raise ValueError(f"{candidate_id}: generation is required for generated_text")
        generation_payload: dict[str, Any] | None = dict(generation)
    else:
        if generation is not None:
            raise ValueError(f"{candidate_id}: generation must be null for authentic material")
        generation_payload = None

    entities_value = row.get("entities", [])
    if not isinstance(entities_value, list):
        raise ValueError(f"{candidate_id}: entities must be a list")
    entities: list[dict[str, str]] = []
    for entity in entities_value:
        if not isinstance(entity, dict):
            raise ValueError(f"{candidate_id}: entity annotations must be objects")
        surface = normalize_record_text(str(entity.get("surface", "")))
        entity_type = str(entity.get("type", "")).strip().upper()
        if not surface or not entity_type:
            raise ValueError(f"{candidate_id}: entity surface and type are required")
        entities.append({"surface": surface, "type": entity_type})

    minimal_pair = row.get("minimal_pair")
    minimal_payload: dict[str, str] | None
    if minimal_pair is None:
        minimal_payload = None
    elif isinstance(minimal_pair, dict):
        family_id = str(minimal_pair.get("family_id", ""))
        contrast = str(minimal_pair.get("contrast", "")).strip()
        assert_safe_id(family_id, "minimal_pair.family_id", candidate_id)
        if not contrast:
            raise ValueError(f"{candidate_id}: minimal_pair.contrast is required")
        minimal_payload = {"family_id": family_id, "contrast": contrast}
    else:
        raise ValueError(f"{candidate_id}: minimal_pair must be null or an object")

    optional_metadata = {
        key: row[key]
        for key in ("source_recording_id", "selection_source_id", "scored_candidate_id")
        if key in row
    }
    return TextRecord(
        schema_version=TEXT_RECORD_SCHEMA_VERSION,
        candidate_id=candidate_id,
        language="sl-SI",
        spoken_text=spoken_text,
        target_text=target_text,
        partition_role=expected_role,
        source_type=source_type,
        source_id=source_id,
        source_family_id=source_family_id,
        template_family_id=template_family_id,
        utterance_family_id=utterance_family_id,
        phenomena=normalized_phenomena,
        domain=domain,
        license=license_name,
        generation=generation_payload,
        entities=tuple(entities),
        minimal_pair=minimal_payload,
        optional_metadata=optional_metadata,
    )


def metadata_leakage(text: str) -> bool:
    normalized = surface_form(text)
    return any(pattern.search(normalized) for pattern in METADATA_TOKEN_PATTERNS)


def load_linguistic_reviews(path: Path) -> dict[str, LinguisticReview]:
    allowed = {
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
    reviews: dict[str, LinguisticReview] = {}
    for line_number, row in enumerate(load_jsonl(path), start=1):
        candidate_id = str(row.get("candidate_id", ""))
        if not SAFE_ID_PATTERN.fullmatch(candidate_id):
            raise ValueError(f"{path}:{line_number}: unsafe candidate_id")
        if candidate_id in reviews:
            raise ValueError(f"{path}:{line_number}: duplicate review for {candidate_id}")
        outcome = str(row.get("outcome", ""))
        if outcome not in allowed:
            raise ValueError(f"{path}:{line_number}: unsupported review outcome {outcome!r}")
        reason_codes = row.get("reason_codes", [])
        if not isinstance(reason_codes, list):
            raise ValueError(f"{path}:{line_number}: reason_codes must be a list")
        reviews[candidate_id] = LinguisticReview(
            candidate_id=candidate_id,
            outcome=outcome,
            review_revision=str(row.get("review_revision", "")),
            reason_codes=tuple(str(item) for item in reason_codes),
            minimal_pair_approved=bool(row.get("minimal_pair_approved", False)),
        )
    return reviews


def minimal_pair_approved(record: TextRecord, reviews: dict[str, LinguisticReview]) -> bool:
    if record.minimal_pair is None:
        return False
    review = reviews.get(record.candidate_id)
    return bool(review and review.outcome == "ACCEPT" and review.minimal_pair_approved)


def group_all(records_by_role: dict[str, list[TextRecord]]) -> list[TextRecord]:
    records: list[TextRecord] = []
    for role in sorted(records_by_role):
        records.extend(sorted(records_by_role[role], key=lambda item: item.candidate_id))
    return records


def _tokens(text: str) -> list[str]:
    return text.split()


def _prefix(tokens: list[str], n: int) -> str:
    return " ".join(tokens[:n])


def _suffix(tokens: list[str], n: int) -> str:
    return " ".join(tokens[-n:])


def discover_carriers(
    records_by_role: dict[str, list[TextRecord]],
    base_forms: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], list[DataQualityIssue], list[dict[str, Any]]]:
    threshold = float(config["carrier_detection"]["max_fraction"])
    min_count = int(config["carrier_detection"]["min_count"])
    token_lengths = list(range(int(config["carrier_detection"]["min_tokens"]), int(config["carrier_detection"]["max_tokens"]) + 1))
    issues: list[DataQualityIssue] = []
    statistics_rows: list[dict[str, Any]] = []
    carriers_by_role: dict[str, dict[str, list[str]]] = {}

    prohibited_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config["carrier_detection"].get("prohibited_patterns", [])]
    for record in group_all(records_by_role):
        normalized = number_masked_form(record.target_text)
        for pattern in prohibited_patterns:
            if pattern.search(normalized):
                issues.append(
                    DataQualityIssue(
                        code="prohibited_carrier",
                        severity="failed",
                        check="carrier_detection",
                        partition_role=record.partition_role,
                        candidate_id=record.candidate_id,
                        detail="configured prohibited carrier pattern matched",
                    )
                )

    for role, records in sorted(records_by_role.items()):
        prefixes: Counter[str] = Counter()
        suffixes: Counter[str] = Counter()
        frames: Counter[str] = Counter()
        internal_ngrams: Counter[str] = Counter()
        for record in records:
            tokens = _tokens(base_forms[record.candidate_id])
            for n in token_lengths:
                if len(tokens) >= n:
                    prefixes[_prefix(tokens, n)] += 1
                    suffixes[_suffix(tokens, n)] += 1
                if len(tokens) >= n * 2:
                    frames[f"{_prefix(tokens, n)} || {_suffix(tokens, n)}"] += 1
                if len(tokens) >= n:
                    for index in range(len(tokens) - n + 1):
                        internal_ngrams[" ".join(tokens[index : index + n])] += 1

        row_count = max(1, len(records))
        hard_prefixes = [item for item, count in prefixes.items() if count >= min_count and count / row_count > threshold]
        hard_suffixes = [item for item, count in suffixes.items() if count >= min_count and count / row_count > threshold]
        hard_frames = [item for item, count in frames.items() if count >= min_count and count / row_count > threshold]
        suspicious_ngrams = [item for item, count in internal_ngrams.items() if count >= min_count and count / row_count > threshold]

        carriers_by_role[role] = {
            "prefixes": sorted(hard_prefixes, key=lambda item: (-len(item.split()), item)),
            "suffixes": sorted(hard_suffixes, key=lambda item: (-len(item.split()), item)),
        }
        for kind, values, severity in (
            ("prefix", hard_prefixes, "failed"),
            ("suffix", hard_suffixes, "failed"),
            ("frame", hard_frames, "failed"),
            ("internal_ngram", suspicious_ngrams, "review_required"),
        ):
            for value in sorted(values):
                count = {
                    "prefix": prefixes,
                    "suffix": suffixes,
                    "frame": frames,
                    "internal_ngram": internal_ngrams,
                }[kind][value]
                statistics_rows.append(
                    {
                        "role": role,
                        "kind": kind,
                        "fingerprint_sha256": fingerprint_hash(value),
                        "count": count,
                        "fraction": round(count / row_count, 6),
                    }
                )
                issues.append(
                    DataQualityIssue(
                        code=f"{kind}_concentration",
                        severity=severity,
                        check="carrier_detection",
                        partition_role=role,
                        detail=f"{kind} concentration {count}/{row_count}",
                    )
                )
    return carriers_by_role, issues, statistics_rows


def strip_carrier(text: str, carriers: dict[str, list[str]]) -> str:
    tokens = _tokens(text)
    for prefix in carriers.get("prefixes", []):
        prefix_tokens = prefix.split()
        if tokens[: len(prefix_tokens)] == prefix_tokens:
            tokens = tokens[len(prefix_tokens) :]
            break
    for suffix in carriers.get("suffixes", []):
        suffix_tokens = suffix.split()
        if suffix_tokens and tokens[-len(suffix_tokens) :] == suffix_tokens:
            tokens = tokens[: -len(suffix_tokens)]
            break
    return " ".join(tokens).strip()


def compute_fingerprints(
    records_by_role: dict[str, list[TextRecord]],
    config: dict[str, Any],
) -> tuple[dict[str, Fingerprints], list[DataQualityIssue], list[dict[str, Any]]]:
    entity_forms = {
        record.candidate_id: entity_masked_form(record.target_text, record.entities)
        for record in group_all(records_by_role)
    }
    carriers_by_role, carrier_issues, carrier_statistics = discover_carriers(records_by_role, entity_forms, config)
    fingerprints: dict[str, Fingerprints] = {}
    for record in group_all(records_by_role):
        surface = surface_form(record.target_text)
        number_masked = number_masked_form(record.target_text)
        entity_masked = entity_forms[record.candidate_id]
        carrier_stripped = strip_carrier(entity_masked, carriers_by_role.get(record.partition_role, {}))
        template_basis = carrier_stripped or entity_masked
        discovered = f"dtf-{fingerprint_hash(template_basis)[:16]}"
        fingerprints[record.candidate_id] = Fingerprints(
            surface=fingerprint_hash(surface),
            number_masked=fingerprint_hash(number_masked),
            entity_masked=fingerprint_hash(entity_masked),
            carrier_stripped=fingerprint_hash(carrier_stripped),
            discovered_template_family=discovered,
        )
    return fingerprints, carrier_issues, carrier_statistics


def _group_by(records: list[TextRecord], key_fn: Any) -> dict[str, list[TextRecord]]:
    grouped: dict[str, list[TextRecord]] = defaultdict(list)
    for record in records:
        grouped[str(key_fn(record))].append(record)
    return grouped


def _all_minimal_pair_approved(group: list[TextRecord], reviews: dict[str, LinguisticReview], config: dict[str, Any]) -> bool:
    if not group:
        return False
    if any(record.minimal_pair is None for record in group):
        return False
    family_ids = {record.minimal_pair["family_id"] for record in group if record.minimal_pair}
    if len(family_ids) != 1:
        return False
    if len({record.partition_role for record in group}) != 1:
        return False
    if len(group) > int(config["minimal_pair"]["max_family_size"]):
        return False
    return all(minimal_pair_approved(record, reviews) for record in group)


def validate_exact_groups(
    records_by_role: dict[str, list[TextRecord]],
    fingerprints: dict[str, Fingerprints],
    reviews: dict[str, LinguisticReview],
    config: dict[str, Any],
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    views = (
        ("surface", False),
        ("number_masked", True),
        ("entity_masked", True),
        ("carrier_stripped", True),
    )
    for role, records in sorted(records_by_role.items()):
        for view, minimal_pair_allowed in views:
            groups = _group_by(records, lambda record, view=view: getattr(fingerprints[record.candidate_id], view))
            for group in groups.values():
                if len(group) <= 1:
                    continue
                if minimal_pair_allowed and _all_minimal_pair_approved(group, reviews, config):
                    continue
                issues.append(
                    DataQualityIssue(
                        code=f"within_partition_{view}_collision",
                        severity="failed",
                        check="fingerprints",
                        partition_role=role,
                        candidate_id=group[0].candidate_id,
                        related_candidate_id=group[1].candidate_id,
                        detail=f"{len(group)} rows share {view} fingerprint",
                    )
                )
    return issues


def validate_template_families(
    records_by_role: dict[str, list[TextRecord]],
    fingerprints: dict[str, Fingerprints],
    reviews: dict[str, LinguisticReview],
    config: dict[str, Any],
) -> tuple[list[DataQualityIssue], dict[str, Any]]:
    issues: list[DataQualityIssue] = []
    family_counts: dict[str, dict[str, int]] = {}
    max_fraction = float(config["template_families"]["max_undeclared_family_fraction"])
    min_count = int(config["template_families"]["min_undeclared_family_count"])
    for role, records in sorted(records_by_role.items()):
        discovered = Counter(fingerprints[record.candidate_id].discovered_template_family for record in records)
        declared = Counter(record.template_family_id for record in records if record.template_family_id)
        row_count = max(1, len(records))
        family_counts[role] = {
            "declared_family_count": len(declared),
            "discovered_family_count": len(discovered),
            "largest_discovered_family_size": max(discovered.values(), default=0),
            "largest_discovered_family_fraction_micros": int((max(discovered.values(), default=0) / row_count) * 1_000_000),
        }
        for family, count in discovered.items():
            if count <= 1:
                continue
            group = [record for record in records if fingerprints[record.candidate_id].discovered_template_family == family]
            if _all_minimal_pair_approved(group, reviews, config):
                continue
            if count >= min_count and count / row_count > max_fraction:
                issues.append(
                    DataQualityIssue(
                        code="undeclared_template_family_concentration",
                        severity="failed",
                        check="template_family_discovery",
                        partition_role=role,
                        detail=f"discovered family has {count}/{row_count} rows",
                    )
                )
        for family, count in declared.items():
            group = [record for record in records if record.template_family_id == family]
            if _all_minimal_pair_approved(group, reviews, config):
                continue
            if count / row_count > float(config["template_families"]["max_declared_family_fraction"]):
                issues.append(
                    DataQualityIssue(
                        code="declared_template_family_concentration",
                        severity="review_required",
                        check="template_family_discovery",
                        partition_role=role,
                        detail=f"declared family has {count}/{row_count} rows",
                    )
                )
    return issues, family_counts


def candidate_pairs_from_shingles(items: dict[str, set[str]]) -> set[tuple[str, str]]:
    inverted: dict[str, list[str]] = defaultdict(list)
    for item_id, shingles in items.items():
        for shingle in shingles:
            inverted[shingle].append(item_id)
    pairs: set[tuple[str, str]] = set()
    for ids in inverted.values():
        if len(ids) < 2:
            continue
        ordered = sorted(set(ids))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pairs.add((left, right))
    return pairs


def fuzzy_similarity_issues(
    records_by_role: dict[str, list[TextRecord]],
    fingerprints: dict[str, Fingerprints],
    config: dict[str, Any],
    reviews: dict[str, LinguisticReview],
    *,
    cross_partition_only: bool = False,
) -> tuple[list[DataQualityIssue], list[dict[str, Any]], dict[str, int]]:
    # Reconstruct privacy-safe comparison views from hashes where possible would be
    # too lossy, so use normalized forms in memory and emit only IDs/scores.
    forms: dict[str, dict[str, str]] = {}
    all_records = group_all(records_by_role)
    by_id = {record.candidate_id: record for record in all_records}
    for record in all_records:
        forms[record.candidate_id] = {
            "surface": surface_form(record.target_text),
            "number_masked": number_masked_form(record.target_text),
            "entity_masked": entity_masked_form(record.target_text, record.entities),
        }
        forms[record.candidate_id]["carrier_stripped"] = forms[record.candidate_id]["entity_masked"]

    token_threshold = float(config["similarity"]["token_jaccard_review_threshold"])
    char_threshold = float(config["similarity"]["character_jaccard_review_threshold"])
    pair_set: set[tuple[str, str]] = set()
    exact_comparisons = 0
    local_review_rows: list[dict[str, Any]] = []
    issues: list[DataQualityIssue] = []
    views = ("surface", "number_masked", "entity_masked", "carrier_stripped")
    for view in views:
        shingle_index: dict[str, set[str]] = {}
        for record in all_records:
            shingles: set[str] = set()
            for n in (2, 3, 4, 5):
                shingles.update(token_shingles(forms[record.candidate_id][view], n))
            shingle_index[record.candidate_id] = shingles
        candidate_pairs = candidate_pairs_from_shingles(shingle_index)
        pair_set.update(candidate_pairs)
        for left_id, right_id in sorted(candidate_pairs):
            left = by_id[left_id]
            right = by_id[right_id]
            if cross_partition_only and {left.partition_role, right.partition_role} != {"selected_training", "synthetic_holdout"}:
                continue
            # Candidate source pool and its selected-training subset are allowed
            # to be related; the strict disjointness target is training vs holdout.
            if {left.partition_role, right.partition_role} == {"synthetic_candidate", "selected_training"}:
                continue
            if (
                left.partition_role == right.partition_role
                and left.minimal_pair
                and right.minimal_pair
                and left.minimal_pair["family_id"] == right.minimal_pair["family_id"]
                and _all_minimal_pair_approved([left, right], reviews, config)
            ):
                continue
            exact_comparisons += 1
            score = jaccard(shingle_index[left_id], shingle_index[right_id])
            if score >= token_threshold:
                local_review_rows.append(
                    {
                        "left_candidate_id": left_id,
                        "right_candidate_id": right_id,
                        "view": view,
                        "similarity": round(score, 6),
                        "reason": "token_shingle_similarity_review_required",
                    }
                )
                issues.append(
                    DataQualityIssue(
                        code="fuzzy_similarity_review_required",
                        severity="review_required",
                        check="fuzzy_similarity",
                        partition_role="cross_partition" if left.partition_role != right.partition_role else left.partition_role,
                        candidate_id=left_id,
                        related_candidate_id=right_id,
                        detail=f"{view} token-shingle similarity {score:.6f}",
                    )
                )
    char_index = {
        record.candidate_id: character_shingles(forms[record.candidate_id]["surface"], 5)
        for record in all_records
    }
    for left_id, right_id in sorted(candidate_pairs_from_shingles(char_index)):
        left = by_id[left_id]
        right = by_id[right_id]
        if cross_partition_only and {left.partition_role, right.partition_role} != {"selected_training", "synthetic_holdout"}:
            continue
        if {left.partition_role, right.partition_role} == {"synthetic_candidate", "selected_training"}:
            continue
        if (
            left.partition_role == right.partition_role
            and left.minimal_pair
            and right.minimal_pair
            and left.minimal_pair["family_id"] == right.minimal_pair["family_id"]
            and _all_minimal_pair_approved([left, right], reviews, config)
        ):
            continue
        exact_comparisons += 1
        score = jaccard(char_index[left_id], char_index[right_id])
        if score >= char_threshold:
            local_review_rows.append(
                {
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "view": "character_5gram",
                    "similarity": round(score, 6),
                    "reason": "character_similarity_review_required",
                }
            )
            issues.append(
                DataQualityIssue(
                    code="character_similarity_review_required",
                    severity="review_required",
                    check="fuzzy_similarity",
                    partition_role="cross_partition" if left.partition_role != right.partition_role else left.partition_role,
                    candidate_id=left_id,
                    related_candidate_id=right_id,
                    detail=f"character 5-gram similarity {score:.6f}",
                )
            )
    possible_pairs = len(all_records) * (len(all_records) - 1) // 2
    stats = {
        "possible_all_pairs": possible_pairs,
        "candidate_pair_count": len(pair_set),
        "exact_comparisons_performed": exact_comparisons,
    }
    return issues, local_review_rows, stats


def validate_cross_partition(
    records_by_role: dict[str, list[TextRecord]],
    fingerprints: dict[str, Fingerprints],
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    training = records_by_role.get("selected_training", [])
    holdout = records_by_role.get("synthetic_holdout", [])
    candidates = records_by_role.get("synthetic_candidate", [])
    candidate_source_ids = {record.source_id for record in candidates}

    if candidates and training:
        for record in training:
            if record.source_id not in candidate_source_ids:
                issues.append(
                    DataQualityIssue(
                        code="selected_training_absent_from_candidate_pool",
                        severity="failed",
                        check="cross_partition",
                        partition_role=record.partition_role,
                        candidate_id=record.candidate_id,
                    )
                )

    comparison_pairs = [
        ("selected_training", training, "synthetic_holdout", holdout),
        ("synthetic_candidate", candidates, "synthetic_holdout", holdout),
    ]
    comparisons = (
        ("candidate_id", lambda record: record.candidate_id),
        ("source_id", lambda record: record.source_id),
        ("source_family_id", lambda record: record.source_family_id),
        ("utterance_family_id", lambda record: record.utterance_family_id),
        ("declared_template_family", lambda record: record.template_family_id or ""),
        ("discovered_template_family", lambda record: fingerprints[record.candidate_id].discovered_template_family),
        ("surface", lambda record: fingerprints[record.candidate_id].surface),
        ("number_masked", lambda record: fingerprints[record.candidate_id].number_masked),
        ("entity_masked", lambda record: fingerprints[record.candidate_id].entity_masked),
        ("carrier_stripped", lambda record: fingerprints[record.candidate_id].carrier_stripped),
        ("source_recording_id", lambda record: str(record.optional_metadata.get("source_recording_id", ""))),
    )
    for left_role, left_records, right_role, right_records in comparison_pairs:
        if not left_records or not right_records:
            continue
        for name, key_fn in comparisons:
            right_index: dict[str, TextRecord] = {}
            for record in right_records:
                key = str(key_fn(record))
                if key:
                    right_index[key] = record
            for record in left_records:
                key = str(key_fn(record))
                if key and key in right_index:
                    issues.append(
                        DataQualityIssue(
                            code=f"cross_partition_{name}_overlap",
                            severity="failed",
                            check="cross_partition",
                            partition_role=f"{left_role}/{right_role}",
                            candidate_id=record.candidate_id,
                            related_candidate_id=right_index[key].candidate_id,
                        )
                    )
    return issues


def load_protected_index(path: Path) -> ProtectedTextIndex:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: protected index must be an object")
    if payload.get("schema_version") != PROTECTED_INDEX_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported protected index schema")
    if payload.get("normalizer") != NORMALIZER_VERSION:
        raise ValueError(f"{path}: unexpected normalizer")
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError(f"{path}: unexpected algorithm version")
    surface = payload.get("surface_hashes", [])
    number = payload.get("number_masked_hashes", [])
    if not isinstance(surface, list) or not isinstance(number, list):
        raise ValueError(f"{path}: protected hash sets must be lists")
    return ProtectedTextIndex(
        gate_id=str(payload["gate_id"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        metadata_manifest_sha256=str(payload.get("metadata_manifest_sha256", "")),
        reference_manifest_sha256=payload.get("reference_manifest_sha256"),
        metadata_reference_manifest_sha256=payload.get("metadata_reference_manifest_sha256"),
        row_count=int(payload["row_count"]),
        normalizer=str(payload["normalizer"]),
        algorithm_version=str(payload["algorithm_version"]),
        surface_hashes={str(item) for item in surface},
        number_masked_hashes={str(item) for item in number},
    )


def validate_protected_indexes(
    records_by_role: dict[str, list[TextRecord]],
    indexes: list[ProtectedTextIndex],
    config: dict[str, Any],
) -> tuple[list[DataQualityIssue], list[dict[str, Any]], dict[str, int]]:
    issues: list[DataQualityIssue] = []
    by_gate = {index.gate_id: index for index in indexes}
    for required_gate in config["protected_indexes"]["required_gate_ids"]:
        if required_gate not in by_gate:
            issues.append(
                DataQualityIssue(
                    code="missing_required_protected_index",
                    severity="review_required",
                    check="protected_indexes",
                    detail=required_gate,
                )
            )
    surface_overlaps = 0
    number_overlaps = 0
    identities: list[dict[str, Any]] = []
    for index in indexes:
        identities.append(index.public_identity())
        if index.manifest_sha256 != index.metadata_manifest_sha256:
            issues.append(
                DataQualityIssue(
                    code="stale_protected_index_manifest",
                    severity="failed",
                    check="protected_indexes",
                    detail=index.gate_id,
                )
            )
        if index.reference_manifest_sha256 and index.metadata_reference_manifest_sha256:
            if index.reference_manifest_sha256 != index.metadata_reference_manifest_sha256:
                issues.append(
                    DataQualityIssue(
                        code="stale_protected_index_reference_manifest",
                        severity="failed",
                        check="protected_indexes",
                        detail=index.gate_id,
                    )
                )
    for record in group_all(records_by_role):
        surface_hash = fingerprint_hash(surface_form(record.target_text))
        number_hash = fingerprint_hash(number_masked_form(record.target_text))
        for index in indexes:
            if surface_hash in index.surface_hashes:
                surface_overlaps += 1
                issues.append(
                    DataQualityIssue(
                        code="protected_surface_overlap",
                        severity="failed",
                        check="protected_indexes",
                        partition_role=record.partition_role,
                        candidate_id=record.candidate_id,
                        detail=index.gate_id,
                    )
                )
            if number_hash in index.number_masked_hashes:
                number_overlaps += 1
                issues.append(
                    DataQualityIssue(
                        code="protected_number_masked_overlap",
                        severity="review_required",
                        check="protected_indexes",
                        partition_role=record.partition_role,
                        candidate_id=record.candidate_id,
                        detail=index.gate_id,
                    )
                )
    return issues, identities, {"surface_overlaps": surface_overlaps, "number_masked_overlaps": number_overlaps}


def build_protected_index_payload(manifest_path: Path, metadata_path: Path) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be an object")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != metadata.get("manifest_sha256"):
        raise ValueError(
            f"{manifest_path}: manifest SHA256 {manifest_sha} does not match metadata {metadata.get('manifest_sha256')}"
        )
    rows = load_jsonl(manifest_path)
    surface_hashes: set[str] = set()
    number_hashes: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        text = row.get("text", row.get("reference"))
        if not isinstance(text, str) or not text:
            raise ValueError(f"{manifest_path}:{line_number}: missing reference text")
        surface_hashes.add(fingerprint_hash(surface_form(text)))
        number_hashes.add(fingerprint_hash(number_masked_form(text)))
    expected_rows = metadata.get("rows", metadata.get("segments", len(rows)))
    if int(expected_rows) != len(rows):
        raise ValueError(f"{manifest_path}: row count {len(rows)} does not match metadata {expected_rows}")
    return {
        "schema_version": PROTECTED_INDEX_SCHEMA_VERSION,
        "gate_id": metadata["gate_id"],
        "manifest_sha256": manifest_sha,
        "metadata_manifest_sha256": metadata["manifest_sha256"],
        "reference_manifest_sha256": metadata.get("reference_manifest_sha256"),
        "metadata_reference_manifest_sha256": metadata.get("reference_manifest_sha256"),
        "row_count": len(rows),
        "normalizer": NORMALIZER_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint_versions": {
            "surface": "surface-normalized-v1",
            "number_masked": "number-masked-v1",
        },
        "surface_hashes": sorted(surface_hashes),
        "number_masked_hashes": sorted(number_hashes),
    }


def validate_linguistic_reviews(
    records_by_role: dict[str, list[TextRecord]],
    reviews: dict[str, LinguisticReview],
) -> tuple[list[DataQualityIssue], dict[str, Any]]:
    issues: list[DataQualityIssue] = []
    rows = group_all(records_by_role)
    known_ids = {record.candidate_id for record in rows}
    review_ids = set(reviews)
    for unknown in sorted(review_ids - known_ids):
        issues.append(
            DataQualityIssue(
                code="review_for_unknown_candidate",
                severity="failed",
                check="linguistic_review",
                candidate_id=unknown,
            )
        )
    outcome_counts: Counter[str] = Counter()
    for record in rows:
        review = reviews.get(record.candidate_id)
        if review is None:
            issues.append(
                DataQualityIssue(
                    code="missing_linguistic_review",
                    severity="review_required",
                    check="linguistic_review",
                    partition_role=record.partition_role,
                    candidate_id=record.candidate_id,
                )
            )
            outcome_counts["MISSING"] += 1
            continue
        outcome_counts[review.outcome] += 1
        if review.outcome == "ACCEPT":
            continue
        severity = "review_required" if review.outcome == "REVISE_AND_REREVIEW" else "failed"
        issues.append(
            DataQualityIssue(
                code="linguistic_review_not_accepted",
                severity=severity,
                check="linguistic_review",
                partition_role=record.partition_role,
                candidate_id=record.candidate_id,
                detail=review.outcome,
            )
        )
    return issues, {
        "required": len(rows),
        "provided": len(review_ids & known_ids),
        "unknown_candidate_reviews": len(review_ids - known_ids),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }


def validate_metadata_leakage(records_by_role: dict[str, list[TextRecord]]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for record in group_all(records_by_role):
        if metadata_leakage(record.target_text):
            issues.append(
                DataQualityIssue(
                    code="metadata_token_in_text",
                    severity="failed",
                    check="metadata_leakage",
                    partition_role=record.partition_role,
                    candidate_id=record.candidate_id,
                )
            )
    return issues


def determine_status(issues: list[DataQualityIssue], config: dict[str, Any]) -> str:
    if any(issue.severity == "failed" for issue in issues):
        return STATUS_TEXT_REJECTED
    if any(issue.severity in {"review_required", "blocked", "not_run"} for issue in issues):
        return STATUS_DRAFT
    if config.get("intended_use") == "diagnostic_only":
        return STATUS_DIAGNOSTIC_ONLY
    return STATUS_TEXT_ACCEPTED


def retired_registry_hashes(registry: dict[str, Any]) -> dict[str, str]:
    entries = registry.get("retired_corpora")
    if not isinstance(entries, list):
        raise ValueError("retired registry must contain retired_corpora list")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("retired registry entries must be objects")
        result[str(entry["artifact"])] = str(entry["sha256"])
    return result


def load_partition_records(
    partitions: dict[str, Path],
    config: dict[str, Any],
) -> tuple[dict[str, list[TextRecord]], list[DataQualityIssue]]:
    records_by_role: dict[str, list[TextRecord]] = {}
    issues: list[DataQualityIssue] = []
    for role in sorted(partitions):
        rows = load_jsonl(partitions[role])
        records: list[TextRecord] = []
        for line_index, row in enumerate(rows, start=1):
            try:
                records.append(validate_text_record(row, expected_role=role, config=config))
            except Exception as exc:
                candidate_id = str(row.get("candidate_id", "")) if isinstance(row, dict) else None
                issues.append(
                    DataQualityIssue(
                        code="record_schema_failure",
                        severity="failed",
                        check="record_schema",
                        partition_role=role,
                        candidate_id=candidate_id or None,
                        detail=f"line {line_index}: {exc}",
                    )
                )
        records_by_role[role] = records
    return records_by_role, issues


def validate_candidate_ids(records_by_role: dict[str, list[TextRecord]]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    seen: dict[str, TextRecord] = {}
    for record in group_all(records_by_role):
        if record.candidate_id in seen:
            issues.append(
                DataQualityIssue(
                    code="duplicate_candidate_id",
                    severity="failed",
                    check="record_schema",
                    partition_role=record.partition_role,
                    candidate_id=record.candidate_id,
                    related_candidate_id=seen[record.candidate_id].candidate_id,
                )
            )
        seen[record.candidate_id] = record
    return issues


def partition_counts(records_by_role: dict[str, list[TextRecord]]) -> dict[str, int]:
    return {role: len(records_by_role.get(role, [])) for role in sorted(records_by_role)}


def unique_counts(records_by_role: dict[str, list[TextRecord]], fingerprints: dict[str, Fingerprints]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for role, records in sorted(records_by_role.items()):
        result[role] = {
            "surface": len({fingerprints[record.candidate_id].surface for record in records}),
            "number_masked": len({fingerprints[record.candidate_id].number_masked for record in records}),
            "entity_masked": len({fingerprints[record.candidate_id].entity_masked for record in records}),
            "carrier_stripped": len({fingerprints[record.candidate_id].carrier_stripped for record in records}),
        }
    return result


def build_check_statuses(issues: list[DataQualityIssue], check_names: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for check in check_names:
        related = [issue for issue in issues if issue.check == check]
        if any(issue.severity == "failed" for issue in related):
            statuses[check] = "FAILED"
        elif any(issue.severity == "blocked" for issue in related):
            statuses[check] = "BLOCKED"
        elif any(issue.severity == "not_run" for issue in related):
            statuses[check] = "NOT_RUN"
        elif any(issue.severity == "review_required" for issue in related):
            statuses[check] = "REVIEW_REQUIRED"
        else:
            statuses[check] = "PASSED"
    return statuses


def validate_corpus(
    *,
    corpus_id: str,
    config: dict[str, Any],
    config_sha256: str,
    retired_registry: dict[str, Any],
    partitions: dict[str, Path],
    linguistic_review_path: Path | None,
    protected_index_paths: list[Path],
    repository_revision: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_hashes = {role: sha256_file(path) for role, path in sorted(partitions.items())}
    retired_hashes = retired_registry_hashes(retired_registry)
    retired_matches = [
        {"role": role, "sha256": digest, "artifact": artifact}
        for role, digest in input_hashes.items()
        for artifact, retired_digest in retired_hashes.items()
        if digest == retired_digest
    ]
    base_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "validator_algorithm_version": ALGORITHM_VERSION,
        "normalizer": NORMALIZER_VERSION,
        "configuration_sha256": config_sha256,
        "input_partition_byte_hashes": input_hashes,
        "retired_hash_checks": {
            "registry_hashes": retired_hashes,
            "matches": retired_matches,
        },
        "repository_revision": repository_revision,
        "privacy": {
            "raw_text_in_report": False,
            "local_paths_in_report": False,
            "reviewer_identity_in_report": False,
        },
        "limitations": [
            "Text-stage validation does not prove Slovenian grammaticality beyond supplied review coverage.",
            "Text-stage validation does not prove acoustic suitability.",
            "No TRAINING_ELIGIBLE status can be emitted by this validator.",
        ],
    }
    if retired_matches:
        report = {
            **base_report,
            "final_text_status": STATUS_RETIRED,
            "decision_reasons": ["retired_input_hash"],
            "checks": {"retired_hash_checks": "FAILED"},
            "issues": [
                {
                    "code": "retired_input_hash",
                    "severity": "failed",
                    "check": "retired_hash_checks",
                    "partition_role": match["role"],
                    "detail": match["artifact"],
                }
                for match in retired_matches
            ],
        }
        return report, []

    issues: list[DataQualityIssue] = []
    try:
        records_by_role, schema_issues = load_partition_records(partitions, config)
        issues.extend(schema_issues)
    except Exception as exc:
        issue = DataQualityIssue(code="input_parse_failure", severity="failed", check="record_schema", detail=str(exc))
        report = {
            **base_report,
            "final_text_status": STATUS_TEXT_REJECTED,
            "decision_reasons": [issue.code],
            "checks": {"record_schema": "FAILED"},
            "issues": [issue.public()],
        }
        return report, []

    issues.extend(validate_candidate_ids(records_by_role))
    issues.extend(validate_metadata_leakage(records_by_role))

    reviews: dict[str, LinguisticReview] = {}
    if linguistic_review_path is None:
        for record in group_all(records_by_role):
            issues.append(
                DataQualityIssue(
                    code="missing_linguistic_review_file",
                    severity="review_required",
                    check="linguistic_review",
                    partition_role=record.partition_role,
                    candidate_id=record.candidate_id,
                )
            )
        review_summary = {
            "required": len(group_all(records_by_role)),
            "provided": 0,
            "unknown_candidate_reviews": 0,
            "outcome_counts": {"MISSING": len(group_all(records_by_role))},
        }
    else:
        try:
            reviews = load_linguistic_reviews(linguistic_review_path)
            review_issues, review_summary = validate_linguistic_reviews(records_by_role, reviews)
            issues.extend(review_issues)
        except Exception as exc:
            issues.append(
                DataQualityIssue(
                    code="linguistic_review_parse_failure",
                    severity="failed",
                    check="linguistic_review",
                    detail=exc.__class__.__name__,
                )
            )
            review_summary = {"required": len(group_all(records_by_role)), "provided": 0, "unknown_candidate_reviews": 0, "outcome_counts": {}}

    fingerprints, carrier_issues, carrier_statistics = compute_fingerprints(records_by_role, config)
    issues.extend(carrier_issues)
    issues.extend(validate_exact_groups(records_by_role, fingerprints, reviews, config))
    template_issues, family_counts = validate_template_families(records_by_role, fingerprints, reviews, config)
    issues.extend(template_issues)
    issues.extend(validate_cross_partition(records_by_role, fingerprints))

    fuzzy_issues, local_review_rows, comparison_stats = fuzzy_similarity_issues(records_by_role, fingerprints, config, reviews)
    issues.extend(fuzzy_issues)

    protected_indexes: list[ProtectedTextIndex] = []
    for path in protected_index_paths:
        try:
            protected_indexes.append(load_protected_index(path))
        except Exception as exc:
            issues.append(DataQualityIssue(code="protected_index_parse_failure", severity="failed", check="protected_indexes", detail=str(exc)))
    protected_issues, protected_identities, protected_counts = validate_protected_indexes(records_by_role, protected_indexes, config)
    issues.extend(protected_issues)

    status = determine_status(issues, config)
    check_names = [
        "retired_hash_checks",
        "record_schema",
        "metadata_leakage",
        "fingerprints",
        "carrier_detection",
        "template_family_discovery",
        "cross_partition",
        "fuzzy_similarity",
        "linguistic_review",
        "protected_indexes",
    ]
    report = {
        **base_report,
        "final_text_status": status,
        "decision_reasons": sorted({issue.code for issue in issues}) if issues else ["all_required_text_checks_passed"],
        "statuses_emittable_by_text_validator": sorted(EMITTABLE_STATUSES),
        "statuses_intentionally_impossible": sorted(IMPOSSIBLE_TEXT_VALIDATOR_STATUSES),
        "row_counts": partition_counts(records_by_role),
        "schema_valid_counts": {role: len(records) for role, records in sorted(records_by_role.items())},
        "fingerprint_unique_counts": unique_counts(records_by_role, fingerprints),
        "metadata_leakage_count": sum(1 for issue in issues if issue.code == "metadata_token_in_text"),
        "template_family_counts": family_counts,
        "prefix_suffix_frame_concentration": carrier_statistics,
        "within_partition_overlap_counts": {
            "surface": sum(1 for issue in issues if issue.code == "within_partition_surface_collision"),
            "number_masked": sum(1 for issue in issues if issue.code == "within_partition_number_masked_collision"),
            "entity_masked": sum(1 for issue in issues if issue.code == "within_partition_entity_masked_collision"),
            "carrier_stripped": sum(1 for issue in issues if issue.code == "within_partition_carrier_stripped_collision"),
        },
        "cross_partition_overlap_counts": {
            issue.code: sum(1 for candidate in issues if candidate.code == issue.code)
            for issue in issues
            if issue.check == "cross_partition"
        },
        "fuzzy_review_pair_counts": {
            "pairs_requiring_review": len(local_review_rows),
        },
        "linguistic_review": review_summary,
        "protected_indexes": protected_identities,
        "protected_overlap_counts": protected_counts,
        "pair_candidate_comparison_counts": comparison_stats,
        "checks": build_check_statuses(issues, check_names),
        "issues": [issue.public() for issue in issues],
    }
    return report, local_review_rows


def assert_privacy_safe_report(report: dict[str, Any]) -> None:
    serialized = json.dumps(report, ensure_ascii=False)
    forbidden_keys = {
        "spoken_text",
        "target_text",
        "text",
        "reference",
        "raw_reference",
        "hypothesis",
        "reviewer_approval",
        "audio_filepath",
    }

    def walk(value: Any, key: str = "") -> None:
        if key in forbidden_keys:
            raise ValueError(f"report contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(report)
    if re.search(r"(^|[\"'\s])/(?:home|mnt/data|tmp)/", serialized):
        raise ValueError("report contains a local absolute path")
