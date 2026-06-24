from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.corpus_v2_generation import config_sha256, load_config as load_reservoir_config, run_dir
from slaif_asr.corpus_v2_holdout import load_config as load_holdout_config
from slaif_asr.data_quality import (
    ALGORITHM_VERSION,
    assert_privacy_safe_report,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_protected_index_payload,
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    validate_corpus,
)


REVIEW_ADMISSION_VERSION = "corpus-v2-review-admission-v1"
REVIEW_REPORT_SCHEMA_VERSION = "1.0"
REVIEWED_CORPUS_ID = "sl-corpus-v2-gams-candidate-reservoir-v1-reviewed"
EXPECTED_PRE_REVIEW_SHA256 = "5cb2520c27b3debd18a2f475368c2cdd8601fc5781ec541287092dbcd3ea0fe6"
EXPECTED_REVIEW_TEMPLATE_SHA256 = "a22d87aa5e6913d2ceeaa1c61414ab946ef4c5c1bb1f8cc5b8063f984e454d84"
WHOLE_FILE_DECISION_VERSION = "whole-file-review-decision-v1"

ALLOWED_REVIEW_OUTCOMES = {
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

REJECT_OUTCOMES = {outcome for outcome in ALLOWED_REVIEW_OUTCOMES if outcome.startswith("REJECT_")}
REASON_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_:-]{1,63}$")
LOCAL_HOME_PATTERN = "/" + "home" + "/"
MNT_DATA_PATTERN = "/" + "mnt" + "/" + "data"
SAFE_PUBLIC_PATTERN = re.compile(
    r"\b(?:spoken_text|target_text|gamsv2-cell\d{2}-a\d{2}-o\d{3}|reviewer_identity)\b"
    + r"|\bgams9holdout-hcell\d{2}-a\d{2}-o\d{3}\b"
    + "|"
    + re.escape(LOCAL_HOME_PATTERN)
    + "|"
    + re.escape(MNT_DATA_PATTERN)
)
REVIEW_DECISION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")


@dataclass(frozen=True)
class ReviewDecision:
    candidate_id: str
    outcome: str
    review_revision: str
    reason_codes: tuple[str, ...]
    minimal_pair_approved: bool
    line_number: int

    def decision_record(self, *, source_row: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "review_revision": self.review_revision,
            "reason_codes": list(self.reason_codes),
            "minimal_pair_approved": self.minimal_pair_approved,
            "source_has_minimal_pair": bool(source_row and source_row.get("minimal_pair")),
        }

    def validator_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "review_revision": self.review_revision,
            "reason_codes": list(self.reason_codes),
            "minimal_pair_approved": self.minimal_pair_approved,
        }


@dataclass(frozen=True)
class ReviewIssue:
    code: str
    line_number: int | None = None
    detail: str | None = None

    def public(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code}
        if self.line_number is not None:
            payload["line_number"] = self.line_number
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root() / path


def load_review_config(path: Path) -> dict[str, Any]:
    try:
        return load_reservoir_config(path)
    except Exception:
        return load_holdout_config(path)


def default_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = repo_root()
    base = run_dir(config)
    if config.get("corpus_id") == "sl-corpus-v2-independent-synthetic-holdout-v1":
        return {
            "pre_review": base / "fixed-holdout.local.jsonl",
            "review_sheet": base / "review-capsule.local.tsv",
            "review_template": base / "review-capsule.local.tsv",
            "decisions": base / "holdout-review-decisions.local.jsonl",
            "accepted_candidates": base / "accepted-holdout.local.jsonl",
            "accepted_review": base / "accepted-holdout-linguistic-review.local.jsonl",
            "post_review_validation": base / "post-review-validation.local.json",
            "post_review_local_review": base / "post-review-review.local.jsonl",
            "rejected_by_human": base / "rejected-holdout-by-human.local.jsonl",
            "public_json": root / "docs/data-reports/0005-corpus-v2-holdout-review-admission.json",
            "public_markdown": root / "docs/data-reports/0005-corpus-v2-holdout-review-admission.md",
            "fleurs_manifest": root / "runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl",
            "fleurs_metadata": root / "docs/evaluation-gates/fleurs-sl-si-test-full-v2.metadata.json",
            "fleurs_index": root / "runs/data-quality/protected/fleurs-v2.hash-index.json",
            "artur_manifest": root / "runs/evaluation-gates/artur-j-public-gate-v1/manifest.jsonl",
            "artur_metadata": root / "docs/evaluation-gates/artur-j-public-gate-v1.metadata.json",
            "artur_index": root / "runs/data-quality/protected/artur-j.hash-index.json",
        }
    return {
        "pre_review": base / "pre-review-candidates.local.jsonl",
        "review_sheet": base / "linguistic-review-sheet.local.tsv",
        "review_template": base / "linguistic-review-template.local.jsonl",
        "decisions": base / "linguistic-review-decisions.local.jsonl",
        "accepted_candidates": base / "accepted-candidates.local.jsonl",
        "accepted_review": base / "accepted-linguistic-review.local.jsonl",
        "post_review_validation": base / "post-review-validation.local.json",
        "post_review_local_review": base / "post-review-review.local.jsonl",
        "rejected_by_human": base / "rejected-by-human.local.jsonl",
        "public_json": root / "docs/data-reports/0002-corpus-v2-linguistic-review-admission.json",
        "public_markdown": root / "docs/data-reports/0002-corpus-v2-linguistic-review-admission.md",
        "fleurs_manifest": root / "runs/evaluation-gates/fleurs-sl-si-test-full-v2/manifest.jsonl",
        "fleurs_metadata": root / "docs/evaluation-gates/fleurs-sl-si-test-full-v2.metadata.json",
        "fleurs_index": root / "runs/data-quality/protected/fleurs-v2.hash-index.json",
        "artur_manifest": root / "runs/evaluation-gates/artur-j-public-gate-v1/manifest.jsonl",
        "artur_metadata": root / "docs/evaluation-gates/artur-j-public-gate-v1.metadata.json",
        "artur_index": root / "runs/data-quality/protected/artur-j.hash-index.json",
    }


def parse_bool(value: str, *, line_number: int) -> tuple[bool | None, ReviewIssue | None]:
    normalized = value.strip().casefold()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True, None
    if normalized in {"false", "f", "0", "no", "n"}:
        return False, None
    return None, ReviewIssue("malformed_minimal_pair_approved", line_number, value.strip() or "<blank>")


def parse_reason_codes(value: str, *, line_number: int) -> tuple[tuple[str, ...], ReviewIssue | None]:
    raw = value.strip()
    if not raw:
        return (), None
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return (), ReviewIssue("malformed_reason_codes", line_number, "invalid JSON array")
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            return (), ReviewIssue("malformed_reason_codes", line_number, "reason_codes JSON must be a string array")
        items = [item.strip() for item in parsed if item.strip()]
    else:
        items = [item.strip() for item in re.split(r"[;,]", raw) if item.strip()]
    if any(not REASON_CODE_PATTERN.fullmatch(item) for item in items):
        return (), ReviewIssue("malformed_reason_codes", line_number, "unsupported reason-code token")
    return tuple(items), None


def read_review_tsv(path: Path) -> tuple[list[ReviewDecision], list[ReviewIssue], str]:
    expected = {
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
    }
    issues: list[ReviewIssue] = []
    decisions: list[ReviewDecision] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(expected - fieldnames)
        if missing:
            return [], [ReviewIssue("review_tsv_missing_columns", detail=",".join(missing))], sha256_file(path)
        for line_number, row in enumerate(reader, start=2):
            candidate_id = (row.get("candidate_id") or "").strip()
            outcome = (row.get("outcome") or "").strip()
            review_revision = (row.get("review_revision") or "").strip()
            if not candidate_id:
                issues.append(ReviewIssue("blank_candidate_id", line_number))
            if not outcome:
                issues.append(ReviewIssue("blank_outcome", line_number))
            elif outcome not in ALLOWED_REVIEW_OUTCOMES:
                issues.append(ReviewIssue("unsupported_outcome", line_number, outcome))
            if not review_revision:
                issues.append(ReviewIssue("blank_review_revision", line_number))
            reason_codes, reason_issue = parse_reason_codes(row.get("reason_codes") or "", line_number=line_number)
            if reason_issue is not None:
                issues.append(reason_issue)
            minimal_pair_approved, bool_issue = parse_bool(row.get("minimal_pair_approved") or "", line_number=line_number)
            if bool_issue is not None:
                issues.append(bool_issue)
            decisions.append(
                ReviewDecision(
                    candidate_id=candidate_id,
                    outcome=outcome,
                    review_revision=review_revision,
                    reason_codes=reason_codes,
                    minimal_pair_approved=bool(minimal_pair_approved),
                    line_number=line_number,
                )
            )
    return decisions, issues, sha256_file(path)


def validate_review_coverage(
    *,
    source_rows: Sequence[dict[str, Any]],
    decisions: Sequence[ReviewDecision],
    tsv_rows_by_id: dict[str, dict[str, str]] | None = None,
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    source_by_id = {str(row["candidate_id"]): row for row in source_rows}
    seen: dict[str, ReviewDecision] = {}
    for decision in decisions:
        if decision.candidate_id in seen:
            issues.append(ReviewIssue("duplicate_review_row", decision.line_number, decision.candidate_id))
        seen[decision.candidate_id] = decision
        source = source_by_id.get(decision.candidate_id)
        if source is None:
            issues.append(ReviewIssue("unknown_candidate_id", decision.line_number, decision.candidate_id))
            continue
        if decision.minimal_pair_approved and not source.get("minimal_pair"):
            issues.append(ReviewIssue("minimal_pair_approval_without_source_family", decision.line_number, decision.candidate_id))
        if tsv_rows_by_id is not None:
            tsv_row = tsv_rows_by_id.get(decision.candidate_id, {})
            if tsv_row.get("spoken_text") != source.get("spoken_text") or tsv_row.get("target_text") != source.get("target_text"):
                issues.append(ReviewIssue("review_text_mismatch", decision.line_number, decision.candidate_id))
    missing = sorted(set(source_by_id) - set(seen))
    for candidate_id in missing:
        issues.append(ReviewIssue("missing_review_row", detail=candidate_id))
    return issues


def read_tsv_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            candidate_id = (row.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in rows:
                rows[candidate_id] = dict(row)
    return rows


def build_whole_file_decisions(
    *,
    source_rows: Sequence[dict[str, Any]],
    outcome: str,
    review_revision: str,
    decision_id: str,
    expected_corpus_sha256: str,
    expected_rows: int,
    actual_corpus_sha256: str,
) -> tuple[list[ReviewDecision], list[ReviewIssue]]:
    issues: list[ReviewIssue] = []
    outcome = outcome.strip()
    review_revision = review_revision.strip()
    decision_id = decision_id.strip()
    if not outcome:
        issues.append(ReviewIssue("blank_outcome"))
    elif outcome not in ALLOWED_REVIEW_OUTCOMES:
        issues.append(ReviewIssue("unsupported_outcome", detail=outcome))
    if not review_revision:
        issues.append(ReviewIssue("blank_review_revision"))
    if not decision_id:
        issues.append(ReviewIssue("blank_decision_id"))
    elif not REVIEW_DECISION_ID_PATTERN.fullmatch(decision_id):
        issues.append(ReviewIssue("unsafe_decision_id", detail=decision_id))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_corpus_sha256):
        issues.append(ReviewIssue("malformed_expected_corpus_sha256"))
    elif expected_corpus_sha256 != actual_corpus_sha256:
        issues.append(ReviewIssue("unexpected_pre_review_reservoir_sha256", detail=actual_corpus_sha256))
    if expected_rows != len(source_rows):
        issues.append(ReviewIssue("unexpected_pre_review_row_count", detail=str(len(source_rows))))
    if issues:
        return [], issues
    decisions = [
        ReviewDecision(
            candidate_id=str(row["candidate_id"]),
            outcome=outcome,
            review_revision=review_revision,
            reason_codes=(),
            minimal_pair_approved=False,
            line_number=0,
        )
        for row in sorted(source_rows, key=lambda item: str(item["candidate_id"]))
    ]
    return decisions, []


def accepted_records_and_reviews(
    source_rows: Sequence[dict[str, Any]],
    decisions: Sequence[ReviewDecision],
    *,
    review_errors: Sequence[ReviewIssue],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    source_by_id = {str(row["candidate_id"]): row for row in source_rows}
    duplicate_or_unknown = {issue.code for issue in review_errors} & {
        "duplicate_review_row",
        "unknown_candidate_id",
        "blank_candidate_id",
        "unsupported_outcome",
        "blank_outcome",
        "malformed_minimal_pair_approved",
        "malformed_reason_codes",
        "minimal_pair_approval_without_source_family",
        "review_text_mismatch",
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_review: list[dict[str, Any]] = []
    complete_review_metadata = not review_errors
    for decision in sorted(decisions, key=lambda item: item.candidate_id):
        source = source_by_id.get(decision.candidate_id)
        if source is None:
            continue
        if decision.outcome == "ACCEPT" and not duplicate_or_unknown:
            accepted.append(source)
            if complete_review_metadata:
                accepted_review.append(decision.validator_record())
        elif decision.outcome in REJECT_OUTCOMES or decision.outcome == "REVISE_AND_REREVIEW":
            rejected.append(
                {
                    "candidate_id": decision.candidate_id,
                    "outcome": decision.outcome,
                    "review_revision": decision.review_revision,
                    "reason_codes": list(decision.reason_codes),
                }
            )
    counts = Counter(decision.outcome or "BLANK" for decision in decisions)
    return accepted, accepted_review, rejected, dict(sorted(counts.items()))


def ensure_protected_indexes(paths: dict[str, Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    pairs = [
        (paths["fleurs_manifest"], paths["fleurs_metadata"], paths["fleurs_index"]),
        (paths["artur_manifest"], paths["artur_metadata"], paths["artur_index"]),
    ]
    identities: list[dict[str, Any]] = []
    index_paths: list[Path] = []
    for manifest, metadata, output in pairs:
        payload = build_protected_index_payload(manifest, metadata)
        if not output.exists() or load_json(output) != payload:
            atomic_write_json(output, payload)
        identities.append(
            {
                "gate_id": payload["gate_id"],
                "manifest_sha256": payload["manifest_sha256"],
                "reference_manifest_sha256": payload.get("reference_manifest_sha256"),
                "row_count": payload["row_count"],
                "surface_hash_count": len(payload["surface_hashes"]),
                "number_masked_hash_count": len(payload["number_masked_hashes"]),
                "index_sha256": sha256_file(output),
            }
        )
        index_paths.append(output)
    return index_paths, identities


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def validate_accepted_subset(
    *,
    data_quality_config_path: Path,
    accepted_candidates_path: Path,
    accepted_review_path: Path | None,
    protected_index_paths: Sequence[Path],
    output_report_path: Path,
    local_review_output_path: Path,
    retired_registry_path: Path,
    partition_role: str = "synthetic_candidate",
    corpus_id: str = REVIEWED_CORPUS_ID,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(data_quality_config_path)
    retired_registry = load_json(retired_registry_path)
    report, local_review = validate_corpus(
        corpus_id=corpus_id,
        config=config,
        config_sha256=canonical_json_sha256(config),
        retired_registry=retired_registry,
        partitions={partition_role: accepted_candidates_path},
        linguistic_review_path=accepted_review_path,
        protected_index_paths=list(protected_index_paths),
        repository_revision=git_revision(),
    )
    assert_privacy_safe_report(report)
    atomic_write_json(output_report_path, report)
    atomic_write_jsonl(local_review_output_path, local_review)
    return report, local_review


def review_error_status(issues: Sequence[ReviewIssue]) -> str | None:
    if not issues:
        return None
    hard = {
        "duplicate_review_row",
        "unknown_candidate_id",
        "unsupported_outcome",
        "blank_outcome",
        "malformed_minimal_pair_approved",
        "malformed_reason_codes",
        "minimal_pair_approval_without_source_family",
        "review_text_mismatch",
    }
    if any(issue.code in hard for issue in issues):
        return "TEXT_REJECTED"
    return "DRAFT"


def build_public_report(
    *,
    generation_config: dict[str, Any],
    source_count: int,
    source_hash: str,
    review_template_hash: str | None,
    review_sheet_hash: str | None,
    decisions: Sequence[ReviewDecision],
    review_issues: Sequence[ReviewIssue],
    accepted_count: int,
    rejected_count: int,
    revise_count: int,
    accepted_partition_hash: str | None,
    protected_index_identities: Sequence[dict[str, Any]],
    validation_report: dict[str, Any] | None,
    review_mode: str = "row_tsv",
    whole_file_decision: dict[str, Any] | None = None,
    partition_role: str = "synthetic_candidate",
    reviewed_corpus_id: str = REVIEWED_CORPUS_ID,
) -> dict[str, Any]:
    outcome_counts = Counter(decision.outcome or "BLANK" for decision in decisions)
    validator_status = validation_report.get("final_text_status") if validation_report else review_error_status(review_issues)
    validator_reasons = validation_report.get("decision_reasons", []) if validation_report else sorted({issue.code for issue in review_issues})
    fingerprint_counts: dict[str, int] | None = None
    family_summary: dict[str, Any] | None = None
    protected_counts: dict[str, int] | None = None
    if validation_report:
        fingerprint_counts = validation_report.get("fingerprint_unique_counts", {}).get(partition_role)
        family_counts = validation_report.get("template_family_counts", {}).get(partition_role, {})
        if family_counts:
            family_summary = {
                "declared_family_count": family_counts.get("declared_family_count"),
                "discovered_family_count": family_counts.get("discovered_family_count"),
                "largest_family_size": family_counts.get("largest_discovered_family_size"),
                "largest_family_fraction": round(
                    int(family_counts.get("largest_discovered_family_size", 0)) / max(1, accepted_count),
                    6,
                ),
            }
        protected_counts = validation_report.get("protected_overlap_counts")
    payload = {
        "schema_version": REVIEW_REPORT_SCHEMA_VERSION,
        "admission_version": REVIEW_ADMISSION_VERSION,
        "corpus_id": reviewed_corpus_id,
        "partition_role": partition_role,
        "source_reservoir": {
            "corpus_id": generation_config["corpus_id"],
            "pre_review_sha256": source_hash,
            "expected_pre_review_sha256": EXPECTED_PRE_REVIEW_SHA256,
            "pre_review_count": source_count,
            "review_template_sha256": review_template_hash,
            "expected_review_template_sha256": EXPECTED_REVIEW_TEMPLATE_SHA256,
        },
        "review": {
            "mode": review_mode,
            "review_sheet_sha256": review_sheet_hash,
            "whole_file_decision": whole_file_decision,
            "total_review_rows": len(decisions),
            "review_coverage": {
                "required": source_count,
                "provided": len({decision.candidate_id for decision in decisions}),
                "complete": len({decision.candidate_id for decision in decisions}) == source_count and not review_issues,
            },
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "revise_and_rereview_count": revise_count,
            "review_issue_counts": dict(sorted(Counter(issue.code for issue in review_issues).items())),
        },
        "accepted_partition": {
            "sha256": accepted_partition_hash,
            "rows": accepted_count,
        },
        "fingerprint_unique_counts": fingerprint_counts,
        "family_summary": family_summary,
        "protected_indexes": list(protected_index_identities),
        "protected_overlap_counts": protected_counts,
        "validator": {
            "status": validator_status,
            "decision_reasons": validator_reasons,
            "algorithm_version": ALGORITHM_VERSION,
            "configuration_sha256": validation_report.get("configuration_sha256") if validation_report else None,
            "repository_revision": validation_report.get("repository_revision") if validation_report else git_revision(),
        },
        "limitations": [
            "Text admission does not prove acoustic suitability.",
            "No synthetic holdout exists in this work order.",
            "No data acceptance certificate was issued.",
            "TRAINING_ELIGIBLE cannot be produced by this text-stage admission step.",
        ],
    }
    assert_public_report_safe(payload)
    return payload


def assert_public_report_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if SAFE_PUBLIC_PATTERN.search(serialized):
        raise ValueError("public review-admission report contains raw text, candidate IDs, reviewer identity, or local paths")


def write_public_markdown(path: Path, payload: dict[str, Any]) -> None:
    validator = payload["validator"]
    review = payload["review"]
    source = payload["source_reservoir"]
    lines = [
        "# Corpus-v2 Linguistic Review Admission",
        "",
        f"Status: `{validator['status']}`",
        "",
        "This privacy-safe report contains only aggregate review and validation evidence. It does not include generated sentences, candidate IDs, reviewer identity, raw comments, protected references, or local paths.",
        "",
        "## Source",
        "",
        f"- Source corpus ID: `{source['corpus_id']}`",
        f"- Pre-review rows: {source['pre_review_count']}",
        f"- Pre-review SHA256: `{source['pre_review_sha256']}`",
        f"- Review sheet SHA256: `{review['review_sheet_sha256']}`",
        "",
        "## Review Funnel",
        "",
        f"- Review mode: `{review.get('mode', 'row_tsv')}`",
        *(
            [
                f"- Whole-file decision ID: `{review['whole_file_decision']['decision_id']}`",
                f"- Whole-file outcome: `{review['whole_file_decision']['outcome']}`",
                f"- Whole-file review revision: `{review['whole_file_decision']['review_revision']}`",
            ]
            if review.get("whole_file_decision")
            else []
        ),
        f"- Review rows: {review['total_review_rows']}",
        f"- Coverage complete: {str(review['review_coverage']['complete']).lower()}",
        f"- Accepted-outcome rows: {review['accepted_count']}",
        f"- Rejected rows: {review['rejected_count']}",
        f"- Revise and rereview rows: {review['revise_and_rereview_count']}",
        f"- Review issue counts: `{json.dumps(review['review_issue_counts'], sort_keys=True)}`",
        "",
        "## Validator",
        "",
        f"- Final status: `{validator['status']}`",
        f"- Decision reasons: `{', '.join(validator['decision_reasons']) if validator['decision_reasons'] else 'none'}`",
        "",
        "## Limitations",
        "",
        "- Text admission does not prove acoustic suitability.",
        "- No synthetic holdout exists.",
        "- No data acceptance certificate was issued.",
        "- `TRAINING_ELIGIBLE` was not produced.",
        "",
    ]
    atomic_write_text(path, "\n".join(lines))


def run_review_admission(
    *,
    generation_config_path: Path,
    data_quality_config_path: Path,
    retired_registry_path: Path,
    require_status: str,
    whole_file_outcome: str | None = None,
    review_revision: str | None = None,
    decision_id: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], int]:
    generation_config = load_review_config(generation_config_path)
    paths = default_paths(generation_config)
    source_rows = load_jsonl(paths["pre_review"])
    source_hash = sha256_file(paths["pre_review"])
    if generation_config.get("corpus_id") == "sl-corpus-v2-gams-candidate-reservoir-v1" and source_hash != EXPECTED_PRE_REVIEW_SHA256:
        raise ValueError(f"unexpected pre-review reservoir SHA256: {source_hash}")
    template_hash = sha256_file(paths["review_template"]) if paths["review_template"].exists() else None
    whole_file_requested = any(
        value is not None
        for value in (whole_file_outcome, review_revision, decision_id, expected_corpus_sha256, expected_rows)
    )
    whole_file_decision: dict[str, Any] | None = None
    if whole_file_requested:
        missing = [
            name
            for name, value in (
                ("whole_file_outcome", whole_file_outcome),
                ("review_revision", review_revision),
                ("decision_id", decision_id),
                ("expected_corpus_sha256", expected_corpus_sha256),
                ("expected_rows", expected_rows),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"whole-file review mode requires: {', '.join(missing)}")
        decisions, read_issues = build_whole_file_decisions(
            source_rows=source_rows,
            outcome=str(whole_file_outcome),
            review_revision=str(review_revision),
            decision_id=str(decision_id),
            expected_corpus_sha256=str(expected_corpus_sha256),
            expected_rows=int(expected_rows),
            actual_corpus_sha256=source_hash,
        )
        review_sheet_hash = None
        tsv_rows_by_id = None
        whole_file_decision = {
            "schema_version": WHOLE_FILE_DECISION_VERSION,
            "decision_id": str(decision_id),
            "outcome": str(whole_file_outcome),
            "review_revision": str(review_revision),
            "corpus_sha256": str(expected_corpus_sha256),
            "row_count": int(expected_rows),
        }
    else:
        decisions, read_issues, review_sheet_hash = read_review_tsv(paths["review_sheet"])
        tsv_rows_by_id = read_tsv_rows_by_id(paths["review_sheet"])
    coverage_issues = validate_review_coverage(source_rows=source_rows, decisions=decisions, tsv_rows_by_id=tsv_rows_by_id)
    review_issues = [*read_issues, *coverage_issues]
    source_by_id = {str(row["candidate_id"]): row for row in source_rows}
    accepted, accepted_reviews, rejected_by_human, outcome_counts = accepted_records_and_reviews(
        source_rows,
        decisions,
        review_errors=review_issues,
    )
    decisions_payload = [
        decision.decision_record(source_row=source_by_id.get(decision.candidate_id))
        for decision in sorted(decisions, key=lambda item: (item.candidate_id, item.line_number))
    ]
    atomic_write_jsonl(paths["decisions"], decisions_payload)
    atomic_write_jsonl(paths["accepted_candidates"], accepted)
    atomic_write_jsonl(paths["rejected_by_human"], rejected_by_human)

    atomic_write_jsonl(paths["accepted_review"], accepted_reviews)
    accepted_review_path: Path | None = paths["accepted_review"]

    protected_index_paths, protected_index_identities = ensure_protected_indexes(paths)
    validation_report, _local_review = validate_accepted_subset(
        data_quality_config_path=data_quality_config_path,
        accepted_candidates_path=paths["accepted_candidates"],
        accepted_review_path=accepted_review_path,
        protected_index_paths=protected_index_paths,
        output_report_path=paths["post_review_validation"],
        local_review_output_path=paths["post_review_local_review"],
        retired_registry_path=retired_registry_path,
        partition_role=str(generation_config.get("partition_role", "synthetic_candidate")),
        corpus_id=f"{generation_config['corpus_id']}-reviewed",
    )
    if review_issues:
        status = review_error_status(review_issues) or "DRAFT"
        validation_report = {
            **validation_report,
            "final_text_status": status,
            "decision_reasons": sorted(set(validation_report.get("decision_reasons", [])) | {issue.code for issue in review_issues}),
            "review_ingestion_issues": [issue.public() for issue in review_issues],
        }
        assert_privacy_safe_report(validation_report)
        atomic_write_json(paths["post_review_validation"], validation_report)

    accepted_hash = sha256_file(paths["accepted_candidates"]) if paths["accepted_candidates"].exists() else None
    public_payload = build_public_report(
        generation_config=generation_config,
        source_count=len(source_rows),
        source_hash=source_hash,
        review_template_hash=template_hash,
        review_sheet_hash=review_sheet_hash,
        decisions=decisions,
        review_issues=review_issues,
        accepted_count=len(accepted),
        rejected_count=sum(count for outcome, count in outcome_counts.items() if outcome in REJECT_OUTCOMES),
        revise_count=outcome_counts.get("REVISE_AND_REREVIEW", 0),
        accepted_partition_hash=accepted_hash,
        protected_index_identities=protected_index_identities,
        validation_report=validation_report,
        review_mode="whole_file" if whole_file_requested else "row_tsv",
        whole_file_decision=whole_file_decision,
        partition_role=str(generation_config.get("partition_role", "synthetic_candidate")),
        reviewed_corpus_id=f"{generation_config['corpus_id']}-reviewed",
    )
    atomic_write_json(paths["public_json"], public_payload)
    write_public_markdown(paths["public_markdown"], public_payload)
    status = str(public_payload["validator"]["status"])
    return public_payload, 0 if status == require_status else 1
