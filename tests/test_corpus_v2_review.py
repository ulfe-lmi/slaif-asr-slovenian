from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.corpus_v2_review import (
    ReviewDecision,
    accepted_records_and_reviews,
    build_public_report,
    parse_bool,
    parse_reason_codes,
    read_review_tsv,
    read_tsv_rows_by_id,
    validate_accepted_subset,
    validate_review_coverage,
)
from slaif_asr.data_quality import (
    ALGORITHM_VERSION,
    NORMALIZER_VERSION,
    PROTECTED_INDEX_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_sha256,
    fingerprint_hash,
    number_masked_form,
    sha256_file,
    surface_form,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/data_quality/training_text_v1.json"
RETIRED_PATH = REPO_ROOT / "configs/data_quality/retired_corpora.json"


def record(candidate_id: str, text: str, *, minimal_pair: dict | None = None) -> dict:
    return {
        "schema_version": "2.0",
        "candidate_id": candidate_id,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": "synthetic_candidate",
        "source_type": "generated_text",
        "source_id": f"src-{candidate_id}",
        "source_family_id": f"family-{candidate_id}",
        "template_family_id": None,
        "utterance_family_id": f"utt-{candidate_id}",
        "phenomena": ["ordinary"],
        "domain": "fixture",
        "license": "fixture",
        "generation": {
            "system": "project-generated",
            "method": "fixture",
            "corpus_id": "fixture-corpus",
            "model_repository": "fixture",
            "model_revision": "a" * 40,
            "prompt_revision": "fixture-v1",
            "seed": 1,
        },
        "entities": [],
        "minimal_pair": minimal_pair,
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
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
        writer.writerows(rows)


def review_row(source: dict, *, outcome: str = "ACCEPT", revision: str = "review-v1", reason_codes: str = "", minimal_pair_approved: str = "False") -> dict[str, str]:
    return {
        "candidate_id": source["candidate_id"],
        "spoken_text": source["spoken_text"],
        "target_text": source["target_text"],
        "domain": source["domain"],
        "phenomena": ",".join(source["phenomena"]),
        "source_family_id": source["source_family_id"],
        "outcome": outcome,
        "review_revision": revision,
        "reason_codes": reason_codes,
        "minimal_pair_approved": minimal_pair_approved,
    }


def fake_index(path: Path, gate_id: str) -> Path:
    payload = {
        "schema_version": PROTECTED_INDEX_SCHEMA_VERSION,
        "gate_id": gate_id,
        "manifest_sha256": "a" * 64,
        "metadata_manifest_sha256": "a" * 64,
        "reference_manifest_sha256": "b" * 64,
        "metadata_reference_manifest_sha256": "b" * 64,
        "row_count": 1,
        "normalizer": NORMALIZER_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint_versions": {
            "surface": "surface-normalized-v1",
            "number_masked": "number-masked-v1",
        },
        "surface_hashes": [fingerprint_hash(surface_form("Zaščiten primer ostane samo kot hash."))],
        "number_masked_hashes": [fingerprint_hash(number_masked_form("Zaščiten primer ostane samo kot hash."))],
    }
    atomic_write_json(path, payload)
    return path


class CorpusV2ReviewAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            record("row-001", "Danes je mirno jutro."),
            record("row-002", "Jutri pride nova pošiljka."),
        ]

    def decisions_from_tsv(self, tmp: Path, rows: list[dict[str, str]]) -> tuple[list[ReviewDecision], list[str]]:
        path = tmp / "review.tsv"
        write_tsv(path, rows)
        decisions, read_issues, _digest = read_review_tsv(path)
        coverage = validate_review_coverage(source_rows=self.rows, decisions=decisions, tsv_rows_by_id=read_tsv_rows_by_id(path))
        return decisions, [issue.code for issue in [*read_issues, *coverage]]

    def test_complete_valid_review_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            decisions, issues = self.decisions_from_tsv(tmp, [review_row(row) for row in self.rows])
            self.assertEqual(issues, [])
            accepted, accepted_reviews, rejected, counts = accepted_records_and_reviews(self.rows, decisions, review_errors=[])
            self.assertEqual(len(accepted), 2)
            self.assertEqual(len(accepted_reviews), 2)
            self.assertEqual(rejected, [])
            self.assertEqual(counts["ACCEPT"], 2)

    def test_missing_review_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            decisions, issues = self.decisions_from_tsv(Path(tmp_text), [review_row(self.rows[0])])
            self.assertEqual(len(decisions), 1)
            self.assertIn("missing_review_row", issues)

    def test_duplicate_review_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            decisions, issues = self.decisions_from_tsv(Path(tmp_text), [review_row(self.rows[0]), review_row(self.rows[0]), review_row(self.rows[1])])
            self.assertEqual(len(decisions), 3)
            self.assertIn("duplicate_review_row", issues)

    def test_unknown_candidate_id(self) -> None:
        unknown = review_row(self.rows[0])
        unknown["candidate_id"] = "unknown-001"
        with tempfile.TemporaryDirectory() as tmp_text:
            _decisions, issues = self.decisions_from_tsv(Path(tmp_text), [unknown, review_row(self.rows[1])])
            self.assertIn("unknown_candidate_id", issues)
            self.assertIn("missing_review_row", issues)

    def test_blank_and_unsupported_outcome(self) -> None:
        blank = review_row(self.rows[0], outcome="")
        unsupported = review_row(self.rows[1], outcome="APPROVE")
        with tempfile.TemporaryDirectory() as tmp_text:
            _decisions, issues = self.decisions_from_tsv(Path(tmp_text), [blank, unsupported])
            self.assertIn("blank_outcome", issues)
            self.assertIn("unsupported_outcome", issues)

    def test_blank_review_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            _decisions, issues = self.decisions_from_tsv(Path(tmp_text), [review_row(row, revision="") for row in self.rows])
            self.assertIn("blank_review_revision", issues)

    def test_rejected_and_revise_outcomes_remain_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            decisions, issues = self.decisions_from_tsv(
                tmp,
                [
                    review_row(self.rows[0], outcome="REJECT_GRAMMAR", reason_codes="GRAMMAR"),
                    review_row(self.rows[1], outcome="REVISE_AND_REREVIEW", reason_codes="NEEDS_REVIEW"),
                ],
            )
            self.assertEqual(issues, [])
            accepted, accepted_reviews, rejected, counts = accepted_records_and_reviews(self.rows, decisions, review_errors=[])
            self.assertEqual(accepted, [])
            self.assertEqual(accepted_reviews, [])
            self.assertEqual(len(rejected), 2)
            self.assertEqual(counts["REJECT_GRAMMAR"], 1)
            self.assertEqual(counts["REVISE_AND_REREVIEW"], 1)

    def test_no_text_mutation(self) -> None:
        mutated = review_row(self.rows[0])
        mutated["spoken_text"] = "Drugačen stavek ne sme zamenjati vira."
        with tempfile.TemporaryDirectory() as tmp_text:
            _decisions, issues = self.decisions_from_tsv(Path(tmp_text), [mutated, review_row(self.rows[1])])
            self.assertIn("review_text_mismatch", issues)

    def test_deterministic_output_after_tsv_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            first, first_issues = self.decisions_from_tsv(tmp, [review_row(row) for row in self.rows])
            second, second_issues = self.decisions_from_tsv(tmp, list(reversed([review_row(row) for row in self.rows])))
            self.assertEqual(first_issues, [])
            self.assertEqual(second_issues, [])
            accepted_a, _, _, _ = accepted_records_and_reviews(self.rows, first, review_errors=[])
            accepted_b, _, _, _ = accepted_records_and_reviews(self.rows, second, review_errors=[])
            self.assertEqual([row["candidate_id"] for row in accepted_a], [row["candidate_id"] for row in accepted_b])

    def test_malformed_boolean_and_reason_codes(self) -> None:
        value, issue = parse_bool("maybe", line_number=7)
        self.assertIsNone(value)
        self.assertEqual(issue.code, "malformed_minimal_pair_approved")
        codes, reason_issue = parse_reason_codes("GOOD_CODE,bad code", line_number=8)
        self.assertEqual(codes, ())
        self.assertEqual(reason_issue.code, "malformed_reason_codes")

    def test_minimal_pair_approval_misuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            _decisions, issues = self.decisions_from_tsv(Path(tmp_text), [review_row(self.rows[0], minimal_pair_approved="True"), review_row(self.rows[1])])
            self.assertIn("minimal_pair_approval_without_source_family", issues)

    def test_privacy_safe_aggregate_report(self) -> None:
        payload = build_public_report(
            generation_config={"corpus_id": "sl-corpus-v2-gams-candidate-reservoir-v1"},
            source_count=2,
            source_hash="a" * 64,
            review_template_hash="b" * 64,
            review_sheet_hash="c" * 64,
            decisions=[
                ReviewDecision("row-001", "ACCEPT", "review-v1", (), False, 2),
                ReviewDecision("row-002", "REJECT_GRAMMAR", "review-v1", ("GRAMMAR",), False, 3),
            ],
            review_issues=[],
            accepted_count=1,
            rejected_count=1,
            revise_count=0,
            accepted_partition_hash="d" * 64,
            protected_index_identities=[],
            validation_report={
                "final_text_status": "TEXT_ACCEPTED",
                "decision_reasons": ["all_required_text_checks_passed"],
                "configuration_sha256": "e" * 64,
                "repository_revision": "f" * 40,
                "fingerprint_unique_counts": {"synthetic_candidate": {"surface": 1}},
                "template_family_counts": {"synthetic_candidate": {"largest_discovered_family_size": 1, "discovered_family_count": 1}},
                "protected_overlap_counts": {"surface_overlaps": 0, "number_masked_overlaps": 0},
            },
        )
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("row-001", serialized)
        self.assertNotIn("spoken_text", serialized)
        self.assertNotIn("/home/", serialized)

    def test_structural_revalidation_after_human_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            rows = [
                record("concentrated-001", "Skupna opomba navaja miren primer."),
                record("concentrated-002", "Skupna opomba navaja jasen primer."),
                record("concentrated-003", "Skupna opomba navaja kratek primer."),
            ]
            accepted_path = tmp / "accepted.jsonl"
            review_path = tmp / "review.jsonl"
            report_path = tmp / "report.json"
            local_path = tmp / "local.jsonl"
            atomic_write_jsonl(accepted_path, rows)
            atomic_write_jsonl(
                review_path,
                [
                    {
                        "candidate_id": row["candidate_id"],
                        "outcome": "ACCEPT",
                        "review_revision": "review-v1",
                        "reason_codes": [],
                        "minimal_pair_approved": False,
                    }
                    for row in rows
                ],
            )
            indexes = [
                fake_index(tmp / "fleurs.json", "fleurs-sl-si-test-full-v2"),
                fake_index(tmp / "artur.json", "artur-j-public-gate-v1"),
            ]
            report, _ = validate_accepted_subset(
                data_quality_config_path=CONFIG_PATH,
                accepted_candidates_path=accepted_path,
                accepted_review_path=review_path,
                protected_index_paths=indexes,
                output_report_path=report_path,
                local_review_output_path=local_path,
                retired_registry_path=RETIRED_PATH,
            )
            self.assertNotEqual(report["final_text_status"], "TEXT_ACCEPTED")
            self.assertIn("prefix_concentration", report["decision_reasons"])


if __name__ == "__main__":
    unittest.main()
