from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.corpus_v2_holdout import (
    build_prompt,
    build_record,
    fixed_counts_by_cell,
    fixed_holdout_path,
    load_config,
    prompt_mentions_forbidden_source,
    public_json_report_path,
    public_markdown_report_path,
    review_capsule_tsv_path,
    select_fixed_by_cell,
    write_public_reports,
    write_review_capsule,
)
from slaif_asr.data_quality import (
    ALGORITHM_VERSION,
    NORMALIZER_VERSION,
    PROTECTED_INDEX_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_sha256,
    fingerprint_hash,
    load_json,
    number_masked_form,
    sha256_file,
    surface_form,
    validate_corpus,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/generation/slovenian_corpus_v2_holdout_v1.json"
DATA_QUALITY_CONFIG = REPO_ROOT / "configs/data_quality/training_text_v1.json"
RETIRED_REGISTRY = REPO_ROOT / "configs/data_quality/retired_corpora.json"


def holdout_config(tmp: Path | None = None) -> dict:
    config = load_config(CONFIG_PATH)
    if tmp is not None:
        config = dict(config)
        config["run_directory"] = str(tmp / "run")
        config["public_reports"] = {
            "json": str(tmp / "report.json"),
            "markdown": str(tmp / "report.md"),
        }
    return config


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
        "surface_hashes": [fingerprint_hash(surface_form("Zaščiten stavek ostane skrit."))],
        "number_masked_hashes": [fingerprint_hash(number_masked_form("Zaščiten stavek ostane skrit."))],
    }
    atomic_write_json(path, payload)
    return path


def make_records(config: dict, *, per_cell: int = 13) -> list[dict]:
    words = [
        "bukev",
        "cvet",
        "dež",
        "gora",
        "hiša",
        "izvir",
        "javor",
        "kamen",
        "lipa",
        "morje",
        "nebo",
        "otok",
        "polje",
        "reka",
        "sonce",
        "trava",
        "ulica",
        "vlak",
        "zvon",
        "žarek",
    ]
    rows: list[dict] = []
    for cell in config["prompt_cells"]:
        for ordinal in range(1, per_cell + 1):
            token = words[(ordinal + len(rows)) % len(words)]
            text = f"Danes {token} mirno spremlja prijeten pogovor v mestu."
            rows.append(
                build_record(
                    config=config,
                    cell=cell,
                    attempt_index=0,
                    output_ordinal=ordinal,
                    text=text,
                    extraction_mode="fixture",
                )
            )
    return rows


def dq_record(candidate_id: str, text: str, *, role: str, source_family_id: str = "family-ok") -> dict:
    return {
        "schema_version": "2.0",
        "candidate_id": candidate_id,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": role,
        "source_type": "generated_text",
        "source_id": f"src-{candidate_id}",
        "source_family_id": source_family_id,
        "template_family_id": None,
        "utterance_family_id": f"utt-{candidate_id}",
        "phenomena": ["ordinary"],
        "domain": "fixture",
        "license": "fixture",
        "generation": {
            "system": "project-generated",
            "method": "fixture",
            "corpus_id": "fixture",
            "model_repository": "fixture",
            "model_revision": "a" * 40,
            "prompt_revision": "fixture-v1",
            "seed": 1,
        },
        "entities": [],
        "minimal_pair": None,
    }


def review_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "outcome": "ACCEPT",
            "review_revision": "fixture-review",
            "reason_codes": [],
            "minimal_pair_approved": False,
        }
        for row in rows
    ]


class CorpusV2HoldoutTests(unittest.TestCase):
    def test_holdout_record_role_and_ids_out_of_text(self) -> None:
        config = holdout_config()
        row = build_record(
            config=config,
            cell=config["prompt_cells"][0],
            attempt_index=0,
            output_ordinal=7,
            text="Zvečer se dobimo pred kulturnim domom.",
            extraction_mode="fixture",
        )
        self.assertEqual(row["schema_version"], "2.0")
        self.assertEqual(row["partition_role"], "synthetic_holdout")
        self.assertEqual(row["source_type"], "generated_text")
        self.assertEqual(row["spoken_text"], row["target_text"])
        self.assertEqual(row["template_family_id"], None)
        self.assertEqual(row["minimal_pair"], None)
        self.assertEqual(row["utterance_family_id"], row["candidate_id"])
        self.assertNotIn(row["candidate_id"], row["spoken_text"])

    def test_prompt_has_no_candidate_or_protected_data(self) -> None:
        config = holdout_config()
        prompt = build_prompt(config["prompt_cells"][0], requested_rows=20)
        self.assertFalse(prompt_mentions_forbidden_source(prompt, config))
        self.assertIn("Ne uporabljaj oštevilčenja", prompt)
        self.assertNotIn("FLEURS", prompt)
        self.assertNotIn("ARTUR", prompt)
        self.assertNotIn(config["candidate_source"]["corpus_id"], prompt)

    def test_deterministic_twelve_per_cell_selection(self) -> None:
        config = holdout_config()
        rows = make_records(config, per_cell=13)
        fixed_a, rejected_a = select_fixed_by_cell(rows, config)
        fixed_b, rejected_b = select_fixed_by_cell(list(reversed(rows)), config)
        self.assertEqual([row["candidate_id"] for row in fixed_a], [row["candidate_id"] for row in fixed_b])
        self.assertEqual(fixed_counts_by_cell(fixed_a), {cell["cell_id"]: 12 for cell in config["prompt_cells"]})
        self.assertEqual(len(fixed_a), 96)
        self.assertEqual(len(rejected_a), 8)
        self.assertEqual([item.candidate_id for item in rejected_a], [item.candidate_id for item in rejected_b])

    def test_cell_shortfall_fails(self) -> None:
        config = holdout_config()
        rows = make_records(config, per_cell=12)
        rows = [row for row in rows if row["generation"]["prompt_cell"] != "holdout-cell08"]
        with self.assertRaisesRegex(ValueError, "shortfall"):
            select_fixed_by_cell(rows, config)

    def test_candidate_holdout_source_family_overlap_rejected(self) -> None:
        candidate = [dq_record("cand-001", "Danes je mirno jutro v vasi.", role="synthetic_candidate", source_family_id="shared-family")]
        holdout = [dq_record("hold-001", "Popoldne se otroci igrajo na dvorišču.", role="synthetic_holdout", source_family_id="shared-family")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            candidate_path = tmp / "candidate.jsonl"
            holdout_path = tmp / "holdout.jsonl"
            review_path = tmp / "review.jsonl"
            atomic_write_jsonl(candidate_path, candidate)
            atomic_write_jsonl(holdout_path, holdout)
            atomic_write_jsonl(review_path, review_rows(candidate))
            indexes = [
                fake_index(tmp / "fleurs.json", "fleurs-sl-si-test-full-v2"),
                fake_index(tmp / "artur.json", "artur-j-public-gate-v1"),
            ]
            config = load_json(DATA_QUALITY_CONFIG)
            report, _ = validate_corpus(
                corpus_id="fixture",
                config=config,
                config_sha256=canonical_json_sha256(config),
                retired_registry=load_json(RETIRED_REGISTRY),
                partitions={
                    "synthetic_candidate": candidate_path,
                    "synthetic_holdout": holdout_path,
                },
                linguistic_review_path=review_path,
                protected_index_paths=indexes,
                repository_revision="fixture",
            )
        self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
        self.assertIn("cross_partition_source_family_id_overlap", report["decision_reasons"])

    def test_candidate_holdout_fingerprint_overlap_rejected(self) -> None:
        candidate = [dq_record("cand-001", "Danes je mirno jutro v vasi.", role="synthetic_candidate", source_family_id="candidate-family")]
        holdout = [dq_record("hold-001", "danes je mirno jutro v vasi", role="synthetic_holdout", source_family_id="holdout-family")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            candidate_path = tmp / "candidate.jsonl"
            holdout_path = tmp / "holdout.jsonl"
            review_path = tmp / "review.jsonl"
            atomic_write_jsonl(candidate_path, candidate)
            atomic_write_jsonl(holdout_path, holdout)
            atomic_write_jsonl(review_path, review_rows(candidate))
            indexes = [
                fake_index(tmp / "fleurs.json", "fleurs-sl-si-test-full-v2"),
                fake_index(tmp / "artur.json", "artur-j-public-gate-v1"),
            ]
            config = load_json(DATA_QUALITY_CONFIG)
            report, _ = validate_corpus(
                corpus_id="fixture",
                config=config,
                config_sha256=canonical_json_sha256(config),
                retired_registry=load_json(RETIRED_REGISTRY),
                partitions={
                    "synthetic_candidate": candidate_path,
                    "synthetic_holdout": holdout_path,
                },
                linguistic_review_path=review_path,
                protected_index_paths=indexes,
                repository_revision="fixture",
            )
        self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
        self.assertIn("cross_partition_surface_overlap", report["decision_reasons"])

    def test_review_capsule_has_96_rows_and_no_implicit_accept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = holdout_config(tmp)
            rows, _ = select_fixed_by_cell(make_records(config, per_cell=12), config)
            atomic_write_jsonl(fixed_holdout_path(config), rows)
            write_review_capsule(config)
            lines = review_capsule_tsv_path(config).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines) - 1, 96)
            self.assertNotIn("\tACCEPT\t", "\n".join(lines))

    def test_privacy_safe_public_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = holdout_config(tmp)
            rows, _ = select_fixed_by_cell(make_records(config, per_cell=12), config)
            atomic_write_jsonl(fixed_holdout_path(config), rows)
            atomic_write_jsonl(tmp / "run" / "generated-all.local.jsonl", rows)
            atomic_write_jsonl(tmp / "run" / "rejected.local.jsonl", [])
            atomic_write_json(tmp / "run" / "validation.local.json", {
                "final_text_status": "DRAFT",
                "decision_reasons": ["missing_linguistic_review"],
                "protected_indexes": [],
                "protected_overlap_counts": {"surface_overlaps": 0, "number_masked_overlaps": 0},
                "cross_partition_overlap_counts": {},
                "template_family_counts": {
                    "synthetic_holdout": {
                        "declared_family_count": 0,
                        "discovered_family_count": 96,
                        "largest_discovered_family_size": 1,
                    }
                },
                "checks": {},
                "fuzzy_review_pair_counts": {"pairs_requiring_review": 0},
            })
            payload = write_public_reports(config)
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn(rows[0]["spoken_text"], serialized)
            self.assertNotIn(rows[0]["candidate_id"], serialized)
            self.assertNotIn(str(tmp), serialized)
            self.assertTrue(public_json_report_path(config).exists())
            self.assertTrue(public_markdown_report_path(config).exists())


if __name__ == "__main__":
    unittest.main()
