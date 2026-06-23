from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from slaif_asr.data_quality import (
    ALGORITHM_VERSION,
    NORMALIZER_VERSION,
    PROTECTED_INDEX_SCHEMA_VERSION,
    assert_privacy_safe_report,
    atomic_write_json,
    build_protected_index_payload,
    canonical_json_sha256,
    fingerprint_hash,
    number_masked_form,
    sha256_file,
    surface_form,
    validate_corpus,
)
from slaif_asr.slovenian_curriculum import CurriculumRecord, validate_collection


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/data_quality/training_text_v1.json"
RETIRED_PATH = REPO_ROOT / "configs/data_quality/retired_corpora.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_retired() -> dict:
    return json.loads(RETIRED_PATH.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def record(
    candidate_id: str,
    text: str,
    *,
    role: str = "synthetic_candidate",
    source_id: str | None = None,
    source_family_id: str | None = None,
    utterance_family_id: str | None = None,
    template_family_id: str | None = None,
    source_type: str = "generated_text",
    minimal_pair: dict | None = None,
    entities: list[dict] | None = None,
    source_recording_id: str | None = None,
) -> dict:
    source_id = source_id or f"src-{candidate_id}"
    source_family_id = source_family_id or f"source-family-{candidate_id}"
    utterance_family_id = utterance_family_id or f"utt-{candidate_id}"
    row = {
        "schema_version": "2.0",
        "candidate_id": candidate_id,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": role,
        "source_type": source_type,
        "source_id": source_id,
        "source_family_id": source_family_id,
        "template_family_id": template_family_id,
        "utterance_family_id": utterance_family_id,
        "phenomena": ["ordinary"],
        "domain": "fixture",
        "license": "fixture",
        "generation": {
            "system": "project-generated",
            "method": "direct-language-generation",
            "round": 0,
            "prompt_revision": "fixture-v1",
            "model_identity": "not-exposed-by-execution-runtime",
            "seed": None,
        }
        if source_type == "generated_text"
        else None,
        "entities": entities or [],
        "minimal_pair": minimal_pair,
    }
    if source_recording_id:
        row["source_recording_id"] = source_recording_id
    return row


def review_rows(rows: list[dict], *, outcome: str = "ACCEPT", minimal_pair_approved: bool = False) -> list[dict]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "outcome": outcome,
            "review_revision": "fixture-review-v1",
            "reviewer_approval": "fixture-reference",
            "reason_codes": [],
            "minimal_pair_approved": minimal_pair_approved,
        }
        for row in rows
    ]


def fake_index(gate_id: str, *, stale: bool = False, surface_texts: list[str] | None = None, number_texts: list[str] | None = None) -> dict:
    manifest_sha = "a" * 64
    reference_sha = "b" * 64
    return {
        "schema_version": PROTECTED_INDEX_SCHEMA_VERSION,
        "gate_id": gate_id,
        "manifest_sha256": manifest_sha,
        "metadata_manifest_sha256": ("c" * 64) if stale else manifest_sha,
        "reference_manifest_sha256": reference_sha,
        "metadata_reference_manifest_sha256": reference_sha,
        "row_count": 2,
        "normalizer": NORMALIZER_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint_versions": {
            "surface": "surface-normalized-v1",
            "number_masked": "number-masked-v1",
        },
        "surface_hashes": sorted(fingerprint_hash(surface_form(text)) for text in (surface_texts or [])),
        "number_masked_hashes": sorted(fingerprint_hash(number_masked_form(text)) for text in (number_texts or [])),
    }


def write_fake_indexes(tmp: Path, *, stale: bool = False, surface_texts: list[str] | None = None, number_texts: list[str] | None = None) -> list[Path]:
    paths = []
    for gate_id in ("fleurs-sl-si-test-full-v2", "artur-j-public-gate-v1"):
        path = tmp / f"{gate_id}.hash-index.json"
        atomic_write_json(path, fake_index(gate_id, stale=stale, surface_texts=surface_texts, number_texts=number_texts))
        paths.append(path)
    return paths


def good_rows() -> tuple[list[dict], list[dict], list[dict]]:
    candidates = [
        record("cand-001", "Danes na tržnici prodajajo sveže češnje.", source_id="src-001"),
        record("cand-002", "Profesorica pojasni nalogo pred začetkom vaj.", source_id="src-002"),
        record("cand-003", "Včeraj je pilot mirno pristal ob obali.", source_id="src-003"),
    ]
    selected = [
        record(
            "train-001",
            "Danes na tržnici prodajajo sveže češnje.",
            role="selected_training",
            source_id="src-001",
            source_family_id="source-family-cand-001",
            utterance_family_id="utt-cand-001",
        ),
        record(
            "train-002",
            "Profesorica pojasni nalogo pred začetkom vaj.",
            role="selected_training",
            source_id="src-002",
            source_family_id="source-family-cand-002",
            utterance_family_id="utt-cand-002",
        ),
    ]
    holdout = [
        record("hold-001", "Na vrtu raste star oreh ob leseni ograji.", role="synthetic_holdout"),
        record("hold-002", "Mlada zdravnica pripravi mirno razlago.", role="synthetic_holdout"),
        record("hold-003", "Oblačno jutro prinese hladen veter.", role="synthetic_holdout"),
    ]
    return candidates, selected, holdout


class DataQualityValidatorTests(unittest.TestCase):
    def run_validator(
        self,
        tmp: Path,
        partitions: dict[str, list[dict]],
        *,
        reviews: list[dict] | None = None,
        protected_indexes: list[Path] | None = None,
        config: dict | None = None,
    ) -> tuple[dict, list[dict]]:
        config = config or load_config()
        partition_paths: dict[str, Path] = {}
        for role, rows in partitions.items():
            path = tmp / f"{role}.jsonl"
            write_jsonl(path, rows)
            partition_paths[role] = path
        review_path = None
        if reviews is not None:
            review_path = tmp / "review.jsonl"
            write_jsonl(review_path, reviews)
        report, local_review = validate_corpus(
            corpus_id="fixture-corpus",
            config=config,
            config_sha256=canonical_json_sha256(config),
            retired_registry=load_retired(),
            partitions=partition_paths,
            linguistic_review_path=review_path,
            protected_index_paths=protected_indexes or [],
            repository_revision="fixture",
        )
        assert_privacy_safe_report(report)
        return report, local_review

    def test_positive_fixture_reaches_text_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            candidates, selected, holdout = good_rows()
            indexes = write_fake_indexes(tmp)
            report, local_review = self.run_validator(
                tmp,
                {
                    "synthetic_candidate": candidates,
                    "selected_training": selected,
                    "synthetic_holdout": holdout,
                },
                reviews=review_rows(candidates + selected + holdout),
                protected_indexes=indexes,
            )
            self.assertEqual(report["final_text_status"], "TEXT_ACCEPTED")
            self.assertEqual(local_review, [])

    def test_declared_minimal_pair_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            rows = [
                record(
                    "mp-001",
                    "Na liniji 12 stoji rdeč vlak.",
                    minimal_pair={"family_id": "mp-line-number", "contrast": "number"},
                ),
                record(
                    "mp-002",
                    "Na liniji 13 stoji rdeč vlak.",
                    minimal_pair={"family_id": "mp-line-number", "contrast": "number"},
                ),
            ]
            report, _ = self.run_validator(
                tmp,
                {"synthetic_candidate": rows},
                reviews=review_rows(rows, minimal_pair_approved=True),
                protected_indexes=write_fake_indexes(tmp),
            )
            self.assertEqual(report["final_text_status"], "TEXT_ACCEPTED")

    def test_same_template_with_row_numbers_rejected_and_legacy_accepts(self) -> None:
        text_a = "Skupina 1 pove, da Ana bere knjigo pri postaji 1 pri postaji 1."
        text_b = "Skupina 2 pove, da Ana bere knjigo pri postaji 2 pri postaji 2."
        legacy_records = [
            CurriculumRecord("1.0", "round1-0001", text_a, text_a, "sl-SI", "synthetic_candidate", ("ordinary",), {"system": "project-generated", "method": "direct-language-generation"}),
            CurriculumRecord("1.0", "round1-0002", text_b, text_b, "sl-SI", "synthetic_candidate", ("ordinary",), {"system": "project-generated", "method": "direct-language-generation"}),
        ]
        legacy_config = {
            "validation": {
                "near_duplicate_char_ngram": 5,
                "near_duplicate_jaccard_threshold": 0.82,
                "max_same_first_two_words": 5,
                "max_same_final_three_words": 5,
            }
        }
        validate_collection(legacy_records, expected_count=2, config=legacy_config)

        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            rows = [record("bad-001", text_a), record("bad-002", text_b)]
            report, _ = self.run_validator(
                tmp,
                {"synthetic_candidate": rows},
                reviews=review_rows(rows),
                protected_indexes=write_fake_indexes(tmp),
            )
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("metadata_token_in_text", report["decision_reasons"])

    def test_same_template_with_different_names_rejected(self) -> None:
        rows = [
            record("bad-ana", "Ana kupi svež kruh v mestni pekarni.", entities=[{"surface": "Ana", "type": "PERSON"}]),
            record("bad-maja", "Maja kupi svež kruh v mestni pekarni.", entities=[{"surface": "Maja", "type": "PERSON"}]),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("within_partition_entity_masked_collision", report["decision_reasons"])

    def test_same_body_with_artificial_prefix_or_suffix_rejected(self) -> None:
        base = "Maja zjutraj pregleda tiho knjižnico."
        rows = [
            record("prefix-001", f"Prva opomba pravi, da {base}"),
            record("prefix-002", f"Druga opomba pravi, da {base}"),
        ]
        suffix_rows = [
            record("suffix-001", f"{base} pri mirnem drevesu"),
            record("suffix-002", f"{base} pri starem mostu"),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows + suffix_rows}, reviews=review_rows(rows + suffix_rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "DRAFT")
            self.assertIn("fuzzy_similarity_review_required", report["decision_reasons"])

    def test_cross_partition_same_body_with_different_ids_rejected(self) -> None:
        candidates, selected, _ = good_rows()
        holdout = [record("hold-overlap", selected[0]["target_text"], role="synthetic_holdout")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(
                tmp,
                {
                    "synthetic_candidate": candidates,
                    "selected_training": selected,
                    "synthetic_holdout": holdout,
                },
                reviews=review_rows(candidates + selected + holdout),
                protected_indexes=write_fake_indexes(tmp),
            )
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("cross_partition_surface_overlap", report["decision_reasons"])

    def test_punctuation_and_casing_variants_rejected(self) -> None:
        rows = [
            record("case-001", "Ali prideš danes v mesto?"),
            record("case-002", "ali prideš danes v mesto"),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("within_partition_surface_collision", report["decision_reasons"])

    def test_suspicious_inflectional_and_threshold_variants_are_draft(self) -> None:
        rows = [
            record("sim-001", "Miren vlak danes počasi pelje mimo stare zelene postaje ob široki reki."),
            record("sim-002", "Miren vlak danes počasi pelje mimo stare zelene postaje ob širokem parku."),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, local_review = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "DRAFT")
            self.assertTrue(any(item["view"] == "character_5gram" for item in local_review))

    def test_legitimate_unrelated_sentences_sharing_function_words_pass(self) -> None:
        rows = [
            record("ok-001", "V mestu danes odprejo novo knjižnico."),
            record("ok-002", "V vasi zvečer zapoje otroški zbor."),
            record("ok-003", "V gozdu pohodniki opazijo srno."),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_ACCEPTED")

    def test_metadata_identifier_rejected(self) -> None:
        rows = [record("meta-001", "Kandidat 17 danes razloži miren primer.")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("metadata_token_in_text", report["decision_reasons"])

    def test_acoustic_variant_and_source_recording_crossing_partitions_rejected(self) -> None:
        candidates, selected, _ = good_rows()
        holdout = [
            record(
                "hold-acoustic",
                "Jutranja oddaja opiše promet na obvoznici.",
                role="synthetic_holdout",
                source_recording_id="recording-1",
                utterance_family_id=selected[0]["utterance_family_id"],
            )
        ]
        selected[0]["source_recording_id"] = "recording-1"
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": candidates, "selected_training": selected, "synthetic_holdout": holdout}, reviews=review_rows(candidates + selected + holdout), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("cross_partition_utterance_family_id_overlap", report["decision_reasons"])
            self.assertIn("cross_partition_source_recording_id_overlap", report["decision_reasons"])

    def test_malformed_slot_with_rejecting_review_is_rejected(self) -> None:
        rows = [record("bad-lang", "Ana gre iz Kamniku z velik dovolilnico.")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows, outcome="REJECT_GRAMMAR"), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("linguistic_review_not_accepted", report["decision_reasons"])

    def test_missing_review_and_revise_are_draft(self) -> None:
        rows = [record("draft-001", "Naravni stavek ostane brez pregleda.")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=None, protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "DRAFT")
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows, outcome="REVISE_AND_REREVIEW"), protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "DRAFT")

    def test_duplicate_and_unknown_review_rows_reject(self) -> None:
        rows = [record("review-001", "Pregledan stavek ima jasen namen.")]
        duplicate_reviews = review_rows(rows) + review_rows(rows)
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=duplicate_reviews, protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")

            unknown_reviews = review_rows(rows) + [
                {
                    "candidate_id": "unknown-001",
                    "outcome": "ACCEPT",
                    "review_revision": "fixture",
                    "reviewer_approval": "fixture",
                    "reason_codes": [],
                    "minimal_pair_approved": False,
                }
            ]
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=unknown_reviews, protected_indexes=write_fake_indexes(tmp))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("review_for_unknown_candidate", report["decision_reasons"])

    def test_missing_and_stale_protected_index_behaviors(self) -> None:
        rows = [record("prot-001", "Jasen stavek nima zaščitenega prekrivanja.")]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=[])
            self.assertEqual(report["final_text_status"], "DRAFT")
            self.assertIn("missing_required_protected_index", report["decision_reasons"])

            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp, stale=True))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("stale_protected_index_manifest", report["decision_reasons"])

    def test_protected_surface_rejects_number_overlap_drafts(self) -> None:
        row = record("prot-002", "Vlak številka 12 odpelje zjutraj.")
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": [row]}, reviews=review_rows([row]), protected_indexes=write_fake_indexes(tmp, surface_texts=[row["target_text"]]))
            self.assertEqual(report["final_text_status"], "TEXT_REJECTED")
            self.assertIn("protected_surface_overlap", report["decision_reasons"])

            report, _ = self.run_validator(tmp, {"synthetic_candidate": [row]}, reviews=review_rows([row]), protected_indexes=write_fake_indexes(tmp, number_texts=["Vlak številka 99 odpelje zjutraj."]))
            self.assertEqual(report["final_text_status"], "DRAFT")
            self.assertIn("protected_number_masked_overlap", report["decision_reasons"])

    def test_retired_artifact_hash_short_circuits(self) -> None:
        retired = load_retired()
        digest = retired["retired_corpora"][0]["sha256"]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            path = tmp / "candidate.jsonl"
            path.write_text("not parsed\n", encoding="utf-8")
            registry = {"schema_version": "1.0", "retired_corpora": [{"artifact": "fixture", "sha256": sha256_file(path)}]}
            report, _ = validate_corpus(
                corpus_id="retired-fixture",
                config=load_config(),
                config_sha256=canonical_json_sha256(load_config()),
                retired_registry=registry,
                partitions={"synthetic_candidate": path},
                linguistic_review_path=None,
                protected_index_paths=[],
                repository_revision="fixture",
            )
            self.assertEqual(report["final_text_status"], "RETIRED")
            self.assertEqual(retired["retired_corpora"][0]["sha256"], digest)

    def test_unrelated_rows_do_not_create_all_pairs_set(self) -> None:
        rows = [
            record("pair-001", "Čebelar pripravi okvirje za panj."),
            record("pair-002", "Meteorolog opazuje oblake nad morjem."),
            record("pair-003", "Arhitekt nariše tloris za šolo."),
            record("pair-004", "Kuhar nareže peteršilj za juho."),
        ]
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report, _ = self.run_validator(tmp, {"synthetic_candidate": rows}, reviews=review_rows(rows), protected_indexes=write_fake_indexes(tmp))
            stats = report["pair_candidate_comparison_counts"]
            self.assertLess(stats["candidate_pair_count"], stats["possible_all_pairs"])

    def test_deterministic_output_after_input_reordering(self) -> None:
        candidates, selected, holdout = good_rows()
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            indexes = write_fake_indexes(tmp)
            report_a, _ = self.run_validator(tmp, {"synthetic_candidate": candidates, "selected_training": selected, "synthetic_holdout": holdout}, reviews=review_rows(candidates + selected + holdout), protected_indexes=indexes)
            report_b, _ = self.run_validator(tmp, {"synthetic_candidate": list(reversed(candidates)), "selected_training": list(reversed(selected)), "synthetic_holdout": list(reversed(holdout))}, reviews=list(reversed(review_rows(candidates + selected + holdout))), protected_indexes=indexes)
            for report in (report_a, report_b):
                report.pop("input_partition_byte_hashes")
            self.assertEqual(json.dumps(report_a, ensure_ascii=False, sort_keys=True), json.dumps(report_b, ensure_ascii=False, sort_keys=True))

    def test_build_protected_index_payload_is_hash_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            manifest = tmp / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {"sample_id": "a", "text": "Zaščiten stavek ostane samo kot hash."},
                    {"sample_id": "b", "text": "Drugi zaščiten stavek ima številko 12."},
                ],
            )
            metadata = tmp / "metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "gate_id": "fake-gate",
                        "manifest_sha256": sha256_file(manifest),
                        "reference_manifest_sha256": "d" * 64,
                        "rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            payload = build_protected_index_payload(manifest, metadata)
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("Zaščiten", serialized)
            self.assertEqual(payload["row_count"], 2)

    def test_cli_writes_report_and_enforces_required_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            candidates, selected, holdout = good_rows()
            paths = {}
            for role, rows in {"synthetic_candidate": candidates, "selected_training": selected, "synthetic_holdout": holdout}.items():
                paths[role] = tmp / f"{role}.jsonl"
                write_jsonl(paths[role], rows)
            review_path = tmp / "review.jsonl"
            write_jsonl(review_path, review_rows(candidates + selected + holdout))
            indexes = write_fake_indexes(tmp)
            report_path = tmp / "report.json"
            local_path = tmp / "local.jsonl"
            command = [
                ".venv/bin/python",
                "scripts/validate_training_corpus.py",
                "--config",
                "configs/data_quality/training_text_v1.json",
                "--corpus-id",
                "fixture-cli",
                "--partition",
                f"synthetic_candidate={paths['synthetic_candidate']}",
                "--partition",
                f"selected_training={paths['selected_training']}",
                "--partition",
                f"synthetic_holdout={paths['synthetic_holdout']}",
                "--linguistic-review",
                str(review_path),
                "--protected-index",
                str(indexes[0]),
                "--protected-index",
                str(indexes[1]),
                "--output-report",
                str(report_path),
                "--local-review-output",
                str(local_path),
                "--require-status",
                "TEXT_ACCEPTED",
            ]
            completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["final_text_status"], "TEXT_ACCEPTED")


if __name__ == "__main__":
    unittest.main()
