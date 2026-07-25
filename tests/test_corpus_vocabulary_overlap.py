from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_corpus_vocabulary_overlap import (
    ARTUR_J_DATASET_ID,
    CLASSIFICATION_COMPLETE,
    DatasetBinding,
    DatasetVocabulary,
    FLEURS_V2_DATASET_ID,
    SCALE8000_DATASET_ID,
    assert_aggregate_only_report,
    build_aggregate_report,
    extract_normalized_tokens,
    overlap_metrics,
    render_markdown,
    summarize_texts,
)


def fake_dataset(
    key: str,
    dataset_id: str,
    role: str,
    texts: list[str],
) -> DatasetVocabulary:
    return DatasetVocabulary(
        binding=DatasetBinding(
            key=key,
            dataset_id=dataset_id,
            source_role=role,
            source_sha256={"scale8000": "a", "fleurs_v2": "b", "artur_j": "c"}[key] * 64,
        ),
        primary=summarize_texts(texts, include_numeric_only=False),
        secondary=summarize_texts(texts, include_numeric_only=True),
    )


def fake_report() -> dict[str, object]:
    return build_aggregate_report(
        fake_dataset("scale8000", SCALE8000_DATASET_ID, "synthetic_training_text", ["Čas teče ena dva tri"]),
        fake_dataset("fleurs_v2", FLEURS_V2_DATASET_ID, "immutable_real_gate", ["Čas teče ena dva"]),
        fake_dataset("artur_j", ARTUR_J_DATASET_ID, "immutable_real_gate", ["Čas teče tri"]),
    )


class CorpusVocabularyOverlapTests(unittest.TestCase):
    def test_unicode_tokenization_preserves_slovenian_diacritics(self) -> None:
        self.assertEqual(
            extract_normalized_tokens("Čas, Šola in Žoga."),
            ["čas", "šola", "in", "žoga"],
        )

    def test_casefolding_merges_uppercase_and_lowercase(self) -> None:
        summary = summarize_texts(["ČAS čas Čas"], include_numeric_only=False)
        self.assertEqual(summary.token_counts, {"čas": 3})

    def test_punctuation_is_split(self) -> None:
        self.assertEqual(
            extract_normalized_tokens("ena,dve;tri-štiri/pet"),
            ["ena", "dve", "tri", "štiri", "pet"],
        )

    def test_numeric_only_tokens_are_excluded_from_primary_mode(self) -> None:
        self.assertEqual(extract_normalized_tokens("vlak 2026 linija7"), ["vlak", "linija7"])

    def test_numeric_only_tokens_are_included_in_secondary_mode(self) -> None:
        self.assertEqual(
            extract_normalized_tokens("vlak 2026 linija7", include_numeric_only=True),
            ["vlak", "2026", "linija7"],
        )

    def test_no_lemmatization_or_stemming_occurs(self) -> None:
        summary = summarize_texts(["hiša hiše hiši"], include_numeric_only=False)
        self.assertEqual(set(summary.token_counts), {"hiša", "hiše", "hiši"})

    def test_overlap_coverage_is_computed_correctly(self) -> None:
        left = summarize_texts(["ena ena dve"], include_numeric_only=False)
        right = summarize_texts(["ena tri tri"], include_numeric_only=False)
        metrics = overlap_metrics(left, right)
        self.assertEqual(metrics["shared_unique_word_forms"], 1)
        self.assertEqual(metrics["coverage_of_right_unique_vocab"], 0.5)
        self.assertEqual(metrics["right_unique_forms_absent"], 1)

    def test_token_mass_coverage_is_computed_correctly(self) -> None:
        left = summarize_texts(["ena dve"], include_numeric_only=False)
        right = summarize_texts(["ena tri tri"], include_numeric_only=False)
        metrics = overlap_metrics(left, right)
        self.assertEqual(metrics["right_tokens_covered"], 1)
        self.assertAlmostEqual(metrics["token_mass_coverage_of_right_tokens"], 1 / 3)
        self.assertEqual(metrics["right_tokens_using_absent_forms"], 2)

    def test_report_redacts_and_rejects_local_paths(self) -> None:
        report = fake_report()
        with tempfile.TemporaryDirectory() as temporary_directory:
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn(temporary_directory, serialized)
            unsafe = copy.deepcopy(report)
            unsafe["datasets"]["fleurs_v2"]["source_path"] = str(Path(temporary_directory) / "gate.jsonl")
            with self.assertRaisesRegex(ValueError, "forbidden field|absolute path"):
                assert_aggregate_only_report(unsafe)

    def test_report_rejects_raw_references(self) -> None:
        report = fake_report()
        report["raw_references"] = ["synthetic fixture only"]
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            assert_aggregate_only_report(report)

    def test_report_rejects_full_gate_word_lists(self) -> None:
        report = fake_report()
        report["full_gate_wordlists"] = ["fixture", "forms"]
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            assert_aggregate_only_report(report)

    def test_report_rejects_positive_leakage_safety_flag(self) -> None:
        report = fake_report()
        report["safety"]["raw_references_committed"] = True
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_aggregate_only_report(report)

    def test_json_schema_contains_aggregate_counts_only(self) -> None:
        report = fake_report()
        self.assertEqual(report["classification"], CLASSIFICATION_COMPLETE)
        assert_aggregate_only_report(report)

        def walk(value: object) -> None:
            self.assertNotIsInstance(value, list)
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)

        walk(report)
        self.assertNotIn("synthetic fixture only", json.dumps(report, ensure_ascii=False))

    def test_markdown_contains_only_aggregate_sections(self) -> None:
        markdown = render_markdown(fake_report())
        self.assertIn("## Dataset sizes", markdown)
        self.assertIn("## Vocabulary overlap", markdown)
        self.assertIn("No missing form is included", markdown)
        self.assertNotIn("source_path", markdown)


if __name__ == "__main__":
    unittest.main()
