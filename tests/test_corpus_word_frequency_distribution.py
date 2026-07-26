from __future__ import annotations

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import scripts.analyze_corpus_word_frequency_distribution as frequency
from scripts.analyze_corpus_word_frequency_distribution import (
    ARTUR_J_DATASET_ID,
    CLASSIFICATION_COMPLETE,
    FLEURS_V2_DATASET_ID,
    SCALE8000_DATASET_ID,
    assert_aggregate_only_report,
    assert_markdown_aggregate_only,
    build_aggregate_report,
    concentration_metrics,
    confidence_interval,
    dataset_from_texts,
    frequency_of_frequency_profile,
    full_union_distribution_metrics,
    normalized_probabilities,
    render_markdown,
    run_bootstrap_analysis,
    shared_conditional_distribution_metrics,
    shared_frequency_ratio_metrics,
    spearman_rank_correlation,
    top_k_agreement_metrics,
)
from scripts.analyze_corpus_vocabulary_overlap import sha256_file


def fake_dataset(
    key: str,
    dataset_id: str,
    role: str,
    texts: list[str],
) -> frequency.DatasetFrequency:
    hash_character = {"scale8000": "a", "fleurs_v2": "b", "artur_j": "c"}[key]
    return dataset_from_texts(
        key=key,
        dataset_id=dataset_id,
        source_role=role,
        source_sha256=hash_character * 64,
        texts=texts,
    )


def fake_datasets() -> dict[str, frequency.DatasetFrequency]:
    return {
        "scale8000": fake_dataset(
            "scale8000",
            SCALE8000_DATASET_ID,
            "synthetic_training_text",
            ["skupno ena ena", "skupno dve", "skupno tri", "skupno ena"],
        ),
        "fleurs_v2": fake_dataset(
            "fleurs_v2",
            FLEURS_V2_DATASET_ID,
            "immutable_real_gate",
            ["skupno ena", "skupno štiri štiri", "skupno dve", "skupno štiri"],
        ),
        "artur_j": fake_dataset(
            "artur_j",
            ARTUR_J_DATASET_ID,
            "immutable_real_gate",
            ["skupno pet", "skupno ena", "skupno pet pet", "skupno dve"],
        ),
    }


def fake_report() -> dict[str, object]:
    datasets = fake_datasets()
    return build_aggregate_report(
        datasets["scale8000"],
        datasets["fleurs_v2"],
        datasets["artur_j"],
        bootstrap_replicates=20,
    )


class CorpusWordFrequencyDistributionTests(unittest.TestCase):
    def test_normalized_probabilities_sum_to_one(self) -> None:
        probabilities = normalized_probabilities({"ena": 2, "dve": 1})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(probabilities, {"ena": 2 / 3, "dve": 1 / 3})

    def test_identical_distributions_produce_zero_jsd(self) -> None:
        metrics = full_union_distribution_metrics({"ena": 2, "dve": 1}, {"ena": 2, "dve": 1})
        self.assertAlmostEqual(metrics["jensen_shannon_divergence_base2"], 0.0)

    def test_jsd_is_symmetric_and_bounded(self) -> None:
        forward = full_union_distribution_metrics({"ena": 3, "dve": 1}, {"ena": 1, "tri": 2})
        reverse = full_union_distribution_metrics({"ena": 1, "tri": 2}, {"ena": 3, "dve": 1})
        self.assertAlmostEqual(
            forward["jensen_shannon_divergence_base2"],
            reverse["jensen_shannon_divergence_base2"],
        )
        self.assertGreaterEqual(forward["jensen_shannon_divergence_base2"], 0.0)
        self.assertLessEqual(forward["jensen_shannon_divergence_base2"], 1.0)

    def test_identical_distributions_zero_hellinger_and_total_variation(self) -> None:
        metrics = full_union_distribution_metrics({"ena": 1, "dve": 1}, {"ena": 2, "dve": 2})
        self.assertAlmostEqual(metrics["hellinger_distance"], 0.0)
        self.assertAlmostEqual(metrics["total_variation_distance"], 0.0)

    def test_identical_distributions_have_unit_overlap(self) -> None:
        metrics = full_union_distribution_metrics({"ena": 1, "dve": 1}, {"ena": 2, "dve": 2})
        self.assertAlmostEqual(metrics["probability_mass_overlap_coefficient"], 1.0)

    def test_overlap_equals_one_minus_total_variation(self) -> None:
        metrics = full_union_distribution_metrics({"ena": 3, "dve": 1}, {"ena": 1, "tri": 2})
        self.assertAlmostEqual(
            metrics["probability_mass_overlap_coefficient"],
            1.0 - metrics["total_variation_distance"],
        )

    def test_disjoint_distributions_have_zero_overlap(self) -> None:
        metrics = full_union_distribution_metrics({"ena": 2}, {"dve": 3})
        self.assertAlmostEqual(metrics["probability_mass_overlap_coefficient"], 0.0)
        self.assertAlmostEqual(metrics["jensen_shannon_divergence_base2"], 1.0)

    def test_shared_vocabulary_renormalization_excludes_support_mismatch(self) -> None:
        metrics = shared_conditional_distribution_metrics(
            {"skupno": 1, "samo_a": 9},
            {"skupno": 2, "samo_b": 8},
        )
        self.assertEqual(metrics["shared_forms"], 1)
        self.assertAlmostEqual(metrics["shared_form_token_mass_in_a"], 0.1)
        self.assertAlmostEqual(metrics["shared_form_token_mass_in_b"], 0.2)
        self.assertAlmostEqual(metrics["conditional_jensen_shannon_divergence_base2"], 0.0)

    def test_shared_form_log_ratio_matches_hand_calculation(self) -> None:
        metrics = shared_frequency_ratio_metrics({"x": 3, "y": 1}, {"x": 1, "y": 1})
        expected_mean = (0.5 * math.log2(1.5) + 0.25 * 1.0) / 0.75
        self.assertAlmostEqual(metrics["weighted_mean_absolute_log2_ratio"], expected_mean)
        self.assertAlmostEqual(
            metrics["weighted_median_absolute_log2_ratio"],
            math.log2(1.5),
        )
        self.assertAlmostEqual(metrics["fraction_of_shared_mass_within_factor_2"], 1.0)

    def test_spearman_tie_handling_is_deterministic(self) -> None:
        counts_a = {"x": 3, "y": 3, "z": 1}
        counts_b = {"x": 1, "y": 1, "z": 3}
        first = spearman_rank_correlation(counts_a, counts_b)
        second = spearman_rank_correlation(dict(reversed(list(counts_a.items()))), counts_b)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 3)
        self.assertAlmostEqual(first[1], -1.0)

    def test_top_k_with_no_boundary_tie_preserves_effective_k(self) -> None:
        metrics = top_k_agreement_metrics(
            {"a": 5, "b": 4, "c": 3},
            {"a": 6, "b": 5, "d": 2},
            2,
        )
        self.assertEqual(metrics["effective_k_in_a"], 2)
        self.assertEqual(metrics["cutoff_occurrence_count_in_a"], 4)
        self.assertEqual(metrics["tie_expansion_count_in_a"], 0)
        self.assertEqual(metrics["effective_k_in_b"], 2)
        self.assertEqual(metrics["cutoff_occurrence_count_in_b"], 5)
        self.assertEqual(metrics["tie_expansion_count_in_b"], 0)

    def test_top_k_boundary_tie_expands_effective_k(self) -> None:
        metrics = top_k_agreement_metrics(
            {"a": 5, "b": 4, "c": 4, "d": 1},
            {"a": 5, "b": 4, "d": 3},
            2,
        )
        self.assertEqual(metrics["effective_k_in_a"], 3)
        self.assertEqual(metrics["cutoff_occurrence_count_in_a"], 4)
        self.assertEqual(metrics["tie_expansion_count_in_a"], 1)

    def test_top_k_is_independent_of_insertion_order_and_lexical_names(self) -> None:
        first = top_k_agreement_metrics(
            {"alpha": 5, "beta": 4, "gamma": 4},
            {"alpha": 5, "beta": 4, "delta": 4},
            2,
        )
        reordered_and_renamed = top_k_agreement_metrics(
            {"third": 4, "first": 5, "second": 4},
            {"fourth": 4, "second": 4, "first": 5},
            2,
        )
        self.assertEqual(first, reordered_and_renamed)

    def test_top_k_both_datasets_can_expand_by_different_amounts(self) -> None:
        metrics = top_k_agreement_metrics(
            {"shared_high": 5, "shared_tie": 4, "a_extra": 4},
            {
                "shared_high": 5,
                "shared_tie": 4,
                "b_extra_1": 4,
                "b_extra_2": 4,
            },
            2,
        )
        self.assertEqual(metrics["effective_k_in_a"], 3)
        self.assertEqual(metrics["tie_expansion_count_in_a"], 1)
        self.assertEqual(metrics["effective_k_in_b"], 4)
        self.assertEqual(metrics["tie_expansion_count_in_b"], 2)

    def test_top_k_agreement_uses_expanded_effective_sets(self) -> None:
        metrics = top_k_agreement_metrics(
            {"shared_high": 5, "shared_tie": 4, "a_extra": 4},
            {
                "shared_high": 5,
                "shared_tie": 4,
                "b_extra_1": 4,
                "b_extra_2": 4,
            },
            2,
        )
        self.assertEqual(metrics["shared_forms"], 2)
        self.assertAlmostEqual(metrics["jaccard_similarity"], 2 / 5)
        self.assertAlmostEqual(metrics["overlap_coefficient"], 2 / 3)

    def test_top_k_uses_available_vocabulary_when_smaller_than_requested(self) -> None:
        metrics = top_k_agreement_metrics(
            {"a": 3, "b": 2, "c": 1},
            {"b": 3, "d": 2},
            5,
        )
        self.assertEqual(metrics["effective_k_in_a"], 3)
        self.assertEqual(metrics["effective_k_in_b"], 2)
        self.assertEqual(metrics["shared_forms"], 1)
        self.assertAlmostEqual(metrics["jaccard_similarity"], 0.25)
        self.assertAlmostEqual(metrics["overlap_coefficient"], 0.5)

    def test_top_k_reports_do_not_emit_lexical_content(self) -> None:
        report = fake_report()
        markdown = render_markdown(report)
        serialized = json.dumps(report, ensure_ascii=False)
        for lexical_fixture in ("skupno", "štiri", "pet"):
            self.assertNotIn(lexical_fixture, markdown)
            self.assertNotIn(lexical_fixture, serialized)
        self.assertIn("Top-k sets are tie-inclusive", markdown)

    def test_frequency_of_frequency_bands_are_correct(self) -> None:
        counts = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 5,
            "e": 6,
            "f": 10,
            "g": 11,
            "h": 50,
            "i": 51,
        }
        profile = frequency_of_frequency_profile(counts)
        self.assertEqual(profile["exactly_1"]["unique_forms"], 1)
        self.assertEqual(profile["exactly_2"]["tokens_contributed"], 2)
        self.assertEqual(profile["3_to_5"]["tokens_contributed"], 8)
        self.assertEqual(profile["6_to_10"]["tokens_contributed"], 16)
        self.assertEqual(profile["11_to_50"]["tokens_contributed"], 61)
        self.assertEqual(profile["more_than_50"]["tokens_contributed"], 51)

    def test_entropy_and_concentration_match_hand_calculation(self) -> None:
        metrics = concentration_metrics({"a": 1, "b": 1})
        self.assertAlmostEqual(metrics["shannon_entropy_bits"], 1.0)
        self.assertAlmostEqual(metrics["normalized_shannon_entropy"], 1.0)
        self.assertAlmostEqual(metrics["simpson_concentration"], 0.5)
        self.assertAlmostEqual(metrics["effective_vocabulary_size_shannon"], 2.0)
        top_ten = metrics["top_k_token_mass"]["k_10"]
        self.assertEqual(top_ten["effective_k"], 2)
        self.assertAlmostEqual(top_ten["token_mass"], 1.0)

    def test_bootstrap_is_deterministic_for_seed_480049(self) -> None:
        datasets = fake_datasets()
        first = run_bootstrap_analysis(datasets, replicates=12, seed=480049)
        second = run_bootstrap_analysis(datasets, replicates=12, seed=480049)
        self.assertEqual(first, second)

    def test_bootstrap_confidence_intervals_are_ordered_and_finite(self) -> None:
        interval = confidence_interval([0.4, 0.1, 0.9, 0.2, 0.6])
        self.assertLessEqual(interval["p2_5"], interval["p50"])
        self.assertLessEqual(interval["p50"], interval["p97_5"])
        self.assertTrue(all(math.isfinite(value) for value in interval.values()))

    def test_cross_corpus_and_within_corpus_bootstrap_paths_are_distinct(self) -> None:
        cross, within = run_bootstrap_analysis(fake_datasets(), replicates=30, seed=480049)
        self.assertEqual(
            set(cross),
            {
                "scale8000_vs_fleurs_v2",
                "scale8000_vs_artur_j",
                "fleurs_v2_vs_artur_j",
            },
        )
        self.assertEqual(set(within), {"scale8000", "fleurs_v2", "artur_j"})
        self.assertNotEqual(
            cross["scale8000_vs_fleurs_v2"][
                "full_union_jensen_shannon_divergence_base2"
            ],
            within["scale8000"]["full_union_jensen_shannon_divergence_base2"],
        )

    def _assert_hash_mismatch_is_invalid(self, mismatched_dataset: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scale_path = root / "scale.jsonl"
            fleurs_path = root / "fleurs.jsonl"
            artur_path = root / "artur.jsonl"
            json_output = root / "report.json"
            markdown_output = root / "report.md"
            scale_path.write_text(
                json.dumps({"target_text": "Sintetični preizkus."}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            fleurs_path.write_text(
                json.dumps(
                    {"dataset": FLEURS_V2_DATASET_ID, "text": "Fleurs preizkus."},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            artur_path.write_text(
                json.dumps(
                    {"dataset": ARTUR_J_DATASET_ID, "text": "Artur preizkus."},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            expected_hashes = {
                "scale8000": sha256_file(scale_path),
                "fleurs_v2": sha256_file(fleurs_path),
                "artur_j": sha256_file(artur_path),
            }
            expected_hashes[mismatched_dataset] = "0" * 64
            stderr = io.StringIO()
            with (
                mock.patch.object(frequency, "EXPECTED_SCALE8000_ROWS", 1),
                mock.patch.object(frequency, "EXPECTED_FLEURS_V2_ROWS", 1),
                mock.patch.object(frequency, "EXPECTED_ARTUR_J_ROWS", 1),
                mock.patch.object(
                    frequency,
                    "EXPECTED_SCALE8000_SHA256",
                    expected_hashes["scale8000"],
                ),
                mock.patch.object(
                    frequency,
                    "EXPECTED_FLEURS_V2_SHA256",
                    expected_hashes["fleurs_v2"],
                ),
                mock.patch.object(
                    frequency,
                    "EXPECTED_ARTUR_J_SHA256",
                    expected_hashes["artur_j"],
                ),
                redirect_stderr(stderr),
            ):
                result = frequency.main(
                    [
                        "--scale8000-text",
                        str(scale_path),
                        "--fleurs-v2-manifest",
                        str(fleurs_path),
                        "--artur-j-manifest",
                        str(artur_path),
                        "--output-json",
                        str(json_output),
                        "--output-markdown",
                        str(markdown_output),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("EXPERIMENT_INVALID", stderr.getvalue())
            self.assertIn("source SHA256 does not match", stderr.getvalue())
            self.assertFalse(json_output.exists())
            self.assertFalse(markdown_output.exists())

    def test_scale8000_sha_mismatch_fails_closed(self) -> None:
        self._assert_hash_mismatch_is_invalid("scale8000")

    def test_fleurs_v2_sha_mismatch_fails_closed(self) -> None:
        self._assert_hash_mismatch_is_invalid("fleurs_v2")

    def test_artur_j_sha_mismatch_fails_closed(self) -> None:
        self._assert_hash_mismatch_is_invalid("artur_j")

    def test_raw_references_are_rejected_from_output(self) -> None:
        report = fake_report()
        report["raw_references"] = "fixture"
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            assert_aggregate_only_report(report)

    def test_word_lists_and_ranked_lists_are_rejected_from_output(self) -> None:
        for field in ("full_gate_wordlists", "ranked_words", "top_words"):
            with self.subTest(field=field):
                report = fake_report()
                report[field] = "fixture"
                with self.assertRaisesRegex(ValueError, "forbidden field"):
                    assert_aggregate_only_report(report)

    def test_local_paths_are_rejected_from_output(self) -> None:
        report = fake_report()
        report["analysis_role"] = str(Path.cwd().resolve() / "private" / "gate.jsonl")
        with self.assertRaisesRegex(ValueError, "absolute path"):
            assert_aggregate_only_report(report)

    def test_positive_leakage_flags_are_rejected(self) -> None:
        report = fake_report()
        report["safety"]["raw_references_committed"] = True
        with self.assertRaisesRegex(ValueError, "leakage|prohibited"):
            assert_aggregate_only_report(report)

    def test_markdown_contains_aggregate_sections_only(self) -> None:
        markdown = render_markdown(fake_report())
        assert_markdown_aggregate_only(markdown)
        self.assertIn("## Pairwise full-union distributions", markdown)
        self.assertIn("## Sampling uncertainty", markdown)
        self.assertIn("No ranked forms are emitted", markdown)
        self.assertNotIn("## Raw references", markdown)
        self.assertNotIn(str(Path.cwd().resolve()), markdown)

    def test_json_contains_no_lexical_arrays_or_per_word_fields(self) -> None:
        report = fake_report()
        assert_aggregate_only_report(report)

        def walk(value: object) -> None:
            self.assertNotIsInstance(value, list)
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn("per_word", key)
                    walk(child)

        walk(report)

    def test_non_finite_numbers_are_rejected(self) -> None:
        report = fake_report()
        report["datasets"]["scale8000"]["total_word_tokens"] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            assert_aggregate_only_report(report)

    def test_unexpected_top_level_fields_are_rejected(self) -> None:
        report = fake_report()
        report["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected top-level"):
            assert_aggregate_only_report(report)

    def test_report_classification_and_safety_boundary(self) -> None:
        report = fake_report()
        self.assertEqual(report["classification"], CLASSIFICATION_COMPLETE)
        self.assertTrue(all(value is False for value in report["safety"].values()))

    def test_work_order_0048_tokenizer_is_reused(self) -> None:
        dataset = dataset_from_texts(
            key="scale8000",
            dataset_id=SCALE8000_DATASET_ID,
            source_role="synthetic_training_text",
            source_sha256="a" * 64,
            texts=["ČAS, čas; 2026 linija7"],
        )
        self.assertEqual(dataset.token_counts, {"čas": 2, "linija7": 1})

    def test_markdown_validator_rejects_absolute_path(self) -> None:
        absolute_path = Path.cwd().resolve() / "private" / "gate.jsonl"
        with self.assertRaisesRegex(ValueError, "absolute path"):
            assert_markdown_aggregate_only(f"local input: {absolute_path}")

    def test_report_validator_rejects_arrays(self) -> None:
        report = fake_report()
        report["analysis_role"] = ["fixture"]
        with self.assertRaisesRegex(ValueError, "arrays"):
            assert_aggregate_only_report(report)

    def test_report_validator_rejects_per_word_frequency_field(self) -> None:
        report = fake_report()
        report["per_word_frequency"] = "fixture"
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            assert_aggregate_only_report(report)

    def test_report_validator_rejects_local_manifest_field(self) -> None:
        report = fake_report()
        report["local_manifest"] = "fixture"
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            assert_aggregate_only_report(report)

    def test_safety_flag_shape_is_exact(self) -> None:
        report = fake_report()
        unsafe = copy.deepcopy(report)
        del unsafe["safety"]["corpus_modified"]
        with self.assertRaisesRegex(ValueError, "invalid safety"):
            assert_aggregate_only_report(unsafe)


if __name__ == "__main__":
    unittest.main()
