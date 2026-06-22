from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from slaif_asr.metrics import CorpusMetricSummary
from slaif_asr.slovenian_curriculum import (
    CurriculumRecord,
    assert_training_disjoint,
    classify_round1,
    near_duplicate,
    select_controls,
    select_hard_examples,
    validate_collection,
    validate_record,
)


def config() -> dict:
    return {
        "validation": {
            "min_words": 4,
            "max_words": 20,
            "min_characters": 10,
            "max_characters": 160,
            "near_duplicate_char_ngram": 5,
            "near_duplicate_jaccard_threshold": 0.82,
            "max_same_first_two_words": 5,
            "max_same_final_three_words": 5,
            "forbidden_substrings": ["http://", "https://", "@", "```", "€", "°"],
        },
        "promotion": {
            "synthetic_holdout_relative_wer_or_cer_improvement_percent": 15.0,
            "fleurs_max_absolute_wer_regression": 1.0,
            "artur_j_max_absolute_wer_regression": 1.0,
            "fleurs_max_absolute_cer_regression": 1.5,
            "artur_j_max_absolute_cer_regression": 1.5,
            "empty_hypotheses_must_not_increase": True,
            "real_improvement_wer_abs": 0.5,
            "real_improvement_cer_abs": 0.75,
        },
    }


def record(index: int, *, role: str = "synthetic_candidate", phenomenon: str = "ordinary", text: str | None = None) -> CurriculumRecord:
    text = text or f"Mirna vasica danes posluša jasen stavek {index}."
    return CurriculumRecord(
        schema_version="1.0",
        candidate_id=f"round1-{index:04d}",
        spoken_text=text,
        target_text=text,
        language="sl-SI",
        partition_role=role,
        phenomena=(phenomenon,),
        generation={
            "system": "project-generated",
            "method": "direct-language-generation",
            "round": 1,
            "prompt_revision": "sl-curriculum-round1-v1",
            "model_identity": "not-exposed-by-execution-runtime",
            "seed": None,
        },
    )


def row(text: str = "Miren stavek opisuje jutranji sprehod.") -> dict:
    item = record(1, text=text)
    return {
        "schema_version": item.schema_version,
        "candidate_id": item.candidate_id,
        "spoken_text": item.spoken_text,
        "target_text": item.target_text,
        "language": item.language,
        "partition_role": item.partition_role,
        "phenomena": list(item.phenomena),
        "generation": item.generation,
    }


def summary(wer: float, cer: float, empty: int = 0) -> CorpusMetricSummary:
    return CorpusMetricSummary(
        corpus_wer=wer,
        corpus_cer=cer,
        mean_utterance_wer=wer,
        mean_utterance_cer=cer,
        median_utterance_wer=wer,
        median_utterance_cer=cer,
        empty_hypothesis_count=empty,
        total_word_edits=0,
        total_reference_words=0,
        total_character_edits=0,
        total_reference_characters=0,
    )


class SlovenianCurriculumTests(unittest.TestCase):
    def test_valid_record(self) -> None:
        item = validate_record(row(), expected_role="synthetic_candidate", config=config())
        self.assertEqual(item.language, "sl-SI")
        self.assertEqual(item.generation["system"], "project-generated")

    def test_unsupported_symbol_rejected(self) -> None:
        bad = row("Temperatura je danes dvajset °C v senci.")
        with self.assertRaises(ValueError):
            validate_record(bad, expected_role="synthetic_candidate", config=config())

    def test_near_duplicate_rejected(self) -> None:
        records = [
            record(1, text="Jasna poved opisuje miren večerni sprehod."),
            record(2, text="Jasna poved opisuje miren večerni sprehod!"),
        ]
        self.assertTrue(near_duplicate(records[0].target_text, records[1].target_text, n=5, threshold=0.82))
        with self.assertRaises(ValueError):
            validate_collection(records, expected_count=2, config=config())

    def test_protected_hash_overlap_rejected(self) -> None:
        item = record(1)
        from slaif_asr.real_eval import stable_text_hash

        with self.assertRaises(ValueError):
            validate_collection([item], expected_count=1, config=config(), protected_hashes={stable_text_hash(item.target_text)})

    def test_holdout_never_enters_training(self) -> None:
        with self.assertRaises(ValueError):
            assert_training_disjoint({"holdout-0001"}, holdout_ids={"holdout-0001"})

    def test_hard_selection_and_controls(self) -> None:
        scored = []
        for index in range(20):
            scored.append(
                {
                    "candidate_id": f"round1-{index:04d}",
                    "empty_hypothesis": index < 3,
                    "normalized_cer": 90 - index,
                    "normalized_wer": 80 - index,
                    "word_deletions": index,
                    "phenomena": ["ordinary" if index < 10 else "questions_requests"],
                }
            )
        hard = select_hard_examples(scored, count=6, category_cap=3)
        self.assertEqual(len(hard), 6)
        self.assertTrue(all(row["candidate_id"] not in {item["candidate_id"] for item in hard} for row in select_controls(scored, exclude_ids={item["candidate_id"] for item in hard}, count=4, seed=1234)))

    def test_round1_real_generalization_classification(self) -> None:
        decision = classify_round1(
            integrity_passed=True,
            synthetic_holdout_base=summary(80, 40),
            synthetic_holdout_challenger=summary(60, 30),
            fleurs_base=summary(52.734, 16.423),
            fleurs_challenger=summary(52.0, 15.6),
            artur_base=summary(67.453, 29.016, empty=12),
            artur_challenger=summary(67.0, 28.8, empty=12),
            thresholds=config()["promotion"],
        )
        self.assertEqual(decision.decision, "ROUND1_ACCEPTED_REAL_GENERALIZATION")

    def test_round1_rejected_on_real_regression(self) -> None:
        decision = classify_round1(
            integrity_passed=True,
            synthetic_holdout_base=summary(80, 40),
            synthetic_holdout_challenger=summary(60, 30),
            fleurs_base=summary(52.734, 16.423),
            fleurs_challenger=summary(54.0, 18.1),
            artur_base=summary(67.453, 29.016, empty=12),
            artur_challenger=summary(67.0, 28.8, empty=12),
            thresholds=config()["promotion"],
        )
        self.assertEqual(decision.decision, "ROUND1_REJECTED")

    def test_integrity_failure_invalidates(self) -> None:
        decision = classify_round1(
            integrity_passed=False,
            synthetic_holdout_base=summary(80, 40),
            synthetic_holdout_challenger=summary(60, 30),
            fleurs_base=summary(52, 16),
            fleurs_challenger=summary(52, 16),
            artur_base=summary(67, 29),
            artur_challenger=summary(67, 29),
            thresholds=config()["promotion"],
        )
        self.assertEqual(decision.decision, "EXPERIMENT_INVALID")


if __name__ == "__main__":
    unittest.main()
