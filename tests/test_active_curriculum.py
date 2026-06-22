from __future__ import annotations

import unittest

from slaif_asr.active_curriculum import (
    assert_training_ids_are_disjoint,
    build_round2_failure_brief,
    decide_promotion,
    deterministic_active_selection,
    deterministic_controls,
    score_candidate,
)


class ActiveCurriculumTests(unittest.TestCase):
    def scored(self):
        return [
            score_candidate(candidate_id="round1-0001", reference="ena dva", hypothesis="", phenomena=("ordinary",)),
            score_candidate(candidate_id="round1-0002", reference="č š ž", hypothesis="č š", phenomena=("diacritic",)),
            score_candidate(candidate_id="round1-0003", reference="Ljubljana danes", hypothesis="Ljubljana danes", phenomena=("place",)),
        ]

    def test_deterministic_active_ranking_prioritizes_empty_then_errors(self) -> None:
        selected = deterministic_active_selection(self.scored(), hard_count=2)
        self.assertEqual([item.candidate_id for item in selected], ["round1-0001", "round1-0002"])

    def test_phenomenon_quota_is_deterministic(self) -> None:
        selected = deterministic_active_selection(self.scored(), hard_count=1, phenomenon_quota={"place": 1})
        self.assertEqual(selected[0].candidate_id, "round1-0003")

    def test_general_controls_are_seeded_and_exclude_hard_examples(self) -> None:
        controls = deterministic_controls(self.scored(), exclude_ids={"round1-0001"}, count=2, seed=1234)
        self.assertNotIn("round1-0001", [item.candidate_id for item in controls])

    def test_holdout_and_real_gate_cannot_enter_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "synthetic holdout"):
            assert_training_ids_are_disjoint(
                training_ids={"a", "b"},
                synthetic_holdout_ids={"b"},
                real_gate_ids=set(),
            )
        with self.assertRaisesRegex(ValueError, "real gate"):
            assert_training_ids_are_disjoint(
                training_ids={"a"},
                synthetic_holdout_ids=set(),
                real_gate_ids={"a"},
            )

    def test_round2_brief_excludes_real_references(self) -> None:
        brief = build_round2_failure_brief(self.scored())
        self.assertFalse(brief["real_gate_reference_text_included"])
        self.assertFalse(brief["synthetic_holdout_errors_included"])
        self.assertEqual(brief["source"], "synthetic_candidate_pool_only")

    def test_promotion_accepts_only_when_synthetic_improves_and_real_does_not_regress(self) -> None:
        decision = decide_promotion(
            integrity_passed=True,
            synthetic_base_rows=[("ena dva", ""), ("tri štiri", "tri")],
            synthetic_challenger_rows=[("ena dva", "ena dva"), ("tri štiri", "tri štiri")],
            real_base_rows=[("realen stavek", "realen stavek")],
            real_challenger_rows=[("realen stavek", "realen stavek")],
        )
        self.assertTrue(decision.promoted)

    def test_promotion_rolls_back_on_real_regression(self) -> None:
        decision = decide_promotion(
            integrity_passed=True,
            synthetic_base_rows=[("ena dva", "")],
            synthetic_challenger_rows=[("ena dva", "ena dva")],
            real_base_rows=[("realen stavek", "realen stavek")],
            real_challenger_rows=[("realen stavek", "")],
        )
        self.assertFalse(decision.promoted)
        self.assertTrue(any("real-gate" in reason for reason in decision.reasons))


if __name__ == "__main__":
    unittest.main()
