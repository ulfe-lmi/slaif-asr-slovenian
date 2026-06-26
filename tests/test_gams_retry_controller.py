from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slaif_asr.gams_retry_controller import (
    AttemptRecord,
    RetryLimits,
    RetryState,
    exhausted_budgets,
    initial_tasks,
    load_state,
    plan_refill_tasks,
    save_state,
    targeted_attempt_count,
    validate_diversity_guidance,
)


class GamsRetryControllerTests(unittest.TestCase):
    def test_initial_shard_count_and_deterministic_seeds(self) -> None:
        first = initial_tasks(["cell02", "cell01"])
        second = initial_tasks(["cell01", "cell02"])
        self.assertEqual(len(first), 18)
        self.assertEqual([task.to_json() for task in first], [task.to_json() for task in second])
        self.assertEqual(first[0].shard_id, "shard01")
        self.assertEqual(first[0].requested_rows, 60)

    def test_targeted_attempt_count(self) -> None:
        self.assertEqual(targeted_attempt_count(0), 0)
        self.assertEqual(targeted_attempt_count(1), 1)
        self.assertEqual(targeted_attempt_count(40), 1)
        self.assertEqual(targeted_attempt_count(41), 2)
        self.assertEqual(targeted_attempt_count(999), 5)

    def test_resume_does_not_repeat_attempts(self) -> None:
        state = RetryState()
        task = initial_tasks(["cell01"])[0]
        state.record(AttemptRecord(task=task, status="completed", parsed_rows=55, retained_rows=44))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            save_state(path, state)
            loaded = load_state(path)
        self.assertIn(task.attempt_id, loaded.completed_attempt_ids)
        self.assertEqual(loaded.next_attempt_index("cell01", "shard01"), 1)

    def test_refill_targets_only_deficient_cells(self) -> None:
        tasks = plan_refill_tasks({"cell02": 85}, verification_round=1, state=RetryState())
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(task.cell_id == "cell02" for task in tasks))
        self.assertTrue(all(task.reason == "targeted_refill" for task in tasks))

    def test_budget_exhaustion(self) -> None:
        limits = RetryLimits(max_attempts_per_cell=1, max_attempts_per_shard=2, max_total_attempts=10, max_requested_rows=600)
        state = RetryState()
        task = initial_tasks(["cell01"], limits=limits)[0]
        state.record(AttemptRecord(task=task, status="completed"))
        self.assertIn("cell:cell01", exhausted_budgets(state, limits))
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            plan_refill_tasks({"cell01": 360}, verification_round=1, state=state, limits=limits)

    def test_unbounded_limits_do_not_stop_refills(self) -> None:
        limits = RetryLimits(
            max_verification_rounds=None,
            max_attempts_per_cell=None,
            max_attempts_per_shard=None,
            max_total_attempts=None,
            max_requested_rows=None,
        )
        state = RetryState()
        for index in range(60):
            task = initial_tasks(["cell01"])[0]
            task = type(task)("cell01", "shard01", index, 0, 60, index, "fixture")
            state.record(AttemptRecord(task=task, status="completed"))
        self.assertEqual(exhausted_budgets(state, limits), [])
        tasks = plan_refill_tasks({"cell01": 360}, verification_round=99, state=state, limits=limits)
        self.assertEqual(len(tasks), 5)

    def test_guidance_rejects_raw_identifiers_and_protected_names(self) -> None:
        self.assertEqual(validate_diversity_guidance(["vary clause structure"]), ("vary clause structure",))
        with self.assertRaisesRegex(ValueError, "retry guidance"):
            validate_diversity_guidance(["avoid gamsv4-cell01-a00-o001"])
        with self.assertRaisesRegex(ValueError, "retry guidance"):
            validate_diversity_guidance(["copy no FLEURS references"])


if __name__ == "__main__":
    unittest.main()
