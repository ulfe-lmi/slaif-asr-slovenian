from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slaif_asr.real_eval import atomic_write_json
from slaif_asr.supertonic3_tts import (
    HELD_OUT_STYLES,
    TRAINING_STYLES,
    _is_oom_error,
    _synthesize_batched_with_fallback,
    _style_batch,
    batch_identity_sha256,
    build_batched_variant_plan,
    deterministic_batch_seed,
    load_supertonic_config,
    partition_batched_plan,
)
from tests.test_supertonic3_tts import Supertonic3TtsTests


class FakeStyle:
    def __init__(self, value: float) -> None:
        self.ttl = [[value, value]]
        self.dp = [[value + 10.0, value + 10.0, value + 10.0]]


def first_column(value: object) -> list[float]:
    rows = value.tolist() if hasattr(value, "tolist") else value
    return [float(row[0]) for row in rows]  # type: ignore[index]


class Supertonic3BatchedSynthesisTests(unittest.TestCase):
    def test_replay_config_id_uses_same_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = Supertonic3TtsTests().fixture_config(tmp, selected_count=1, holdout_count=1)
            config["tts_id"] = "supertonic3-sl-multivoice-batched-replay-v1"
            config["batch_synthesis"] = {"batch_size": 32, "oom_fallback_batch_size": 16, "seed": 1234}
            config["runtime"] = {"execution_device": "cuda", "required_provider": "CUDAExecutionProvider", "cpu_provider_fallback_allowed": False, "cuda_visible_devices": "1"}
            path = tmp / "tts.json"
            atomic_write_json(path, config)
            loaded = load_supertonic_config(path)
            self.assertEqual(loaded["tts_id"], "supertonic3-sl-multivoice-batched-replay-v1")
            self.assertEqual(loaded["model"]["revision"], "724fb5abbf5502583fb520898d45929e62f02c0b")

    def test_batched_plan_contains_all_voice_pairs_and_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = Supertonic3TtsTests().fixture_config(tmp, selected_count=2, holdout_count=2)
            plan = build_batched_variant_plan(config)
            self.assertEqual(len(plan), 2 * len(TRAINING_STYLES) + 2 * len(HELD_OUT_STYLES))
            self.assertEqual({row.voice_style for row in plan if row.partition_stage == "training"}, set(TRAINING_STYLES))
            self.assertEqual({row.voice_style for row in plan if row.partition_stage == "holdout"}, set(HELD_OUT_STYLES))
            sort_keys = [
                (row.preprocessed_text_length, row.item.partition_role, row.source_key_hash, row.voice_style)
                for row in plan
            ]
            self.assertEqual(sort_keys, sorted(sort_keys))

    def test_partition_batched_plan_and_seed_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            plan = build_batched_variant_plan(Supertonic3TtsTests().fixture_config(Path(tmp_text), selected_count=5, holdout_count=3))
            batches = partition_batched_plan(plan, 7)
            self.assertEqual(sum(len(batch) for batch in batches), len(plan))
            self.assertLessEqual(max(len(batch) for batch in batches), 7)
            first_sha = batch_identity_sha256(batches[0])
            self.assertEqual(first_sha, batch_identity_sha256(list(batches[0])))
            self.assertEqual(
                deterministic_batch_seed(experiment_seed=1234, batch_index=1, identity_sha256=first_sha),
                deterministic_batch_seed(experiment_seed=1234, batch_index=1, identity_sha256=first_sha),
            )
            self.assertNotEqual(
                deterministic_batch_seed(experiment_seed=1234, batch_index=1, identity_sha256=first_sha),
                deterministic_batch_seed(experiment_seed=1234, batch_index=2, identity_sha256=first_sha),
            )

    def test_style_batch_concatenates_ttl_and_duration_predictor_arrays(self) -> None:
        styles = {"A": FakeStyle(1.0), "B": FakeStyle(2.0), "C": FakeStyle(3.0)}
        batched = _style_batch(styles, ["A", "C", "B"])
        self.assertEqual(batched.ttl.shape, (3, 2))
        self.assertEqual(batched.dp.shape, (3, 3))
        self.assertEqual(first_column(batched.ttl), [1.0, 3.0, 2.0])
        self.assertEqual(first_column(batched.dp), [11.0, 13.0, 12.0])

    def test_oom_detector_is_narrow(self) -> None:
        self.assertTrue(_is_oom_error(RuntimeError("CUDA out of memory while allocating tensor")))
        self.assertTrue(_is_oom_error(RuntimeError("BFCArena::AllocateRawInternal failed to allocate memory for requested buffer")))
        self.assertFalse(_is_oom_error(RuntimeError("invalid voice style")))

    def test_oom_fallback_recurses_to_smaller_configured_batches(self) -> None:
        import slaif_asr.supertonic3_tts as module

        original = module._synthesize_batched_core
        calls: list[int] = []

        def fake_core(**kwargs):  # type: ignore[no-untyped-def]
            batch = kwargs["batch"]
            calls.append(len(batch))
            if len(batch) > 4:
                raise RuntimeError("CUDA out of memory")
            return (
                [{"row": index} for index, _ in enumerate(batch)],
                {
                    "batch_index": kwargs["batch_index"],
                    "requested_size": len(batch),
                    "actual_size": len(batch),
                    "oom_fallback": False,
                },
            )

        try:
            module._synthesize_batched_core = fake_core  # type: ignore[assignment]
            rows, summaries = _synthesize_batched_with_fallback(
                config={},
                assets={},
                provider_summary={},
                paths=object(),
                tts=object(),
                styles={},
                batch=list(range(16)),  # type: ignore[arg-type]
                batch_index=1,
                experiment_seed=1234,
                fallback_sizes=[8, 4, 2, 1],
            )
        finally:
            module._synthesize_batched_core = original  # type: ignore[assignment]

        self.assertEqual(len(rows), 16)
        self.assertEqual([summary["actual_size"] for summary in summaries], [4, 4, 4, 4])
        self.assertTrue(all(summary["oom_fallback"] for summary in summaries))
        self.assertIn(16, calls)
        self.assertIn(8, calls)
        self.assertEqual(calls.count(4), 4)

    def test_oom_fallback_stops_when_no_smaller_configured_batch_exists(self) -> None:
        import slaif_asr.supertonic3_tts as module

        original = module._synthesize_batched_core

        def fake_core(**kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("CUDA out of memory")

        try:
            module._synthesize_batched_core = fake_core  # type: ignore[assignment]
            with self.assertRaisesRegex(RuntimeError, "no smaller fallback"):
                _synthesize_batched_with_fallback(
                    config={},
                    assets={},
                    provider_summary={},
                    paths=object(),
                    tts=object(),
                    styles={},
                    batch=list(range(16)),  # type: ignore[arg-type]
                    batch_index=1,
                    experiment_seed=1234,
                    fallback_sizes=[],
                )
        finally:
            module._synthesize_batched_core = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
