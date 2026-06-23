from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.real_eval import (
    FleursOccurrencePlan,
    NORMALIZER_VERSION,
    ArturSegment,
    ensure_no_references_or_paths,
    plan_fleurs_occurrences,
    normalize_sl_asr_text,
    parse_artur_trs,
    reject_real_gate_for_generation,
    reject_real_gate_for_training,
    select_artur_segments,
    summarize_predictions,
    validate_fleurs_occurrence_plan,
)


class RealEvalTests(unittest.TestCase):
    def test_normalizer_preserves_slovenian_letters(self) -> None:
        self.assertEqual(normalize_sl_asr_text("ČŠŽ, test-ena  12!"), "čšž test ena 12")
        self.assertEqual(NORMALIZER_VERSION, "sl-asr-normalization-v1")

    def test_raw_and_normalized_metrics_are_separate(self) -> None:
        summary = summarize_predictions([{"reference": "Čez cesto.", "hypothesis": "čez cesto"}])
        self.assertGreater(summary["raw"]["corpus_wer"], 0)
        self.assertEqual(summary["normalized"]["corpus_wer"], 0)
        self.assertIn("mean_utterance_wer", summary["raw"])

    def test_empty_hypothesis_counted(self) -> None:
        summary = summarize_predictions([{"reference": "ena dva", "hypothesis": ""}])
        self.assertEqual(summary["raw"]["empty_hypothesis_count"], 1)

    def test_metadata_rejects_reference_and_local_path(self) -> None:
        with self.assertRaises(ValueError):
            ensure_no_references_or_paths({"selected": [{"reference": "secret"}]})
        with self.assertRaises(ValueError):
            ensure_no_references_or_paths({"selected": [{"raw_reference": "secret"}]})
        with self.assertRaises(ValueError):
            ensure_no_references_or_paths({"path": "/home/user/audio.wav"})

    def test_artur_parser_accepts_valid_std(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Artur-J-Splosni" / "std" / "recording.trs"
            path.parent.mkdir(parents=True)
            path.write_text(
                """<?xml version='1.0' encoding='UTF-8'?>
<Trans><Episode><Section><Turn startTime="0" endTime="4">
<Sync time="0.0"/>Prvi stavek.
<Sync time="2.0"/>Drugi stavek.
<Sync time="4.0"/>
</Turn></Section></Episode></Trans>
""",
                encoding="utf-8",
            )
            segments = parse_artur_trs(path)
            self.assertEqual([item.text for item in segments], ["Prvi stavek.", "Drugi stavek."])

    def test_artur_parser_rejects_pog_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pog = Path(temp) / "Artur-J-Splosni" / "pog" / "recording.trs"
            pog.parent.mkdir(parents=True)
            pog.write_text("<Trans />", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_artur_trs(pog)

            bad = Path(temp) / "Artur-J-Splosni" / "std" / "bad.trs"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_text(
                """<Trans><Episode><Section><Turn><Sync time="2.0"/>A<Sync time="1.0"/></Turn></Section></Episode></Trans>""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_artur_trs(bad)

    def test_artur_selection_enforces_per_recording_cap(self) -> None:
        segments = [
            ArturSegment(f"id-{index:03d}", "rec-a", float(index * 3), float(index * 3 + 2), "text", "x")
            for index in range(10)
        ]
        segments.extend(
            ArturSegment(f"id-b-{index:03d}", "rec-b", float(index * 3), float(index * 3 + 2), "text", "x")
            for index in range(10)
        )
        selected = select_artur_segments(segments, required_count=6, duration_min=1, duration_max=3, max_per_recording=4)
        counts = {}
        for item in selected:
            counts[item.recording_id] = counts.get(item.recording_id, 0) + 1
        self.assertLessEqual(max(counts.values()), 4)

    def test_fleurs_occurrence_identity_survives_repeated_source_id(self) -> None:
        rows = [
            {"id": 1738, "transcription": "Prvi primer."},
            {"id": 1738, "transcription": "Drugi primer."},
        ]
        plans = plan_fleurs_occurrences(rows)
        self.assertEqual([item.source_id for item in plans], [1738, 1738])
        self.assertEqual(
            [item.sample_id for item in plans],
            ["fleurs-sl-si-test-occ-00000", "fleurs-sl-si-test-occ-00001"],
        )
        self.assertEqual(
            [item.relative_audio_path for item in plans],
            ["audio/fleurs-sl-si-test-occ-00000.wav", "audio/fleurs-sl-si-test-occ-00001.wav"],
        )
        self.assertEqual(len({item.sample_id for item in plans}), 2)
        self.assertEqual(len({item.relative_audio_path for item in plans}), 2)

    def test_fleurs_occurrence_plan_rejects_collisions(self) -> None:
        duplicate_index = [
            FleursOccurrencePlan(0, 10, "fleurs-sl-si-test-occ-00000", "audio/fleurs-sl-si-test-occ-00000.wav"),
            FleursOccurrencePlan(0, 11, "fleurs-sl-si-test-occ-00001", "audio/fleurs-sl-si-test-occ-00001.wav"),
        ]
        with self.assertRaises(ValueError):
            validate_fleurs_occurrence_plan(duplicate_index)

        duplicate_sample_id = [
            FleursOccurrencePlan(0, 10, "fleurs-sl-si-test-occ-00000", "audio/a.wav"),
            FleursOccurrencePlan(1, 11, "fleurs-sl-si-test-occ-00000", "audio/b.wav"),
        ]
        with self.assertRaises(ValueError):
            validate_fleurs_occurrence_plan(duplicate_sample_id)

        duplicate_audio_path = [
            FleursOccurrencePlan(0, 10, "fleurs-sl-si-test-occ-00000", "audio/same.wav"),
            FleursOccurrencePlan(1, 11, "fleurs-sl-si-test-occ-00001", "audio/same.wav"),
        ]
        with self.assertRaises(ValueError):
            validate_fleurs_occurrence_plan(duplicate_audio_path)

    def test_gate_config_is_pinned(self) -> None:
        config = json.loads(Path("configs/evaluation/real_gates.json").read_text(encoding="utf-8"))
        self.assertEqual(config["fleurs_sl_si_test_full_v1"]["revision"], "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
        self.assertTrue(config["fleurs_sl_si_test_full_v1"]["use_complete_split"])
        self.assertEqual(config["fleurs_sl_si_test_full_v2"]["revision"], "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
        self.assertEqual(config["fleurs_sl_si_test_full_v2"]["gate_id"], "fleurs-sl-si-test-full-v2")
        self.assertTrue(config["fleurs_sl_si_test_full_v2"]["use_complete_split"])
        self.assertEqual(config["artur_j_public_gate_v1"]["transcript_archive"]["md5"], "6f21947593ccdea7dc23ecc3c9a7c012")
        self.assertEqual(config["artur_j_public_gate_v1"]["audio_archives"][0]["md5"], "bc8b4e0625fce2b47d99ed7da8db7393")

    def test_real_gates_cannot_enter_training_or_generation(self) -> None:
        rows = [{"partition_role": "immutable_real_gate", "source_type": "public_real", "text": "skrito"}]
        with self.assertRaises(ValueError):
            reject_real_gate_for_training(rows)
        with self.assertRaises(ValueError):
            reject_real_gate_for_generation(rows)


if __name__ == "__main__":
    unittest.main()
