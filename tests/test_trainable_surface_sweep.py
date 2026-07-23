import copy
import json
import unittest
from pathlib import Path

from slaif_asr.config import REPO_ROOT
from slaif_asr.trainable_surface_sweep import (
    PR36_METRICS,
    SURFACE_ID,
    assert_public_report_safe,
    bind_post_selection_metrics,
    classify_surface04,
    component_or_not_recorded,
    load_config,
    mark_controller_selection,
    should_stop_controller_curve,
    validate_config,
)


class TrainableSurfaceSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.adr_text = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(encoding="utf-8")

    def test_adr_0009_is_required_and_names_surface04(self):
        validate_config(self.config, adr_text=self.adr_text)
        with self.assertRaises(ValueError):
            validate_config(self.config, adr_text="ADR without authorization")

    def test_phase1_rejects_other_encoder_depths_and_full_encoder(self):
        for index, surface_id in ((22, "SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS"), (20, "SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS")):
            bad = copy.deepcopy(self.config)
            bad["trainable_surface"]["surface_id"] = surface_id
            bad["trainable_surface"]["final_encoder_layer_index"] = index
            with self.assertRaises(ValueError):
                validate_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["trainable_surface"]["full_encoder_allowed"] = True
        with self.assertRaises(ValueError):
            validate_config(bad, adr_text=self.adr_text)

    def test_fixed_data_rejects_s6tts_scale8000_and_schedule_drift(self):
        for key, value in (("source", "s6tts-hardvoice"), ("source", "scale8000-clean")):
            bad = copy.deepcopy(self.config)
            bad["data"][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["data"]["exposure_schedule_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_config(bad, adr_text=self.adr_text)

    def test_encoder_lr_is_lower_than_decoder_and_joint(self):
        rates = self.config["training"]["learning_rates"]
        self.assertLess(rates["final_encoder_block"], rates["decoder"])
        self.assertEqual(rates["decoder"], rates["joint"])

    def test_agents_records_narrow_encoder_exception(self):
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("Work Order 0037 provide one bounded exception", text)
        self.assertIn("final encoder block", text)
        self.assertIn("not authorize lower blocks or full-encoder training", text)

    def test_controller_stop_after_three_rounds_without_new_raw_best(self):
        rows = [
            {"round": 0, "wer": 80.0, "empty": 2},
            {"round": 1, "wer": 60.0, "empty": 0},
            {"round": 2, "wer": 60.2, "empty": 0},
            {"round": 3, "wer": 60.1, "empty": 0},
            {"round": 4, "wer": 60.3, "empty": 0},
        ]
        result = should_stop_controller_curve(rows)
        self.assertTrue(result["stop"])
        self.assertEqual(result["best_round"], 1)

    def test_controller_hard_stops_when_empty_hypotheses_reappear(self):
        rows = [
            {"round": 0, "wer": 80.0, "empty": 2},
            {"round": 1, "wer": 60.0, "empty": 0},
            {"round": 2, "wer": 59.0, "empty": 1},
            {"round": 3, "wer": 58.0, "empty": 0},
        ]
        self.assertEqual(should_stop_controller_curve(rows)["reason"], "empty_hypotheses_reappeared")

    def test_controller_selection_marker_matches_primary_wer_rule(self):
        rows = [
            {"round": 0, "available": True, "wer": 66.467, "cer": 27.409, "empty": 13},
            {"round": 3, "available": True, "wer": 53.182, "cer": 19.037, "empty": 0},
            {"round": 4, "available": True, "wer": 53.353, "cer": 17.760, "empty": 0},
        ]
        marked = mark_controller_selection(rows, base_empty_count=13)
        self.assertEqual(marked["selected_round"], 3)
        by_round = {row["round"]: row for row in marked["rows"]}
        self.assertTrue(by_round[3]["eligible"])
        self.assertTrue(by_round[3]["selected_by_rule"])
        self.assertTrue(by_round[4]["eligible"])

    def test_surface04_classification_boundaries(self):
        beats = {split: dict(values) for split, values in PR36_METRICS.items()}
        for values in beats.values():
            values["wer"] -= 0.1
            values["cer"] -= 0.1
        self.assertEqual(classify_surface04(beats, selected_round=2), "SURFACE04_BEATS_PR36_DIRECTIONAL")
        beats["artur_j"]["wer"] = 70.0
        self.assertEqual(
            classify_surface04(beats, selected_round=2),
            "SURFACE04_ARTUR_DEV_GOOD_BUT_GATE_DIRECTIONAL_REGRESSES",
        )

    def test_post_selection_metrics_cannot_change_selected_round(self):
        result = bind_post_selection_metrics(4, {"fleurs_v2": {"wer": 99.0}})
        self.assertEqual(result["selected_round"], 4)

    def test_edit_components_are_not_fabricated(self):
        self.assertEqual(component_or_not_recorded({}, "delete"), "NOT_RECORDED")
        self.assertEqual(component_or_not_recorded({"delete": 0}, "delete"), 0)

    def test_public_report_rejects_raw_fields_and_local_paths(self):
        assert_public_report_safe({"surface_id": SURFACE_ID, "selected_round": 4})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"reference": "forbidden"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"note": "/data-nvme/private"})

    def test_config_contains_no_s6tts_or_scale8000(self):
        serialized = json.dumps(self.config).lower()
        self.assertNotIn("s6tts", serialized)
        self.assertNotIn("scale8000", serialized)


if __name__ == "__main__":
    unittest.main()
