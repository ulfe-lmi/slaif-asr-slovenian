import copy
import json
import unittest

from slaif_asr.config import REPO_ROOT
from slaif_asr.trainable_surface_sweep import (
    SURFACE05_METRICS,
    SURFACE06_BEST_REAL_GATE_ENVELOPE,
    SURFACE06_ALLOWED_TRAINABLE_PREFIXES,
    assert_public_report_safe,
    classify_surface06,
    configure_surface06_trainable,
    load_surface06_config,
    microbatch_plan,
    select_surface06_microbatch,
    surface06_changed_tensor_summary,
    surface06_envelope_comparison,
    surface06_optimizer_parameter_groups,
    validate_surface06_config,
    verify_surface06_optimizer_scope,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.shape = ()

    def __eq__(self, other):
        return FakeTensor(self.value == other.value)

    def all(self):
        return bool(self.value)


class FakeParam:
    def __init__(self, count):
        self._count = count
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self._count


class FakeModel:
    def __init__(self):
        self.params = {
            "preprocessor.featurizer.window": FakeParam(1),
            "encoder.pre_encode.conv.weight": FakeParam(2),
            "prompt_kernel.0.weight": FakeParam(7),
            "decoder.prediction.weight": FakeParam(8),
            "joint.joint_net.weight": FakeParam(9),
        }
        for index in range(24):
            self.params[f"encoder.layers.{index}.self_attn.weight"] = FakeParam(index + 1)

    def parameters(self):
        return list(self.params.values())

    def named_parameters(self):
        return list(self.params.items())

    def named_modules(self):
        return [("", self), ("encoder", object()), ("decoder", object()), ("joint", object())]


class FakeOptimizer:
    def __init__(self, groups):
        self.param_groups = groups


class Surface06LastFourEncoderBlocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_surface06_config()
        cls.adr_text = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )

    def setUp(self):
        self.model = FakeModel()
        self.summary = configure_surface06_trainable(self.model)

    def test_selects_exactly_decoder_joint_and_final_four_encoder_blocks(self):
        self.assertEqual(self.summary.decoder_parameter_count, 8)
        self.assertEqual(self.summary.joint_parameter_count, 9)
        self.assertEqual(self.summary.final_four_encoder_blocks_parameter_count, 90)
        self.assertEqual(
            self.summary.final_encoder_blocks,
            ("encoder.layers.20", "encoder.layers.21", "encoder.layers.22", "encoder.layers.23"),
        )
        for name, parameter in self.model.named_parameters():
            expected = name.startswith(SURFACE06_ALLOWED_TRAINABLE_PREFIXES)
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_lower_encoder_preprocessor_frontend_and_prompt_are_frozen(self):
        for prefix in ("preprocessor.", "encoder.pre_encode.", "encoder.layers.0.", "encoder.layers.19.", "prompt_kernel."):
            self.assertTrue(all(not parameter.requires_grad for name, parameter in self.model.named_parameters() if name.startswith(prefix)))

    def test_optimizer_scope_and_learning_rates(self):
        rates = {"decoder": 0.0005, "joint": 0.0005, "final_four_encoder_blocks": 0.00001}
        groups = surface06_optimizer_parameter_groups(self.model, rates)
        optimizer = FakeOptimizer(groups)
        verify_surface06_optimizer_scope(optimizer, self.model, rates)
        self.assertLess(groups[2]["lr"], groups[0]["lr"])

    def test_optimizer_rejects_unauthorized_parameter(self):
        rates = {"decoder": 0.0005, "joint": 0.0005, "final_four_encoder_blocks": 0.00001}
        groups = surface06_optimizer_parameter_groups(self.model, rates)
        groups[0]["params"].append(self.model.params["encoder.layers.19.self_attn.weight"])
        with self.assertRaises(RuntimeError):
            verify_surface06_optimizer_scope(FakeOptimizer(groups), self.model, rates)

    def test_parameter_integrity_flags_lower_encoder_change(self):
        before = {
            "decoder.x": FakeTensor(1),
            "joint.x": FakeTensor(1),
            "encoder.layers.20.x": FakeTensor(1),
            "encoder.layers.21.x": FakeTensor(1),
            "encoder.layers.22.x": FakeTensor(1),
            "encoder.layers.23.x": FakeTensor(1),
        }
        after = {
            "decoder.x": FakeTensor(2),
            "joint.x": FakeTensor(2),
            "encoder.layers.20.x": FakeTensor(2),
            "encoder.layers.21.x": FakeTensor(2),
            "encoder.layers.22.x": FakeTensor(2),
            "encoder.layers.23.x": FakeTensor(2),
        }
        self.assertTrue(surface06_changed_tensor_summary(before, after)["only_surface06_changed"])
        before["encoder.layers.19.x"] = FakeTensor(1)
        after["encoder.layers.19.x"] = FakeTensor(2)
        self.assertFalse(surface06_changed_tensor_summary(before, after)["only_surface06_changed"])

    def test_microbatch_preserves_effective_batch_or_blocks(self):
        self.assertEqual(microbatch_plan(2)["gradient_accumulation_steps"], 4)
        selected = select_surface06_microbatch({4: {"status": "FAILED"}, 2: {"status": "PASSED"}})
        self.assertEqual(selected["effective_batch_size"], 8)
        self.assertEqual(selected["physical_microbatch"], 2)
        blocked = select_surface06_microbatch({4: {"status": "FAILED"}, 2: {"status": "FAILED"}, 1: {"status": "FAILED"}})
        self.assertEqual(blocked["status"], "BLOCKED_SURFACE06_OOM")

    def test_unresolved_encoder_surface_is_rejected(self):
        del self.model.params["encoder.layers.19.self_attn.weight"]
        with self.assertRaisesRegex(RuntimeError, "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED"):
            configure_surface06_trainable(self.model)

    def test_adr_phase3_and_exact_surface_are_required(self):
        validate_surface06_config(self.config, adr_text=self.adr_text)
        with self.assertRaises(ValueError):
            validate_surface06_config(self.config, adr_text="ADR 0009 without Phase 3")
        bad = copy.deepcopy(self.config)
        bad["trainable_surface"]["final_encoder_layer_indices"] = [18, 19, 20, 21, 22, 23]
        with self.assertRaises(ValueError):
            validate_surface06_config(bad, adr_text=self.adr_text)

    def test_surface06_must_start_from_untouched_base(self):
        bad = copy.deepcopy(self.config)
        bad["model"]["initialization"] = "surface05_checkpoint"
        with self.assertRaises(ValueError):
            validate_surface06_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["model"]["checkpoint_sha256"] = "5" * 64
        with self.assertRaises(ValueError):
            validate_surface06_config(bad, adr_text=self.adr_text)

    def test_fixed_data_rejects_forbidden_sources_and_schedule_drift(self):
        for source in ("s6tts", "scale8000", "database-extension-v1"):
            bad = copy.deepcopy(self.config)
            bad["data"]["source_override"] = source
            with self.assertRaises(ValueError):
                validate_surface06_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["data"]["exposure_schedule_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_surface06_config(bad, adr_text=self.adr_text)

    def test_config_rejects_text_only_and_temporary_lm_head(self):
        for key in ("text_only_objective_allowed", "temporary_lm_head_allowed"):
            bad = copy.deepcopy(self.config)
            bad["trainable_surface"][key] = True
            with self.assertRaises(ValueError):
                validate_surface06_config(bad, adr_text=self.adr_text)

    def test_classifier_uses_best_known_one_sided_envelope(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 30.0, "cer": 10.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 12.0, "cer": 4.0, "empty": 0},
            "fleurs_v2": {"wer": 46.0, "cer": 14.7, "empty": 0},
            "artur_j": {"wer": 53.0, "cer": 17.6, "empty": 0},
        }
        self.assertEqual(
            classify_surface06(metrics, selected_round=2),
            "SURFACE06_NEW_BEST_DIRECTIONAL_CANDIDATE",
        )
        rows = surface06_envelope_comparison(metrics)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["within_tolerance"] for row in rows))
        self.assertEqual(SURFACE06_BEST_REAL_GATE_ENVELOPE["artur_j"]["wer"]["source"], "Surface05")

    def test_classifier_matches_with_acceptable_tradeoff(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 40.0, "cer": 14.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 16.0, "cer": 4.8, "empty": 0},
            "fleurs_v2": {"wer": 46.3, "cer": 14.9, "empty": 0},
            "artur_j": {"wer": 53.4, "cer": 17.6, "empty": 0},
        }
        self.assertEqual(
            classify_surface06(metrics, selected_round=3),
            "SURFACE06_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF",
        )

    def test_classifier_rejects_empty_hypothesis_regression(self):
        metrics = {split: dict(values) for split, values in SURFACE05_METRICS.items()}
        metrics["fleurs_v2"]["empty"] = 1
        self.assertEqual(
            classify_surface06(metrics, selected_round=3),
            "SURFACE06_SYNTHETIC_OR_REAL_REGRESSION",
        )

    def test_classifier_names_artur_gain_with_fleurs_regression(self):
        metrics = {split: dict(values) for split, values in SURFACE05_METRICS.items()}
        metrics["fleurs_v2"]["wer"] = 46.696
        metrics["artur_j"]["wer"] = 53.0
        self.assertEqual(
            classify_surface06(metrics, selected_round=3),
            "SURFACE06_ARTUR_DEV_GOOD_BUT_FLEURS_REGRESSES",
        )

    def test_config_rejects_surface07_and_prompt_fusion(self):
        for key in ("surface07_allowed", "prompt_acoustic_fusion_allowed"):
            bad = copy.deepcopy(self.config)
            bad["trainable_surface"][key] = True
            with self.assertRaises(ValueError):
                validate_surface06_config(bad, adr_text=self.adr_text)

    def test_public_report_rejects_raw_fields_and_local_paths(self):
        assert_public_report_safe({"surface_id": self.config["trainable_surface"]["surface_id"]})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"hypothesis": "forbidden"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"note": "/data-nvme/private"})

    def test_config_contains_only_fixed_scale2000_data(self):
        serialized = json.dumps(self.config).lower()
        for marker in ("s6tts", "scale8000", "database-extension"):
            self.assertNotIn(marker, serialized)

    def test_text_only_or_temporary_head_is_rejected(self):
        self.model.named_modules = lambda: [("", self.model), ("decoder_lm_adapter", object())]
        with self.assertRaises(RuntimeError):
            configure_surface06_trainable(self.model)


if __name__ == "__main__":
    unittest.main()
