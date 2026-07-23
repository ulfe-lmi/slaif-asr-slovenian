import copy
import json
import unittest

from slaif_asr.config import REPO_ROOT
from slaif_asr.trainable_surface_sweep import (
    SURFACE06_METRICS,
    SURFACE07_ALLOWED_TRAINABLE_PREFIXES,
    SURFACE07_BEST_REAL_GATE_ENVELOPE,
    assert_public_report_safe,
    bind_post_selection_metrics,
    classify_surface07,
    component_or_not_recorded,
    configure_surface07_trainable,
    discover_surface07_fusion_bridge,
    load_surface07_config,
    microbatch_plan,
    select_surface07_microbatch,
    surface07_changed_tensor_summary,
    surface07_envelope_comparison,
    surface07_optimizer_parameter_groups,
    validate_surface07_config,
    verify_surface07_optimizer_scope,
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
            "prompt_kernel.0.weight": FakeParam(20),
            "prompt_kernel.0.bias": FakeParam(2),
            "prompt_kernel.2.weight": FakeParam(10),
            "prompt_kernel.2.bias": FakeParam(1),
            "decoder.prediction.weight": FakeParam(8),
            "joint.joint_net.weight": FakeParam(9),
        }
        for index in range(24):
            self.params[f"encoder.layers.{index}.self_attn.weight"] = FakeParam(index + 1)
        self.module_names = [
            "",
            "encoder",
            "decoder",
            "joint",
            "prompt_kernel",
            "prompt_kernel.0",
            "prompt_kernel.1",
            "prompt_kernel.2",
        ]

    def parameters(self):
        return list(self.params.values())

    def named_parameters(self):
        return list(self.params.items())

    def named_modules(self):
        return [(name, object()) for name in self.module_names]


class FakeOptimizer:
    def __init__(self, groups):
        self.param_groups = groups


class Surface07TopEncoderFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_surface07_config()
        cls.adr_text = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )

    def setUp(self):
        self.model = FakeModel()
        self.summary = configure_surface07_trainable(self.model)

    def test_selects_exact_surface_and_one_fusion_bridge(self):
        self.assertEqual(self.summary.decoder_parameter_count, 8)
        self.assertEqual(self.summary.joint_parameter_count, 9)
        self.assertEqual(self.summary.final_four_encoder_blocks_parameter_count, 90)
        self.assertEqual(self.summary.fusion_bridge_parameter_count, 33)
        self.assertEqual(self.summary.fusion_bridge_module, "prompt_kernel")
        self.assertEqual(
            self.summary.final_encoder_blocks,
            ("encoder.layers.20", "encoder.layers.21", "encoder.layers.22", "encoder.layers.23"),
        )
        for name, parameter in self.model.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name.startswith(SURFACE07_ALLOWED_TRAINABLE_PREFIXES),
                name,
            )

    def test_bridge_discovery_proves_post_concat_module(self):
        discovery = discover_surface07_fusion_bridge(self.model)
        self.assertEqual(discovery["status"], "PASSED")
        self.assertEqual(discovery["module_name"], "prompt_kernel")
        self.assertEqual(discovery["parameter_count"], 33)
        self.assertEqual(discovery["prompt_identity_storage"], "one_hot_config_not_parameter")
        included = [row["module"] for row in discovery["candidate_modules"] if row["included"]]
        self.assertEqual(included, ["prompt_kernel"])

    def test_unresolved_or_overlapping_bridge_blocks(self):
        missing = FakeModel()
        missing.module_names.remove("prompt_kernel")
        self.assertEqual(
            discover_surface07_fusion_bridge(missing)["status"],
            "BLOCKED_FUSION_BRIDGE_UNRESOLVED",
        )
        overlapping = FakeModel()
        overlapping.params["prompt_embedding.weight"] = FakeParam(4)
        overlapping.module_names.append("prompt_embedding")
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_FUSION_BRIDGE_UNRESOLVED"):
            configure_surface07_trainable(overlapping)

    def test_lower_encoder_frontend_and_prompt_identity_stay_frozen(self):
        for prefix in (
            "preprocessor.",
            "encoder.pre_encode.",
            "encoder.layers.0.",
            "encoder.layers.19.",
        ):
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for name, parameter in self.model.named_parameters()
                    if name.startswith(prefix)
                )
            )

    def test_optimizer_scope_and_learning_rates(self):
        rates = {
            "decoder": 0.0005,
            "joint": 0.0005,
            "final_four_encoder_blocks": 0.00001,
            "fusion_bridge": 0.00005,
        }
        groups = surface07_optimizer_parameter_groups(self.model, rates)
        verify_surface07_optimizer_scope(FakeOptimizer(groups), self.model, rates)
        by_name = {group["name"]: group["lr"] for group in groups}
        self.assertLess(by_name["final_four_encoder_blocks"], by_name["decoder"])
        self.assertLess(by_name["fusion_bridge"], by_name["decoder"])

    def test_optimizer_rejects_unauthorized_parameter(self):
        rates = {
            "decoder": 0.0005,
            "joint": 0.0005,
            "final_four_encoder_blocks": 0.00001,
            "fusion_bridge": 0.00005,
        }
        groups = surface07_optimizer_parameter_groups(self.model, rates)
        groups[0]["params"].append(self.model.params["encoder.layers.19.self_attn.weight"])
        with self.assertRaises(RuntimeError):
            verify_surface07_optimizer_scope(FakeOptimizer(groups), self.model, rates)

    def test_parameter_integrity_allows_only_surface07(self):
        before = {
            "decoder.x": FakeTensor(1),
            "joint.x": FakeTensor(1),
            "encoder.layers.20.x": FakeTensor(1),
            "encoder.layers.21.x": FakeTensor(1),
            "encoder.layers.22.x": FakeTensor(1),
            "encoder.layers.23.x": FakeTensor(1),
            "prompt_kernel.0.weight": FakeTensor(1),
        }
        after = {name: FakeTensor(2) for name in before}
        summary = surface07_changed_tensor_summary(before, after)
        self.assertTrue(summary["only_surface07_changed"])
        self.assertTrue(summary["fusion_bridge_changed"])
        before["encoder.layers.19.x"] = FakeTensor(1)
        after["encoder.layers.19.x"] = FakeTensor(2)
        self.assertFalse(surface07_changed_tensor_summary(before, after)["only_surface07_changed"])

    def test_parameter_integrity_rejects_nonselected_prompt_module(self):
        before = {"prompt_other.weight": FakeTensor(1)}
        after = {"prompt_other.weight": FakeTensor(2)}
        summary = surface07_changed_tensor_summary(before, after)
        self.assertFalse(summary["only_surface07_changed"])
        self.assertFalse(summary["non_selected_prompt_or_fusion_unchanged"])

    def test_microbatch_preserves_effective_batch_or_blocks(self):
        self.assertEqual(microbatch_plan(2)["gradient_accumulation_steps"], 4)
        selected = select_surface07_microbatch(
            {4: {"status": "FAILED"}, 2: {"status": "PASSED"}}
        )
        self.assertEqual(selected["effective_batch_size"], 8)
        blocked = select_surface07_microbatch(
            {4: {"status": "FAILED"}, 2: {"status": "FAILED"}, 1: {"status": "FAILED"}}
        )
        self.assertEqual(blocked["status"], "BLOCKED_SURFACE07_OOM")

    def test_unresolved_encoder_surface_is_rejected(self):
        del self.model.params["encoder.layers.19.self_attn.weight"]
        with self.assertRaisesRegex(
            RuntimeError,
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED",
        ):
            configure_surface07_trainable(self.model)

    def test_adr_phase4_and_exact_surface_are_required(self):
        validate_surface07_config(self.config, adr_text=self.adr_text)
        with self.assertRaises(ValueError):
            validate_surface07_config(self.config, adr_text="ADR 0009 without Phase 4")
        bad = copy.deepcopy(self.config)
        bad["trainable_surface"]["final_encoder_layer_indices"] = [18, 19, 20, 21, 22, 23]
        with self.assertRaises(ValueError):
            validate_surface07_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["trainable_surface"]["fusion_bridge_module"] = "prompt_embedding"
        with self.assertRaises(ValueError):
            validate_surface07_config(bad, adr_text=self.adr_text)

    def test_surface07_must_start_from_untouched_base(self):
        bad = copy.deepcopy(self.config)
        bad["model"]["initialization"] = "surface06_checkpoint"
        with self.assertRaises(ValueError):
            validate_surface07_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["model"]["checkpoint_sha256"] = "5" * 64
        with self.assertRaises(ValueError):
            validate_surface07_config(bad, adr_text=self.adr_text)

    def test_fixed_data_rejects_forbidden_sources_and_schedule_drift(self):
        for source in ("s6tts", "scale8000", "database-extension-v1"):
            bad = copy.deepcopy(self.config)
            bad["data"]["source_override"] = source
            with self.assertRaises(ValueError):
                validate_surface07_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["data"]["exposure_schedule_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_surface07_config(bad, adr_text=self.adr_text)

    def test_protected_surface_flags_are_fail_closed(self):
        for key in (
            "full_encoder_allowed",
            "surface08_allowed",
            "prompt_labels_tables_embeddings_allowed",
            "language_id_mapping_changes_allowed",
            "target_lang_machinery_changes_allowed",
            "non_selected_prompt_fusion_changes_allowed",
            "text_only_objective_allowed",
            "temporary_lm_head_allowed",
        ):
            bad = copy.deepcopy(self.config)
            bad["trainable_surface"][key] = True
            with self.assertRaises(ValueError):
                validate_surface07_config(bad, adr_text=self.adr_text)

    def test_classifier_uses_surface06_one_sided_envelope(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 30.0, "cer": 9.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 8.0, "cer": 2.0, "empty": 0},
            "fleurs_v2": {"wer": 44.0, "cer": 13.0, "empty": 0},
            "artur_j": {"wer": 50.0, "cer": 15.9, "empty": 0},
        }
        self.assertEqual(
            classify_surface07(metrics, selected_round=2),
            "SURFACE07_NEW_BEST_DIRECTIONAL_CANDIDATE",
        )
        rows = surface07_envelope_comparison(metrics)
        self.assertTrue(all(row["within_tolerance"] for row in rows))
        self.assertEqual(
            SURFACE07_BEST_REAL_GATE_ENVELOPE["artur_j"]["wer"]["source"],
            "Surface06",
        )

    def test_classifier_matches_with_acceptable_tradeoff(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 33.0, "cer": 9.5, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 9.0, "cer": 2.5, "empty": 0},
            "fleurs_v2": {"wer": 44.6, "cer": 13.7, "empty": 0},
            "artur_j": {"wer": 50.4, "cer": 15.7, "empty": 0},
        }
        self.assertEqual(
            classify_surface07(metrics, selected_round=3),
            "SURFACE07_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF",
        )

    def test_classifier_handles_blocker_empty_and_fleurs_regression(self):
        self.assertEqual(
            classify_surface07(SURFACE06_METRICS, fusion_bridge_proven=False),
            "BLOCKED_FUSION_BRIDGE_UNRESOLVED",
        )
        empty = {split: dict(values) for split, values in SURFACE06_METRICS.items()}
        empty["fleurs_v2"]["empty"] = 1
        self.assertEqual(
            classify_surface07(empty, selected_round=3),
            "SURFACE07_SYNTHETIC_OR_REAL_REGRESSION",
        )
        regression = {split: dict(values) for split, values in SURFACE06_METRICS.items()}
        regression["fleurs_v2"]["wer"] = 45.1
        regression["artur_j"]["wer"] = 50.0
        self.assertEqual(
            classify_surface07(regression, selected_round=3),
            "SURFACE07_FUSION_GOOD_BUT_FLEURS_REGRESSES",
        )

    def test_post_selection_metrics_do_not_change_round(self):
        binding = bind_post_selection_metrics(3, {"fleurs_v2": {"wer": 1.0}})
        self.assertEqual(binding["selected_round"], 3)

    def test_edit_components_are_not_fabricated(self):
        self.assertEqual(component_or_not_recorded({}, "delete"), "NOT_RECORDED")
        self.assertEqual(component_or_not_recorded({"delete": 0}, "delete"), 0)

    def test_public_report_rejects_raw_fields_and_local_paths(self):
        assert_public_report_safe({"surface_id": self.config["trainable_surface"]["surface_id"]})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"hypothesis": "forbidden"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"note": "/data-nvme/private"})

    def test_config_contains_only_fixed_scale2000_data(self):
        serialized = json.dumps(self.config["data"]).lower()
        for marker in ("s6tts", "scale8000", "database-extension"):
            self.assertNotIn(marker, serialized)

    def test_text_only_or_temporary_head_is_rejected(self):
        self.model.module_names.append("decoder_lm_adapter")
        with self.assertRaises(RuntimeError):
            configure_surface07_trainable(self.model)


if __name__ == "__main__":
    unittest.main()
