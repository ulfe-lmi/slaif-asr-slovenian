import copy
import json
import unittest

from slaif_asr.config import REPO_ROOT
from slaif_asr.gpu_policy import is_approved_development_gpu
from slaif_asr.trainable_surface_sweep import (
    SURFACE07_METRICS,
    SURFACE08_ALLOWED_TRAINABLE_PREFIXES,
    SURFACE08_BEST_REAL_GATE_ENVELOPE,
    apply_observed_training_ooms,
    assert_public_report_safe,
    bind_post_selection_metrics,
    classify_surface08,
    component_or_not_recorded,
    configure_surface08_trainable,
    discover_surface07_fusion_bridge,
    load_surface08_config,
    microbatch_plan,
    select_surface08_microbatch,
    should_stop_surface08_controller_curve,
    surface08_changed_tensor_summary,
    surface08_envelope_comparison,
    surface08_optimizer_parameter_groups,
    validate_surface08_config,
    verify_surface08_optimizer_scope,
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


class Surface08FullEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_surface08_config()
        cls.adr_text = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )

    def setUp(self):
        self.model = FakeModel()
        self.summary = configure_surface08_trainable(self.model)

    def test_selects_decoder_joint_all_encoder_layers_and_prompt_kernel(self):
        self.assertEqual(self.summary.decoder_parameter_count, 8)
        self.assertEqual(self.summary.joint_parameter_count, 9)
        self.assertEqual(self.summary.encoder_all_layers_parameter_count, 300)
        self.assertEqual(self.summary.fusion_bridge_parameter_count, 33)
        self.assertEqual(self.summary.fusion_bridge_module, "prompt_kernel")
        self.assertEqual(
            self.summary.encoder_layers,
            tuple(f"encoder.layers.{index}" for index in range(24)),
        )
        for name, parameter in self.model.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name.startswith(SURFACE08_ALLOWED_TRAINABLE_PREFIXES),
                name,
            )

    def test_frontend_and_prompt_identity_stay_frozen(self):
        for prefix in ("preprocessor.", "encoder.pre_encode."):
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for name, parameter in self.model.named_parameters()
                    if name.startswith(prefix)
                )
            )
        discovery = discover_surface07_fusion_bridge(self.model)
        self.assertEqual(discovery["status"], "PASSED")
        self.assertEqual(discovery["prompt_identity_storage"], "one_hot_config_not_parameter")

    def test_prompt_kernel_unresolved_blocks(self):
        missing = FakeModel()
        missing.module_names.remove("prompt_kernel")
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_PROMPT_KERNEL_UNRESOLVED"):
            configure_surface08_trainable(missing)

    def test_optimizer_scope_and_lower_full_encoder_lr(self):
        rates = {
            "decoder": 0.0005,
            "joint": 0.0005,
            "encoder_all_layers": 0.000005,
            "fusion_bridge": 0.00005,
        }
        groups = surface08_optimizer_parameter_groups(self.model, rates)
        verify_surface08_optimizer_scope(FakeOptimizer(groups), self.model, rates)
        by_name = {group["name"]: group["lr"] for group in groups}
        self.assertLess(by_name["encoder_all_layers"], by_name["fusion_bridge"])
        self.assertLess(by_name["fusion_bridge"], by_name["decoder"])

    def test_optimizer_rejects_unauthorized_parameter(self):
        rates = dict(self.config["training"]["learning_rates"])
        groups = surface08_optimizer_parameter_groups(self.model, rates)
        groups[0]["params"].append(self.model.params["encoder.pre_encode.conv.weight"])
        with self.assertRaises(RuntimeError):
            verify_surface08_optimizer_scope(FakeOptimizer(groups), self.model, rates)

    def test_parameter_integrity_allows_only_surface08(self):
        before = {
            "decoder.x": FakeTensor(1),
            "joint.x": FakeTensor(1),
            "encoder.layers.0.x": FakeTensor(1),
            "encoder.layers.23.x": FakeTensor(1),
            "prompt_kernel.0.weight": FakeTensor(1),
            "encoder.pre_encode.x": FakeTensor(1),
        }
        after = {name: FakeTensor(2) for name in before}
        after["encoder.pre_encode.x"] = FakeTensor(1)
        summary = surface08_changed_tensor_summary(before, after)
        self.assertTrue(summary["only_surface08_changed"])
        self.assertTrue(summary["encoder_all_layers_changed"])
        self.assertTrue(summary["fusion_bridge_changed"])
        self.assertTrue(summary["subsampling_frontend_unchanged"])

        after["encoder.pre_encode.x"] = FakeTensor(2)
        self.assertFalse(surface08_changed_tensor_summary(before, after)["only_surface08_changed"])

    def test_parameter_integrity_rejects_prompt_identity_movement(self):
        before = {"prompt_identity.weight": FakeTensor(1)}
        after = {"prompt_identity.weight": FakeTensor(2)}
        summary = surface08_changed_tensor_summary(before, after)
        self.assertFalse(summary["only_surface08_changed"])
        self.assertFalse(summary["prompt_identity_unchanged"])

    def test_microbatch_preserves_effective_batch_or_blocks(self):
        self.assertEqual(microbatch_plan(8)["gradient_accumulation_steps"], 1)
        self.assertEqual(microbatch_plan(2)["gradient_accumulation_steps"], 4)
        selected = select_surface08_microbatch(
            {8: {"status": "FAILED"}, 4: {"status": "PASSED"}}
        )
        self.assertEqual(selected["physical_microbatch"], 4)
        self.assertEqual(selected["effective_batch_size"], 8)
        blocked = select_surface08_microbatch(
            {
                8: {"status": "FAILED"},
                4: {"status": "FAILED"},
                2: {"status": "FAILED"},
                1: {"status": "FAILED"},
            }
        )
        self.assertEqual(blocked["status"], "BLOCKED_SURFACE08_OOM")

    def test_observed_training_oom_overrides_bounded_probe(self):
        outcomes = {
            4: {"status": "PASSED", "free_vram_mib": 2500},
            2: {"status": "PASSED", "free_vram_mib": 1000},
        }
        adjusted = apply_observed_training_ooms(
            outcomes,
            [
                {
                    "status": "FAILED_TRAINING_OOM",
                    "physical_microbatch": 4,
                    "optimizer_step": 1211,
                    "round": 1,
                }
            ],
        )
        self.assertEqual(adjusted[4]["status"], "FAILED")
        self.assertEqual(adjusted[4]["probe_status_before_override"], "PASSED")
        self.assertEqual(adjusted[4]["observed_optimizer_step"], 1211)
        self.assertEqual(select_surface08_microbatch(adjusted)["physical_microbatch"], 2)

    def test_encoder_surface_must_resolve_exactly_24_layers(self):
        del self.model.params["encoder.layers.19.self_attn.weight"]
        with self.assertRaisesRegex(
            RuntimeError,
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED",
        ):
            configure_surface08_trainable(self.model)

    def test_adr_phase5_exact_surface_and_untouched_base_are_required(self):
        validate_surface08_config(self.config, adr_text=self.adr_text)
        with self.assertRaises(ValueError):
            validate_surface08_config(self.config, adr_text="ADR 0009 without Phase 5")
        bad = copy.deepcopy(self.config)
        bad["trainable_surface"]["encoder_layer_indices"] = list(range(1, 24))
        with self.assertRaises(ValueError):
            validate_surface08_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["model"]["initialization"] = "surface07_checkpoint"
        with self.assertRaises(ValueError):
            validate_surface08_config(bad, adr_text=self.adr_text)

    def test_fixed_data_rejects_forbidden_sources_and_schedule_drift(self):
        for source in ("s6tts", "scale8000", "database-extension-v1", "real_speech"):
            bad = copy.deepcopy(self.config)
            bad["data"]["source_override"] = source
            with self.assertRaises(ValueError):
                validate_surface08_config(bad, adr_text=self.adr_text)
        bad = copy.deepcopy(self.config)
        bad["data"]["exposure_schedule_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_surface08_config(bad, adr_text=self.adr_text)

    def test_surface09_full_model_and_protected_surfaces_are_rejected(self):
        for key in (
            "surface09_allowed",
            "full_model_allowed",
            "preprocessor_training_allowed",
            "frontend_subsampling_training_allowed",
            "prompt_labels_tables_embeddings_allowed",
            "prompt_identity_mapping_changes_allowed",
            "language_id_mapping_changes_allowed",
            "target_lang_machinery_changes_allowed",
            "non_selected_prompt_fusion_changes_allowed",
            "text_only_objective_allowed",
            "temporary_lm_head_allowed",
        ):
            bad = copy.deepcopy(self.config)
            bad["trainable_surface"][key] = True
            with self.assertRaises(ValueError):
                validate_surface08_config(bad, adr_text=self.adr_text)

    def test_surface08_extra_stop_guards(self):
        rows = [
            {
                "round": 0,
                "wer": 67.0,
                "empty": 12,
                "synthetic_anchor_probe_loss": 4.0,
                "synthetic_scale_probe_loss": 4.0,
            },
            {
                "round": 1,
                "wer": 45.0,
                "empty": 0,
                "synthetic_anchor_probe_loss": 3.0,
                "synthetic_scale_probe_loss": 3.0,
            },
            {
                "round": 2,
                "wer": 45.5,
                "empty": 1,
                "synthetic_anchor_probe_loss": 2.5,
                "synthetic_scale_probe_loss": 2.5,
            },
        ]
        self.assertEqual(
            should_stop_surface08_controller_curve(rows)["reason"],
            "surface08_empty_hypotheses_reappeared",
        )
        rows[-1]["empty"] = 0
        rows[-2]["wer"] = 49.0
        rows[-1]["wer"] = 49.1
        self.assertEqual(
            should_stop_surface08_controller_curve(rows)["reason"],
            "surface08_two_consecutive_rounds_worse_than_surface07_by_5",
        )

    def test_classifier_uses_surface07_one_sided_envelope(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 20.0, "cer": 7.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 7.0, "cer": 2.0, "empty": 0},
            "fleurs_v2": {"wer": 41.9, "cer": 12.8, "empty": 0},
            "artur_j": {"wer": 47.2, "cer": 15.0, "empty": 0},
        }
        self.assertEqual(
            classify_surface08(metrics, selected_round=2),
            "SURFACE08_NEW_BEST_DIRECTIONAL_CANDIDATE",
        )
        self.assertTrue(all(row["within_tolerance"] for row in surface08_envelope_comparison(metrics)))
        self.assertEqual(
            SURFACE08_BEST_REAL_GATE_ENVELOPE["artur_j"]["wer"]["source"],
            "Surface07",
        )

    def test_classifier_match_regression_and_empty(self):
        match = {split: dict(values) for split, values in SURFACE07_METRICS.items()}
        match["fleurs_v2"]["wer"] = 42.4
        match["artur_j"]["wer"] = 47.2
        self.assertEqual(
            classify_surface08(match, selected_round=3),
            "SURFACE08_MATCHES_SURFACE07_WITH_ACCEPTABLE_TRADEOFF",
        )
        regression = {split: dict(values) for split, values in SURFACE07_METRICS.items()}
        regression["fleurs_v2"]["wer"] = 43.0
        self.assertEqual(
            classify_surface08(regression, selected_round=3),
            "SURFACE08_ARTUR_DEV_GOOD_BUT_FLEURS_REGRESSES",
        )
        empty = {split: dict(values) for split, values in SURFACE07_METRICS.items()}
        empty["artur_j"]["empty"] = 1
        self.assertEqual(
            classify_surface08(empty, selected_round=3),
            "SURFACE08_SYNTHETIC_OVERFIT_OR_REAL_REGRESSION",
        )

    def test_post_selection_public_safety_and_edit_components(self):
        binding = bind_post_selection_metrics(3, {"fleurs_v2": {"wer": 1.0}})
        self.assertEqual(binding["selected_round"], 3)
        self.assertEqual(component_or_not_recorded({}, "delete"), "NOT_RECORDED")
        self.assertEqual(component_or_not_recorded({"delete": 0}, "delete"), 0)
        assert_public_report_safe({"surface_id": self.config["trainable_surface"]["surface_id"]})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"hypothesis": "forbidden"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"note": "/data-nvme/private"})

    def test_config_has_no_forbidden_data_and_rtx3090_is_approved(self):
        serialized = json.dumps(self.config["data"]).lower()
        for marker in ("s6tts", "scale8000", "database-extension"):
            self.assertNotIn(marker, serialized)
        self.assertTrue(is_approved_development_gpu("NVIDIA GeForce RTX 3090", 24576))

    def test_text_only_or_temporary_head_is_rejected(self):
        self.model.module_names.append("decoder_lm_adapter")
        with self.assertRaises(RuntimeError):
            configure_surface08_trainable(self.model)


if __name__ == "__main__":
    unittest.main()
