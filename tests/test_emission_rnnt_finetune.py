import unittest

from slaif_asr.emission_rnnt_finetune import (
    assert_public_report_safe,
    changed_tensor_summary,
    configure_decoder_joint_trainable,
    microbatch_plan,
    optimizer_scope_summary,
    validate_config,
    validate_microbatch_selection,
    verify_optimizer_scope,
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
        self.grad = None

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self._count


class FakeModel:
    def __init__(self):
        self.params = {
            "encoder.weight": FakeParam(100),
            "preprocessor.weight": FakeParam(5),
            "prompt_encoder.weight": FakeParam(7),
            "decoder.embedding.weight": FakeParam(11),
            "joint.joint_net.weight": FakeParam(13),
        }

    def parameters(self):
        return list(self.params.values())

    def named_parameters(self):
        return list(self.params.items())

    def named_modules(self):
        return [("", self), ("encoder", object()), ("decoder", object()), ("joint", object())]


class FakeOptimizer:
    def __init__(self, params):
        self.param_groups = [{"params": params}]


VALID_CONFIG = {
    "work_order_id": "0030",
    "status": "DIAGNOSTIC_ONLY",
    "accepted_parent": "none",
    "data": {
        "semantic_rows": 16000,
        "view_records": 320000,
        "clean_files": 144000,
        "augmented_files": 176000,
        "fixed_text_sha256": "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14",
        "all_views_sha256": "9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca",
        "exposure_schedule_sha256": "6757018f3306839ce8564ba758e13e231ab4784bf98049b65701b963b55e5842",
    },
    "training": {
        "semantic_rows": 16000,
        "sample_exposures": 320000,
        "effective_batch_size": 8,
        "optimizer_steps": 40000,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_clipping": "none",
        "seed": 1234,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": False,
        "physical_microbatch_candidates": [8, 4, 2, 1],
    },
    "trainable_surface": {
        "allowed_prefixes": ["decoder.", "joint."],
        "text_only_path_allowed": False,
    },
    "evaluation": {
        "batch_size": 32,
        "duration_bucketing": True,
        "canonical": False,
        "promotion_eligible": False,
    },
}


class EmissionRnntFinetuneTests(unittest.TestCase):
    def test_config_requires_decoder_joint_audio_protocol(self):
        validate_config(VALID_CONFIG)
        bad = {**VALID_CONFIG, "trainable_surface": {"allowed_prefixes": ["decoder."], "text_only_path_allowed": False}}
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_config_rejects_text_only_path(self):
        bad = {**VALID_CONFIG, "trainable_surface": {"allowed_prefixes": ["decoder.", "joint."], "text_only_path_allowed": True}}
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_decoder_joint_trainable_selection_freezes_encoder_prompt(self):
        model = FakeModel()
        summary = configure_decoder_joint_trainable(model)
        self.assertEqual(summary.decoder_parameter_count, 11)
        self.assertEqual(summary.joint_parameter_count, 13)
        self.assertFalse(model.params["encoder.weight"].requires_grad)
        self.assertFalse(model.params["prompt_encoder.weight"].requires_grad)
        self.assertTrue(model.params["decoder.embedding.weight"].requires_grad)
        self.assertTrue(model.params["joint.joint_net.weight"].requires_grad)

    def test_optimizer_scope_only_decoder_joint(self):
        model = FakeModel()
        configure_decoder_joint_trainable(model)
        optimizer = FakeOptimizer([model.params["decoder.embedding.weight"], model.params["joint.joint_net.weight"]])
        verify_optimizer_scope(optimizer, model)
        bad_optimizer = FakeOptimizer([model.params["encoder.weight"]])
        with self.assertRaises(RuntimeError):
            verify_optimizer_scope(bad_optimizer, model)
        self.assertFalse(optimizer_scope_summary(model)["contains_text_only_lm"])

    def test_microbatch_selection_preserves_effective_batch(self):
        selected = validate_microbatch_selection(
            [8, 4, 2, 1],
            {8: {"status": "FAILED"}, 4: {"status": "PASSED"}, 2: {"status": "PASSED"}, 1: {"status": "PASSED"}},
        )
        self.assertEqual(selected, {"status": "PASSED", "physical_microbatch": 4, "gradient_accumulation_steps": 2, "effective_batch_size": 8})
        self.assertEqual(microbatch_plan(1)["gradient_accumulation_steps"], 8)

    def test_physical_batch_one_failure_blocks(self):
        selected = validate_microbatch_selection(
            [8, 4, 2, 1],
            {8: {"status": "FAILED"}, 4: {"status": "FAILED"}, 2: {"status": "FAILED"}, 1: {"status": "FAILED"}},
        )
        self.assertEqual(selected["status"], "ENVIRONMENT_BLOCKED")

    def test_changed_tensor_summary_allows_only_decoder_joint(self):
        before = {"encoder.x": FakeTensor(1), "decoder.x": FakeTensor(1), "joint.x": FakeTensor(1), "prompt.x": FakeTensor(1)}
        after = {"encoder.x": FakeTensor(1), "decoder.x": FakeTensor(2), "joint.x": FakeTensor(2), "prompt.x": FakeTensor(1)}
        summary = changed_tensor_summary(before, after)
        self.assertTrue(summary["only_decoder_joint_changed"])
        self.assertTrue(summary["encoder_unchanged"])
        self.assertTrue(summary["prompt_kernel_unchanged"])
        after["encoder.x"] = FakeTensor(3)
        self.assertFalse(changed_tensor_summary(before, after)["only_decoder_joint_changed"])

    def test_public_report_privacy_rejects_raw_fields(self):
        assert_public_report_safe({"classification": "DECODER_JOINT_RNNT_BEATS_BASE_BUT_NOT_SCALE2000"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"text": "raw sentence"})


if __name__ == "__main__":
    unittest.main()
