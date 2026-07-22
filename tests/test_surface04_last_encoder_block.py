import unittest

from slaif_asr.trainable_surface_sweep import (
    FINAL_ENCODER_BLOCK_PREFIX,
    changed_tensor_summary,
    configure_surface04_trainable,
    microbatch_plan,
    optimizer_parameter_groups,
    select_microbatch,
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
            "encoder.layers.0.self_attn.weight": FakeParam(3),
            "encoder.layers.22.self_attn.weight": FakeParam(4),
            "encoder.layers.23.self_attn.weight": FakeParam(5),
            "encoder.layers.23.norm_out.weight": FakeParam(6),
            "prompt_kernel.0.weight": FakeParam(7),
            "decoder.prediction.weight": FakeParam(8),
            "joint.joint_net.weight": FakeParam(9),
        }

    def parameters(self):
        return list(self.params.values())

    def named_parameters(self):
        return list(self.params.items())

    def named_modules(self):
        return [("", self), ("encoder", object()), ("decoder", object()), ("joint", object())]


class FakeOptimizer:
    def __init__(self, groups):
        self.param_groups = groups


class Surface04LastEncoderBlockTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.summary = configure_surface04_trainable(self.model)

    def test_selects_exactly_decoder_joint_and_final_encoder_block(self):
        self.assertEqual(self.summary.decoder_parameter_count, 8)
        self.assertEqual(self.summary.joint_parameter_count, 9)
        self.assertEqual(self.summary.final_encoder_block_parameter_count, 11)
        for name, parameter in self.model.named_parameters():
            expected = name.startswith(("decoder.", "joint.", FINAL_ENCODER_BLOCK_PREFIX))
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_lower_encoder_preprocessor_frontend_and_prompt_are_frozen(self):
        for prefix in ("preprocessor.", "encoder.pre_encode.", "encoder.layers.0.", "encoder.layers.22.", "prompt_kernel."):
            self.assertTrue(all(not parameter.requires_grad for name, parameter in self.model.named_parameters() if name.startswith(prefix)))

    def test_optimizer_scope_and_learning_rates(self):
        rates = {"decoder": 0.0005, "joint": 0.0005, "final_encoder_block": 0.00002}
        groups = optimizer_parameter_groups(self.model, rates)
        optimizer = FakeOptimizer(groups)
        verify_optimizer_scope(optimizer, self.model, rates)
        self.assertLess(groups[2]["lr"], groups[0]["lr"])

    def test_optimizer_rejects_unauthorized_parameter(self):
        rates = {"decoder": 0.0005, "joint": 0.0005, "final_encoder_block": 0.00002}
        groups = optimizer_parameter_groups(self.model, rates)
        groups[0]["params"].append(self.model.params["encoder.layers.22.self_attn.weight"])
        with self.assertRaises(RuntimeError):
            verify_optimizer_scope(FakeOptimizer(groups), self.model, rates)

    def test_parameter_integrity_flags_lower_encoder_change(self):
        before = {"decoder.x": FakeTensor(1), "joint.x": FakeTensor(1), "encoder.layers.23.x": FakeTensor(1), "encoder.layers.22.x": FakeTensor(1)}
        after = {"decoder.x": FakeTensor(2), "joint.x": FakeTensor(2), "encoder.layers.23.x": FakeTensor(2), "encoder.layers.22.x": FakeTensor(1)}
        self.assertTrue(changed_tensor_summary(before, after)["only_surface04_changed"])
        after["encoder.layers.22.x"] = FakeTensor(2)
        self.assertFalse(changed_tensor_summary(before, after)["only_surface04_changed"])

    def test_microbatch_preserves_effective_batch_or_blocks(self):
        self.assertEqual(microbatch_plan(2)["gradient_accumulation_steps"], 4)
        selected = select_microbatch({4: {"status": "FAILED"}, 2: {"status": "PASSED"}})
        self.assertEqual(selected["effective_batch_size"], 8)
        self.assertEqual(selected["physical_microbatch"], 2)
        blocked = select_microbatch({4: {"status": "FAILED"}, 2: {"status": "FAILED"}, 1: {"status": "FAILED"}})
        self.assertEqual(blocked["status"], "BLOCKED_SURFACE04_OOM")

    def test_text_only_or_temporary_head_is_rejected(self):
        self.model.named_modules = lambda: [("", self.model), ("decoder_lm_adapter", object())]
        with self.assertRaises(RuntimeError):
            configure_surface04_trainable(self.model)


if __name__ == "__main__":
    unittest.main()
