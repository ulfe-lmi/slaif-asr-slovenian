from __future__ import annotations

import types
import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CPU-only CI
    raise unittest.SkipTest("PyTorch is not installed in the CPU-only CI environment") from exc

from slaif_asr.prompt_column import (
    PromptColumnDelta,
    compare_prompt_column_state_dicts,
    derive_prompt_column_selection,
    install_prompt_delta,
    merge_prompt_delta,
    trainable_delta_parameters,
)


class FakePromptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(
            model_defaults=types.SimpleNamespace(prompt_dictionary={"en-US": 0, "sl-SI": 2}, num_prompts=4),
            encoder=types.SimpleNamespace(d_model=3),
        )
        self.prompt_kernel = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        self.decoder = nn.Linear(3, 3)


class PromptColumnTests(unittest.TestCase):
    def test_prompt_index_and_column_are_derived(self) -> None:
        model = FakePromptModel()
        selection = derive_prompt_column_selection(model, "sl-SI")
        self.assertEqual(selection.prompt_index, 2)
        self.assertEqual(selection.encoder_width, 3)
        self.assertEqual(selection.selected_column, 5)
        self.assertEqual(selection.first_linear_shape, (5, 7))

    def test_wrong_prompt_name_fails(self) -> None:
        with self.assertRaises(ValueError):
            derive_prompt_column_selection(FakePromptModel(), "sl")

    def test_unexpected_prompt_kernel_shape_fails(self) -> None:
        model = FakePromptModel()
        model.prompt_kernel[0] = nn.Linear(8, 5)
        with self.assertRaises(ValueError):
            derive_prompt_column_selection(model, "sl-SI")

    def test_only_delta_is_trainable(self) -> None:
        model = FakePromptModel()
        selection, wrapper = install_prompt_delta(model, "sl-SI")
        trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(trainable, [("prompt_kernel.0.delta", selection.effective_trainable_parameters)])
        self.assertEqual(wrapper.effective_trainable_parameters, 5)

    def test_nonzero_weight_decay_is_rejected(self) -> None:
        _, wrapper = install_prompt_delta(FakePromptModel(), "sl-SI")
        with self.assertRaises(ValueError):
            trainable_delta_parameters(wrapper, weight_decay=0.01)

    def test_wrapper_matches_direct_selected_column_modification(self) -> None:
        torch.manual_seed(7)
        linear = nn.Linear(7, 5)
        wrapper = PromptColumnDelta(linear, selected_column=5)
        wrapper.delta.data.copy_(torch.randn(5))
        inputs = torch.randn(2, 4, 7)
        direct = nn.Linear(7, 5)
        direct.load_state_dict(linear.state_dict())
        with torch.no_grad():
            direct.weight[:, 5].add_(wrapper.delta)
        self.assertTrue(torch.allclose(wrapper(inputs), direct(inputs), atol=1e-6))

    def test_delta_inactive_for_other_prompt_columns(self) -> None:
        linear = nn.Linear(7, 5)
        wrapper = PromptColumnDelta(linear, selected_column=5)
        wrapper.delta.data.fill_(3.0)
        inputs = torch.randn(2, 4, 7)
        inputs[..., 5] = 0.0
        self.assertTrue(torch.equal(wrapper(inputs), linear(inputs)))

    def test_merge_restores_linear_and_preserves_output(self) -> None:
        torch.manual_seed(3)
        model = FakePromptModel()
        selection, wrapper = install_prompt_delta(model, "sl-SI")
        wrapper.delta.data.normal_()
        inputs = torch.randn(2, 2, 7)
        before = model.prompt_kernel[0](inputs)
        merged = merge_prompt_delta(model, selection)
        after = merged(inputs)
        self.assertIsInstance(model.prompt_kernel[0], nn.Linear)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_integrity_allows_only_selected_column(self) -> None:
        model = FakePromptModel()
        base = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        adapted = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        adapted["prompt_kernel.0.weight"][:, 5] += 1.0
        report = compare_prompt_column_state_dicts(
            base,
            adapted,
            first_linear_weight_name="prompt_kernel.0.weight",
            first_linear_bias_name="prompt_kernel.0.bias",
            selected_column=5,
            selected_prompt="sl-SI",
            prompt_index=2,
            effective_trainable_parameters=5,
        )
        self.assertEqual(report.unexpected_changed_tensors, [])
        self.assertEqual(report.unexpected_changed_elements, 0)
        self.assertTrue(report.selected_column_changed)
        self.assertTrue(report.other_columns_bitwise_identical)

    def test_integrity_fails_closed_for_other_columns(self) -> None:
        model = FakePromptModel()
        base = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        adapted = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        adapted["prompt_kernel.0.weight"][:, 4] += 1.0
        report = compare_prompt_column_state_dicts(
            base,
            adapted,
            first_linear_weight_name="prompt_kernel.0.weight",
            first_linear_bias_name="prompt_kernel.0.bias",
            selected_column=5,
            selected_prompt="sl-SI",
            prompt_index=2,
            effective_trainable_parameters=5,
        )
        self.assertIn("prompt_kernel.0.weight", report.unexpected_changed_tensors)
        self.assertGreater(report.unexpected_changed_elements, 0)


if __name__ == "__main__":
    unittest.main()
