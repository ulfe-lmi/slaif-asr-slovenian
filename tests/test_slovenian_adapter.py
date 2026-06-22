from __future__ import annotations

import types
import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CPU-only CI
    raise unittest.SkipTest("PyTorch is not installed in the CPU-only CI environment") from exc

from slaif_asr.slovenian_adapter import (
    ResidualIntegrityReport,
    SlovenianResidualAdapter,
    adapter_state_hashes,
    changed_adapter_tensors,
    compare_base_hashes,
    derive_residual_adapter_selection,
    install_slovenian_residual_adapter,
    original_state_dict_from_wrapped_model,
    state_hashes,
    trainable_adapter_parameters,
)


class FakePromptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(
            model_defaults=types.SimpleNamespace(prompt_dictionary={"en-US": 0, "sl-SI": 2}, num_prompts=4),
            encoder=types.SimpleNamespace(d_model=3),
        )
        self.prompt_kernel = nn.Sequential(nn.Linear(7, 5), nn.GELU(), nn.Linear(5, 3))
        self.encoder = nn.Linear(3, 3)
        self.decoder = nn.Linear(3, 3)
        self.joint = nn.Linear(3, 3)


class SlovenianResidualAdapterTests(unittest.TestCase):
    def test_prompt_index_column_and_rank_are_derived(self) -> None:
        model = FakePromptModel()
        selection = derive_residual_adapter_selection(model, rank=4, prompt_name="sl-SI")
        self.assertEqual(selection.prompt_index, 2)
        self.assertEqual(selection.encoder_width, 3)
        self.assertEqual(selection.selected_column, 5)
        self.assertEqual(selection.prompt_kernel_output_width, 3)
        self.assertEqual(selection.trainable_parameters, 24)

    def test_invalid_rank_fails(self) -> None:
        with self.assertRaises(ValueError):
            install_slovenian_residual_adapter(FakePromptModel(), rank=0)

    def test_zero_initialization_reproduces_base_output(self) -> None:
        torch.manual_seed(4)
        model = FakePromptModel()
        inputs = torch.randn(2, 3, 7)
        base = model.prompt_kernel(inputs)
        _, adapter = install_slovenian_residual_adapter(model, rank=4)
        wrapped = model.prompt_kernel(inputs)
        self.assertIsInstance(adapter, SlovenianResidualAdapter)
        self.assertTrue(torch.allclose(base, wrapped, atol=0.0, rtol=0.0))

    def test_adapter_activates_only_for_slovenian_column(self) -> None:
        torch.manual_seed(5)
        linear = nn.Sequential(nn.Linear(7, 3))
        adapter = SlovenianResidualAdapter(linear, selected_column=5, output_width=3, rank=2)
        adapter.up.weight.data.fill_(1.0)
        adapter.down.weight.data.fill_(0.25)
        inputs = torch.randn(2, 3, 7)
        other_prompt = inputs.clone()
        other_prompt[..., 5] = 0.0
        sl_prompt = inputs.clone()
        sl_prompt[..., 5] = 1.0
        self.assertTrue(torch.equal(adapter(other_prompt), linear(other_prompt)))
        self.assertFalse(torch.equal(adapter(sl_prompt), linear(sl_prompt)))

    def test_only_adapter_parameters_require_grad(self) -> None:
        model = FakePromptModel()
        selection, adapter = install_slovenian_residual_adapter(model, rank=4)
        trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(
            trainable,
            [
                ("prompt_kernel.down.weight", 12),
                ("prompt_kernel.up.weight", 12),
            ],
        )
        self.assertEqual(sum(size for _, size in trainable), selection.trainable_parameters)

    def test_optimizer_parameters_reject_weight_decay(self) -> None:
        _, adapter = install_slovenian_residual_adapter(FakePromptModel(), rank=4)
        with self.assertRaises(ValueError):
            trainable_adapter_parameters(adapter, weight_decay=0.1)

    def test_base_state_hashes_remain_identical_after_adapter_change(self) -> None:
        model = FakePromptModel()
        base_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        base_hashes = state_hashes(base_state)
        _, adapter = install_slovenian_residual_adapter(model, rank=4)
        initial_adapter = adapter_state_hashes(adapter)
        adapter.up.weight.data.fill_(0.5)
        final_adapter = adapter_state_hashes(adapter)
        current_state = original_state_dict_from_wrapped_model(model)
        unexpected, missing, changed = compare_base_hashes(base_hashes, state_hashes(current_state))
        self.assertEqual(unexpected, [])
        self.assertEqual(missing, [])
        self.assertEqual(changed, [])
        self.assertEqual(changed_adapter_tensors(initial_adapter, final_adapter), ["up.weight"])

    def test_corrupted_base_parameter_causes_integrity_failure(self) -> None:
        model = FakePromptModel()
        base_hashes = state_hashes({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        install_slovenian_residual_adapter(model, rank=4)
        model.encoder.weight.data.add_(1.0)
        _, _, changed = compare_base_hashes(base_hashes, state_hashes(original_state_dict_from_wrapped_model(model)))
        self.assertIn("encoder.weight", changed)

    def test_integrity_report_passed_requires_no_base_changes(self) -> None:
        report = ResidualIntegrityReport(
            selected_prompt="sl-SI",
            prompt_index=2,
            selected_column=5,
            rank=4,
            trainable_parameters=24,
            base_tensors_identical=True,
            prompt_kernel_identical=True,
            encoder_identical=True,
            decoder_joint_identical=True,
            tokenizer_config_identical=True,
            changed_adapter_tensors=["up.weight"],
            unexpected_base_tensors=[],
            missing_base_tensors=[],
            unexpected_changed_base_tensors=[],
            unexpected_changed_elements=0,
            prompt_dictionary_unchanged=True,
            step_zero_equivalent=True,
            non_sl_residual_zero=True,
            optimizer_parameter_count=24,
        )
        self.assertTrue(report.passed())
        bad = report.__dict__ | {"unexpected_changed_base_tensors": ["encoder.weight"]}
        self.assertFalse(ResidualIntegrityReport(**bad).passed())


if __name__ == "__main__":
    unittest.main()
