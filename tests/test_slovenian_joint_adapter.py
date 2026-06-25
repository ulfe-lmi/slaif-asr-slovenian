from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for joint-adapter tests") from exc

from slaif_asr.slovenian_joint_adapter import (
    ADAPTER_NAME,
    add_joint_adapter,
    compare_adapter_state,
    compare_base_state,
    configure_only_adapter_trainable,
    disable_joint_adapters,
    enable_for_target_language,
    enabled_joint_adapters,
    expected_trainable_count,
    load_adapter_artifact,
    load_adapter_spec,
    optimizer_parameter_ids,
    save_adapter_artifact,
    state_dict_cpu,
)


class FakeJoint(torch.nn.Module):
    def __init__(self, joint_hidden: int = 4) -> None:
        super().__init__()
        self.joint_hidden = joint_hidden
        self.base = torch.nn.Linear(joint_hidden, joint_hidden)
        self.adapter_layer = torch.nn.ModuleDict()
        self.adapter_cfg: dict[str, dict[str, bool]] = {}

    def add_adapter(self, name: str, cfg) -> None:
        if name in self.adapter_layer:
            raise ValueError("duplicate adapter")
        self.adapter_layer[name] = torch.nn.Sequential(
            torch.nn.LayerNorm(self.joint_hidden),
            torch.nn.Linear(self.joint_hidden, 2, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(2, self.joint_hidden, bias=False),
        )
        self.adapter_layer[name][-1].weight.data *= 0
        self.adapter_cfg[name] = {"enabled": True}

    def get_adapter_module(self, name: str):
        return self.adapter_layer[name] if name in self.adapter_layer else None

    def is_adapter_available(self) -> bool:
        return bool(self.adapter_layer)

    def set_enabled_adapters(self, name: str | None = None, enabled: bool = True) -> None:
        if name is None:
            for key in self.adapter_cfg:
                self.adapter_cfg[key]["enabled"] = enabled
        else:
            self.adapter_cfg[name]["enabled"] = enabled

    def get_enabled_adapters(self) -> list[str]:
        return [name for name, cfg in self.adapter_cfg.items() if cfg["enabled"]]

    def forward(self, x):
        x = self.base(x)
        for name in self.get_enabled_adapters():
            x = x + self.adapter_layer[name](x)
        return x


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.joint = FakeJoint()


class SlovenianJointAdapterTests(unittest.TestCase):
    def spec(self):
        return load_adapter_spec("configs/adapters/sl_si_joint_adapter_v1.json")

    def test_adapter_config_parsing_and_trainable_formula(self) -> None:
        spec = self.spec()
        self.assertEqual(spec.name, ADAPTER_NAME)
        self.assertEqual(spec.bottleneck_dim, 32)
        self.assertEqual(expected_trainable_count(4, 2), 24)

    def test_adapter_added_only_to_joint_and_disabled_by_default(self) -> None:
        model = FakeModel()
        summary = add_joint_adapter(model, self.spec())
        self.assertIn(ADAPTER_NAME, model.joint.adapter_layer)
        self.assertFalse(enabled_joint_adapters(model))
        self.assertEqual(summary["joint_hidden"], 4)
        self.assertFalse(hasattr(model.encoder, "adapter_layer"))

    def test_zero_initialized_enabled_and_disabled_parity(self) -> None:
        torch.manual_seed(1)
        model = FakeModel()
        x = torch.randn(2, 4)
        base = model.joint(x).detach().clone()
        add_joint_adapter(model, self.spec())
        enable_for_target_language(model, "sl-SI")
        self.assertTrue(torch.equal(base, model.joint(x).detach()))
        disable_joint_adapters(model)
        self.assertTrue(torch.equal(base, model.joint(x).detach()))

    def test_language_gating(self) -> None:
        model = FakeModel()
        add_joint_adapter(model, self.spec())
        self.assertEqual(enable_for_target_language(model, "sl-SI"), [ADAPTER_NAME])
        self.assertEqual(enable_for_target_language(model, "en-US"), [])
        with self.assertRaisesRegex(ValueError, "target language"):
            enable_for_target_language(model, "")

    def test_only_adapter_parameters_are_trainable_and_optimizer_ids_match(self) -> None:
        model = FakeModel()
        add_joint_adapter(model, self.spec())
        summary = configure_only_adapter_trainable(model)
        self.assertGreater(summary["trainable_parameters"], 0)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(all(name.startswith(f"joint.adapter_layer.{ADAPTER_NAME}.") for name in trainable))
        optimizer = torch.optim.AdamW([parameter for _name, parameter in model.named_parameters() if parameter.requires_grad])
        actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(actual, optimizer_parameter_ids(model))

    def test_base_integrity_and_adapter_change_detection(self) -> None:
        model = FakeModel()
        add_joint_adapter(model, self.spec())
        initial = state_dict_cpu(model)
        with torch.no_grad():
            model.joint.adapter_layer[ADAPTER_NAME][1].weight.add_(1.0)
        trained = state_dict_cpu(model)
        self.assertTrue(compare_base_state(initial, trained)["base_tensors_identical"])
        changed = compare_adapter_state(initial, trained)["adapter_tensors_changed"]
        self.assertTrue(any(name.endswith("1.weight") for name in changed))
        with torch.no_grad():
            model.encoder.weight.add_(1.0)
        self.assertFalse(compare_base_state(initial, state_dict_cpu(model))["base_tensors_identical"])

    def test_adapter_save_restore_disabled(self) -> None:
        model = FakeModel()
        spec = self.spec()
        add_joint_adapter(model, spec)
        enable_for_target_language(model, "sl-SI")
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "adapter.pt"
            digest = save_adapter_artifact(path, model=model, spec=spec, metadata={"adapter_config": {"safe": True}})
            self.assertEqual(len(digest), 64)
            restored = FakeModel()
            payload = load_adapter_artifact(path, model=restored, spec=spec)
            self.assertEqual(payload["adapter_name"], ADAPTER_NAME)
            self.assertEqual(enabled_joint_adapters(restored), [])


if __name__ == "__main__":
    unittest.main()
