import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None:

    class DecoderLMAdapterTests(unittest.TestCase):
        @unittest.skip("PyTorch is not installed in the CPU repository-check environment")
        def test_torch_required(self):
            pass

else:
    from slaif_asr.decoder_lm_adapter import (
        ADAPTER_NAME,
        TemporaryLMHead,
        compare_pretrained_state,
        configure_text_only_trainable,
        disable_decoder_lm_adapter,
        enable_decoder_lm_adapter,
        enabled_decoder_lm_adapters,
        install_decoder_lm_adapter,
        pretrained_parameters_with_grad,
        state_dict_cpu,
        text_only_optimizer_parameters,
        verify_text_only_optimizer_scope,
    )


    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pred_hidden = 4
            self.embedding = torch.nn.Embedding(16, 4)
            self.base = torch.nn.Linear(4, 4)

        def predict(self, y=None, state=None, add_sos=True, batch_size=None):
            if y is None:
                out = torch.zeros((batch_size or 1, 1, 4))
            else:
                out = self.base(self.embedding(y))
            if add_sos:
                out = torch.cat([torch.zeros((out.shape[0], 1, out.shape[2])), out], dim=1)
            return out, state

        def forward(self, targets, target_length):
            out, state = self.predict(targets, add_sos=False)
            return out.transpose(1, 2), target_length, state


    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.decoder = FakeDecoder()
            self.joint = torch.nn.Linear(4, 4)


    class DecoderLMAdapterTests(unittest.TestCase):
        def test_zero_effect_and_enable_disable(self):
            torch.manual_seed(4)
            model = FakeModel()
            y = torch.tensor([[1, 2, 3]])
            base, _ = model.decoder.predict(y, add_sos=False)
            summary = install_decoder_lm_adapter(model)
            self.assertEqual(summary["adapter_name"], ADAPTER_NAME)
            self.assertEqual(enabled_decoder_lm_adapters(model), [])
            enabled = enable_decoder_lm_adapter(model)
            adapted, _ = model.decoder.predict(y, add_sos=False)
            self.assertEqual(enabled, [ADAPTER_NAME])
            self.assertTrue(torch.equal(base, adapted))
            disable_decoder_lm_adapter(model)
            disabled, _ = model.decoder.predict(y, add_sos=False)
            self.assertTrue(torch.equal(base, disabled))

        def test_trainable_whitelist_and_optimizer_scope(self):
            model = FakeModel()
            install_decoder_lm_adapter(model)
            lm_head = TemporaryLMHead(4, 16)
            summary = configure_text_only_trainable(model, lm_head)
            self.assertGreater(summary["adapter_trainable_parameters"], 0)
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            self.assertTrue(all(name.startswith("decoder.decoder_lm_adapter.") for name in trainable))
            optimizer = torch.optim.AdamW(text_only_optimizer_parameters(model, lm_head), lr=0.001)
            verify_text_only_optimizer_scope(optimizer, model, lm_head)

        def test_pretrained_integrity_detects_only_base_change(self):
            model = FakeModel()
            install_decoder_lm_adapter(model)
            before = state_dict_cpu(model)
            with torch.no_grad():
                model.decoder.decoder_lm_adapter.down.weight.add_(1.0)
            self.assertTrue(compare_pretrained_state(before, state_dict_cpu(model))["pretrained_tensors_unchanged"])
            with torch.no_grad():
                model.encoder.weight.add_(1.0)
            self.assertFalse(compare_pretrained_state(before, state_dict_cpu(model))["pretrained_tensors_unchanged"])

        def test_pretrained_grad_detection(self):
            model = FakeModel()
            install_decoder_lm_adapter(model)
            y = torch.tensor([[1, 2]])
            out, _ = model.decoder.predict(y, add_sos=False)
            out.sum().backward()
            self.assertIn("decoder.embedding.weight", pretrained_parameters_with_grad(model))


if __name__ == "__main__":
    unittest.main()
