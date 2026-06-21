from __future__ import annotations

import json
import unittest

from slaif_asr.contract import RuntimeContract, build_runtime_contract


class FakeParameter:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        total = 1
        for value in self.shape:
            total *= value
        return total


class FakeEncoder:
    layers = [object(), object(), object()]
    att_context_size = [[56, 0], [56, 1]]


class FakeTokenizer:
    vocab_size = 128


class FakeModel:
    cfg = {
        "encoder": {"d_model": 1024},
        "preprocessor": {"sample_rate": 16000},
        "prompt_dictionary": {"sl-SI": 37, "sl": 38},
    }
    encoder = FakeEncoder()
    tokenizer = FakeTokenizer()

    def parameters(self):
        return [FakeParameter([2, 3]), FakeParameter([5])]

    def named_parameters(self):
        return [("prompt_kernel.weight", FakeParameter([128, 1024]))]

    def set_inference_prompt(self, lang):
        return None


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_contract_serializes_expected_fields(self):
        contract = build_runtime_contract(FakeModel(), checkpoint_path="models/checkpoints/model.nemo")

        self.assertEqual(contract.total_parameters, 11)
        self.assertEqual(contract.encoder_layer_count, 3)
        self.assertEqual(contract.encoder_width, 1024)
        self.assertEqual(contract.sample_rate, 16000)
        self.assertEqual(contract.prompt_indices["sl-SI"], 37)
        self.assertEqual(contract.available_streaming_contexts, [[56, 0], [56, 1]])
        self.assertEqual(contract.prompt_kernel_structure[0]["parameters"], 131072)

        payload = json.loads(contract.to_json())
        self.assertEqual(payload["checkpoint"]["local_path"], "models/checkpoints/model.nemo")

    def test_runtime_contract_dataclass_json_round_trip(self):
        contract = RuntimeContract(
            loaded_class="example.Model",
            total_parameters=1,
            encoder_layer_count=None,
            encoder_width=None,
            tokenizer_vocabulary_size=None,
            sample_rate=None,
            prompt_indices={"sl-SI": None, "sl": None},
            prompt_kernel_structure=[],
            available_streaming_contexts=[[56, 13]],
            default_streaming_context=[56, 13],
            checkpoint={},
            environment={},
        )

        self.assertEqual(json.loads(contract.to_json())["loaded_class"], "example.Model")


if __name__ == "__main__":
    unittest.main()
