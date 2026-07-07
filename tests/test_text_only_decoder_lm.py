import unittest

import torch

from slaif_asr.text_only_decoder_lm import (
    TextRow,
    TokenizedRow,
    batch_order,
    decoder_lm_forward_loss,
    deterministic_text_split,
    make_lm_batch,
    perplexity,
    tokenize_split,
)


class FakeTokenizer:
    bos_id = 10
    eos_id = 11
    pad_id = 0
    vocab_size = 32

    def text_to_ids(self, text):
        return [ord(ch) % 20 + 1 for ch in text if ch != " "]


class TextOnlyDecoderLMTests(unittest.TestCase):
    def rows(self, count=64000):
        return [TextRow(row_id=f"row-{index:05d}", normalized_text=f"besedilo {index}", source_family_id="s", utterance_family_id=f"u{index}") for index in range(count)]

    def test_deterministic_split_counts_and_disjointness(self):
        split = deterministic_text_split(self.rows(), "sl-corpus-v5-scale8000-training-v1")
        self.assertEqual(len(split["train"]), 60800)
        self.assertEqual(len(split["validation"]), 3200)
        self.assertFalse({row.row_id for row in split["train"]} & {row.row_id for row in split["validation"]})
        self.assertEqual([row.row_id for row in split["train"][:10]], [row.row_id for row in deterministic_text_split(self.rows(), "sl-corpus-v5-scale8000-training-v1")["train"][:10]])

    def test_tokenize_stats(self):
        split = {"train": self.rows(3), "validation": self.rows(2)}
        tokenized, stats = tokenize_split(split, FakeTokenizer())
        self.assertEqual(stats["vocabulary_size"], 32)
        self.assertEqual(stats["rows_rejected_by_tokenization"], 0)
        self.assertEqual(len(tokenized["train"]), 3)

    def test_label_shift_and_padding_mask(self):
        rows = [TokenizedRow("a", [3, 4], "train"), TokenizedRow("b", [5], "train")]
        batch = make_lm_batch(rows, bos_id=10, eos_id=11, pad_id=0, device="cpu")
        self.assertEqual(batch["input_ids"].tolist(), [[10, 3, 4], [10, 5, 0]])
        self.assertEqual(batch["labels"].tolist(), [[3, 4, 11], [5, 11, 0]])
        self.assertEqual(batch["mask"].tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(batch["lengths"].tolist(), [3, 2])

    def test_batch_order_is_deterministic(self):
        rows = [TokenizedRow(str(index), [1] * (index % 5 + 1), "train") for index in range(20)]
        self.assertEqual(batch_order(rows, epoch=1, seed=1234, batch_size=4), batch_order(rows, epoch=1, seed=1234, batch_size=4))
        self.assertNotEqual(batch_order(rows, epoch=1, seed=1234, batch_size=4), batch_order(rows, epoch=2, seed=1234, batch_size=4))

    def test_perplexity(self):
        self.assertAlmostEqual(perplexity(0.0), 1.0)
        self.assertGreater(perplexity(1.0), 2.0)

    def test_decoder_lm_loss_uses_matching_timesteps(self):
        calls = []

        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)

            def predict(self, y, add_sos=False):
                calls.append(add_sos)
                return self.embedding(y), None

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = FakeDecoder()

        rows = [TokenizedRow("a", [3, 4], "train"), TokenizedRow("b", [5], "train")]
        batch = make_lm_batch(rows, bos_id=10, eos_id=11, pad_id=0, device="cpu")
        loss = decoder_lm_forward_loss(FakeModel(), torch.nn.Linear(4, 16), batch, pad_id=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(calls, [False])


if __name__ == "__main__":
    unittest.main()
