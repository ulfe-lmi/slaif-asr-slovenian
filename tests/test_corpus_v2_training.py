from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for corpus-v2 training tests") from exc

from slaif_asr.corpus_v2_training import (
    TrainingRecord,
    assert_epoch_covers_once,
    assert_public_report_safe,
    deterministic_epoch_batches,
    git_tracked_and_clean_at_head,
    make_training_batch,
    parameter_integrity_before_merge,
    select_probe_records,
    selection_from_benchmark,
)
from slaif_asr.prompt_column import PromptColumnSelection


def record(index: int, *, duration: float | None = None) -> TrainingRecord:
    return TrainingRecord(
        selected_training_id=f"selected-{index:03d}",
        audio_filepath=f"/ignored/{index}.wav",
        duration=duration if duration is not None else 1.0 + index / 10.0,
        text=f"Besedilo {index}",
        text_sha256=f"text-{index}",
        audio_sha256=f"audio-{index}",
        selection_reason="hard" if index % 2 == 0 else "control",
        selection_rank=index + 1,
    )


class FakeTokenizer:
    def text_to_ids(self, text: str) -> list[int]:
        return [ord(char) % 31 + 1 for char in text]


class FakeModel:
    tokenizer = FakeTokenizer()


def write_wav(path: Path, samples: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((b"\x01\x00") * samples)


class CorpusV2TrainingTests(unittest.TestCase):
    def test_epoch_batches_cover_every_training_row_once(self) -> None:
        rows = [record(index) for index in range(10)]
        layout = deterministic_epoch_batches(rows, batch_size=4, epoch=1, seed=1234, bucketed=True)
        assert_epoch_covers_once(layout, len(rows))
        self.assertEqual(sorted(len(batch) for batch in layout.batches), [2, 4, 4])
        self.assertEqual(layout.final_partial_batch_size, 2)

    def test_epoch_order_is_deterministic(self) -> None:
        rows = [record(index, duration=2.0 - index / 100.0) for index in range(12)]
        first = deterministic_epoch_batches(rows, batch_size=3, epoch=2, seed=7, bucketed=True)
        second = deterministic_epoch_batches(rows, batch_size=3, epoch=2, seed=7, bucketed=True)
        self.assertEqual(first.batches, second.batches)
        self.assertGreater(first.padding_ratio, 1.0)

    def test_probe_selection_uses_stable_hash(self) -> None:
        rows = [record(index) for index in range(20)]
        first = [item.selected_training_id for item in select_probe_records(rows, 5)]
        second = [item.selected_training_id for item in select_probe_records(list(reversed(rows)), 5)]
        self.assertEqual(first, second)

    def test_padded_audio_and_transcript_batch_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            left = tmp / "left.wav"
            right = tmp / "right.wav"
            write_wav(left, 20)
            write_wav(right, 35)
            rows = [
                record(1, duration=20 / 16000),
                record(2, duration=35 / 16000),
            ]
            rows[0] = TrainingRecord(**(rows[0].__dict__ | {"audio_filepath": str(left), "text": "ena"}))
            rows[1] = TrainingRecord(**(rows[1].__dict__ | {"audio_filepath": str(right), "text": "daljše"}))
            signal, signal_len, transcript, transcript_len = make_training_batch(FakeModel(), rows, device="cpu")
            self.assertEqual(tuple(signal.shape), (2, 35))
            self.assertEqual(signal_len.tolist(), [20, 35])
            self.assertEqual(transcript_len.tolist(), [3, 6])
            self.assertEqual(tuple(transcript.shape), (2, 6))

    def test_batch_policy_selects_smallest_within_best(self) -> None:
        rows = [
            {"batch_size": 1, "status": "PASSED", "correctness": {"passed": True}, "audio_seconds_per_wall_second": 1.0},
            {"batch_size": 2, "status": "PASSED", "correctness": {"passed": True}, "audio_seconds_per_wall_second": 9.6},
            {"batch_size": 4, "status": "PASSED", "correctness": {"passed": True}, "audio_seconds_per_wall_second": 10.0},
            {"batch_size": 8, "status": "FAILED", "correctness": {"passed": False}, "audio_seconds_per_wall_second": 20.0},
        ]
        selected = selection_from_benchmark(rows, within_best_fraction=0.95)
        self.assertEqual(selected["selected_batch_size"], 2)

    def test_batch_policy_unavailable_without_valid_above_one(self) -> None:
        selected = selection_from_benchmark(
            [{"batch_size": 1, "status": "PASSED", "correctness": {"passed": True}, "audio_seconds_per_wall_second": 1.0}],
            within_best_fraction=0.95,
        )
        self.assertIsNone(selected["selected_batch_size"])

    def test_public_report_rejects_raw_ids_text_and_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            assert_public_report_safe({"sample_id": "selected-001"})
        with self.assertRaisesRegex(ValueError, "row IDs"):
            assert_public_report_safe({"safe": "gamsv2-cell01-a00-o001"})
        with self.assertRaisesRegex(ValueError, "local paths"):
            assert_public_report_safe({"safe": "/home/example/file"})

    def test_pre_merge_integrity_detects_base_parameter_change(self) -> None:
        base = {"encoder.weight": torch.zeros(2, 2)}
        current = {"encoder.weight": torch.ones(2, 2)}
        selection = PromptColumnSelection(
            prompt_name="sl-SI",
            prompt_index=2,
            encoder_width=3,
            num_prompts=4,
            selected_column=5,
            first_linear_name="prompt_kernel.0",
            first_linear_shape=(5, 7),
            effective_trainable_parameters=5,
        )
        report = parameter_integrity_before_merge(base, current, selection=selection)
        self.assertFalse(report["pretrained_tensors_identical"])
        self.assertEqual(report["changed_pretrained_tensors"], ["encoder.weight"])

    def test_certificate_head_check_requires_exact_head_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            path = root / "cert.json"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch("slaif_asr.corpus_v2_training.REPO_ROOT", root), mock.patch(
                "slaif_asr.corpus_v2_training.subprocess.run"
            ) as run_mock, mock.patch("slaif_asr.corpus_v2_training.git_show_head", return_value=b"{}\n"):
                run_mock.return_value = mock.Mock(stdout="")
                result = git_tracked_and_clean_at_head(path)
                self.assertTrue(result["matches_head"])
            with mock.patch("slaif_asr.corpus_v2_training.REPO_ROOT", root), mock.patch(
                "slaif_asr.corpus_v2_training.subprocess.run"
            ) as run_mock, mock.patch("slaif_asr.corpus_v2_training.git_show_head", return_value=b"{\"stale\": true}\n"):
                run_mock.return_value = mock.Mock(stdout="")
                with self.assertRaisesRegex(RuntimeError, "differs from HEAD"):
                    git_tracked_and_clean_at_head(path)


if __name__ == "__main__":
    unittest.main()
