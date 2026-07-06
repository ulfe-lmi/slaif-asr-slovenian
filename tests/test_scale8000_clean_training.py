import tempfile
import unittest
from pathlib import Path

from slaif_asr.data_quality import sha256_file
from slaif_asr.scale8000_clean_training import (
    CLEAN_VOICE_ORDER,
    build_clean_exposure_schedule,
    microbatch_plan,
    validate_clean_exposure_schedule,
)


def fake_bank(count=64000):
    class Row:
        def __init__(self, key, voice):
            self.semantic_key = key
            self.voice = voice
            self.audio_sha256 = f"audio-{key}-{voice}"
            self.text_sha256 = f"text-{key}"
            self.duration = 1.0

    bank = {}
    for index in range(count):
        key = f"row-{index:05d}"
        bank[key] = {voice: Row(key, voice) for voice in CLEAN_VOICE_ORDER}
    return bank


class Scale8000CleanTrainingTests(unittest.TestCase):
    def test_exact_clean_schedule_counts(self):
        config = {"training": {"semantic_rows": 64000}}
        schedule, summary = build_clean_exposure_schedule(config, fake_bank())
        self.assertEqual(len(schedule), 576000)
        self.assertEqual(summary["rounds"], 9)
        self.assertEqual(summary["optimizer_steps"], 72000)
        self.assertEqual(summary["augmented_views"], 0)
        self.assertEqual(summary["heldout_voice_exposures"]["supertonic-M5"], 0)
        self.assertTrue(all(summary["voice_counts"][voice] == 64000 for voice in CLEAN_VOICE_ORDER))

    def test_rejects_hidden_augmentation(self):
        schedule, _summary = build_clean_exposure_schedule({"training": {"semantic_rows": 64000}}, fake_bank())
        schedule[0] = {**schedule[0], "view_type": "augmented"}
        with self.assertRaises(ValueError):
            validate_clean_exposure_schedule(schedule)

    def test_rejects_heldout_voice(self):
        schedule, _summary = build_clean_exposure_schedule({"training": {"semantic_rows": 64000}}, fake_bank())
        schedule[0] = {**schedule[0], "voice": "supertonic-M5"}
        with self.assertRaises(ValueError):
            validate_clean_exposure_schedule(schedule)

    def test_microbatch_plan_preserves_effective_batch(self):
        self.assertEqual(microbatch_plan(1), {"physical_microbatch": 1, "gradient_accumulation_steps": 8, "effective_batch_size": 8})
        self.assertEqual(microbatch_plan(2)["gradient_accumulation_steps"], 4)
        self.assertEqual(microbatch_plan(8)["gradient_accumulation_steps"], 1)


if __name__ == "__main__":
    unittest.main()
