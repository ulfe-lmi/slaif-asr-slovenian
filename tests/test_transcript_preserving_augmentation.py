from __future__ import annotations

import unittest
from pathlib import Path

from slaif_asr.scale200_corpus import load_augmentation_config
from slaif_asr.transcript_preserving_augmentation import assignment_for, db_to_linear, parameters_for_profile


class TranscriptPreservingAugmentationTests(unittest.TestCase):
    def test_all_profiles_have_deterministic_parameters(self) -> None:
        config = load_augmentation_config(Path("configs/augmentation/scale200_transcript_preserving_v1.json"))
        for profile in config["augmentation_profiles"]:
            first = parameters_for_profile(profile, semantic_key="semantic-001")
            second = parameters_for_profile(profile, semantic_key="semantic-001")
            self.assertEqual(first, second)
            self.assertTrue(first)

    def test_assignment_rotates_across_clean_voices(self) -> None:
        voices = ("piper", "s1", "s2")
        assignments = [
            assignment_for(
                semantic_key=f"row-{idx}",
                semantic_position=idx,
                profile_id="noise",
                profile_index=2,
                clean_voices=voices,
            ).source_voice
            for idx in range(9)
        ]
        self.assertEqual(assignments.count("piper"), 3)
        self.assertEqual(assignments.count("s1"), 3)
        self.assertEqual(assignments.count("s2"), 3)

    def test_gain_conversion(self) -> None:
        self.assertAlmostEqual(db_to_linear(0.0), 1.0)
        self.assertGreater(db_to_linear(6.0), 1.0)
        self.assertLess(db_to_linear(-6.0), 1.0)


if __name__ == "__main__":
    unittest.main()
