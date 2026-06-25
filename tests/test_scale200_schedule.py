from __future__ import annotations

import unittest
from pathlib import Path

from slaif_asr.scale200_corpus import build_exposure_schedule, build_record, load_augmentation_config, load_generation_config


class Scale200ScheduleTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        config = load_generation_config(Path("configs/generation/gams_corpus_v3_1600_v1.json"))
        rows = []
        for cell in config["prompt_cells"]:
            for ordinal in range(1, 41):
                rows.append(
                    build_record(
                        config=config,
                        cell=cell,
                        attempt_index=0,
                        output_ordinal=ordinal,
                        text=f"Urnik vsebuje testno poved {cell['cell_id']} {ordinal}.",
                    )
                )
        return rows

    def test_exact_200x_schedule(self) -> None:
        augmentation = load_augmentation_config(Path("configs/augmentation/scale200_transcript_preserving_v1.json"))
        schedule, summary = build_exposure_schedule(self._rows(), augmentation)
        self.assertEqual(len(schedule), 32000)
        self.assertEqual(summary["optimizer_steps"], 4000)
        self.assertEqual(summary["heldout_voice_exposures"], {"supertonic-M5": 0, "supertonic-F5": 0})
        for profile in augmentation["augmentation_profiles"]:
            self.assertEqual(summary["augmentation_profile_counts"][profile["profile_id"]], 1600)
        for voice in augmentation["clean_voices"]:
            self.assertGreater(summary["clean_voice_counts"][voice], 0)

    def test_every_round_has_every_semantic_item_once(self) -> None:
        augmentation = load_augmentation_config(Path("configs/augmentation/scale200_transcript_preserving_v1.json"))
        schedule, _summary = build_exposure_schedule(self._rows(), augmentation)
        by_round: dict[int, set[str]] = {idx: set() for idx in range(1, 21)}
        for row in schedule:
            key = row["semantic_key"]
            self.assertNotIn(key, by_round[row["round"]])
            by_round[row["round"]].add(key)
        self.assertTrue(all(len(values) == 1600 for values in by_round.values()))


if __name__ == "__main__":
    unittest.main()
