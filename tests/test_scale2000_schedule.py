from __future__ import annotations

import unittest
from pathlib import Path

from slaif_asr.gams_retry_controller import AttemptTask
from slaif_asr.scale2000_corpus import build_new_record, build_scale2000_exposure_schedule, load_scale2000_generation_config
from slaif_asr.scale200_corpus import build_record, load_augmentation_config, load_generation_config


class Scale2000ScheduleTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        anchor = load_generation_config(Path("configs/generation/gams_corpus_v3_1600_v1.json"))
        config = load_scale2000_generation_config(Path("configs/generation/gams_corpus_v4_16000_v1.json"))
        rows = []
        for cell in anchor["prompt_cells"]:
            for ordinal in range(1, 41):
                rows.append(
                    build_record(
                        config=anchor,
                        cell=cell,
                        attempt_index=0,
                        output_ordinal=ordinal,
                        text=f"Podedovana poved za urnik {cell['cell_id']} {ordinal}.",
                    )
                )
        for cell in config["prompt_cells"]:
            for ordinal in range(1, 361):
                task = AttemptTask(str(cell["cell_id"]), f"shard{((ordinal - 1) % 9) + 1:02d}", ordinal // 61, 0, 60, ordinal, "fixture")
                rows.append(
                    build_new_record(
                        config=config,
                        cell=cell,
                        task=task,
                        output_ordinal=ordinal,
                        text=f"Nova poved za urnik {cell['cell_id']} {ordinal}.",
                    )
                )
        return rows

    def test_exact_scale2000_schedule(self) -> None:
        augmentation = load_augmentation_config(Path("configs/augmentation/scale200_transcript_preserving_v1.json"))
        schedule, summary = build_scale2000_exposure_schedule(self._rows(), augmentation)
        self.assertEqual(len(schedule), 320000)
        self.assertEqual(summary["optimizer_steps"], 40000)
        self.assertEqual(summary["heldout_voice_exposures"], {"supertonic-M5": 0, "supertonic-F5": 0})
        for profile in augmentation["augmentation_profiles"]:
            self.assertEqual(summary["augmentation_profile_counts"][profile["profile_id"]], 16000)
        for voice in augmentation["clean_voices"]:
            self.assertGreater(summary["clean_voice_counts"][voice], 0)

    def test_every_round_has_every_semantic_item_once(self) -> None:
        augmentation = load_augmentation_config(Path("configs/augmentation/scale200_transcript_preserving_v1.json"))
        schedule, _summary = build_scale2000_exposure_schedule(self._rows(), augmentation)
        by_round: dict[int, set[str]] = {idx: set() for idx in range(1, 21)}
        for row in schedule:
            key = row["semantic_key"]
            self.assertNotIn(key, by_round[row["round"]])
            by_round[row["round"]].add(key)
        self.assertTrue(all(len(values) == 16000 for values in by_round.values()))


if __name__ == "__main__":
    unittest.main()

