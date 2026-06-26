from __future__ import annotations

import unittest
from pathlib import Path

from slaif_asr.gams_retry_controller import AttemptTask
from slaif_asr.scale2000_corpus import (
    build_combined_rows,
    build_new_record,
    build_task_prompt,
    counts_by_cell,
    load_scale2000_generation_config,
    scale2000_multiplier_table,
    select_new_rows,
    verify_prompt_cells_match_anchor,
)
from slaif_asr.scale200_corpus import build_record, load_generation_config, stable_sha256


CONFIG_PATH = Path("configs/generation/gams_corpus_v4_16000_v1.json")
ANCHOR_CONFIG_PATH = Path("configs/generation/gams_corpus_v3_1600_v1.json")


class Scale2000CorpusTests(unittest.TestCase):
    def _word(self, prefix: str, index: int) -> str:
        alphabet = "abcdefghijklmnoprstuvz"
        value = index
        letters = []
        for _ in range(5):
            letters.append(alphabet[value % len(alphabet)])
            value //= len(alphabet)
        return prefix + "".join(letters)

    def test_config_reuses_exact_forty_prompt_cells(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        verify_prompt_cells_match_anchor(config)
        self.assertEqual(len(config["prompt_cells"]), 40)
        self.assertEqual(config["shards_per_cell"], 9)
        self.assertEqual(config["new_rows_per_cell"], 360)

    def test_build_new_record_uses_v4_identity_and_no_text_id(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        cell = config["prompt_cells"][0]
        task = AttemptTask("cell01", "shard03", 2, 1, 60, 123, "targeted_refill")
        row = build_new_record(config=config, cell=cell, task=task, output_ordinal=7, text="Danes grem mirno domov.")
        self.assertTrue(row["candidate_id"].startswith("gamsv4-cell01-shard03-a02-o007"))
        self.assertEqual(row["partition_role"], "selected_training")
        self.assertEqual(row["spoken_text"], row["target_text"])
        self.assertNotIn(row["candidate_id"], row["spoken_text"])
        self.assertEqual(row["generation"]["generation_shard"], "shard03")

    def test_prompt_does_not_leak_shard_or_attempt(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        task = AttemptTask("cell01", "shard03", 2, 1, 60, 123, "targeted_refill", ("vary clause structure",))
        prompt = build_task_prompt(config, task)
        self.assertNotIn("shard03", prompt)
        self.assertNotIn(task.attempt_id, prompt)
        self.assertIn("vary clause structure", prompt)

    def _inherited_rows(self) -> list[dict]:
        anchor = load_generation_config(ANCHOR_CONFIG_PATH)
        rows = []
        for cell in anchor["prompt_cells"]:
            for ordinal in range(1, 41):
                index = int(str(cell["cell_id"]).replace("cell", "")) * 1000 + ordinal
                rows.append(
                    build_record(
                        config=anchor,
                        cell=cell,
                        attempt_index=0,
                        output_ordinal=ordinal,
                        text=(
                            f"{self._word('ank', index)} {self._word('bnk', index)} "
                            f"{self._word('cnk', index)} {self._word('dnk', index)}."
                        ),
                    )
                )
        return rows

    def _new_rows(self, per_cell: int = 400) -> list[dict]:
        config = load_scale2000_generation_config(CONFIG_PATH)
        rows = []
        for cell in config["prompt_cells"]:
            for ordinal in range(1, per_cell + 1):
                index = int(str(cell["cell_id"]).replace("cell", "")) * 1000 + ordinal
                task = AttemptTask(str(cell["cell_id"]), f"shard{((ordinal - 1) % 9) + 1:02d}", ordinal // 61, 0, 60, ordinal, "fixture")
                rows.append(
                    build_new_record(
                        config=config,
                        cell=cell,
                        task=task,
                        output_ordinal=ordinal,
                        text=(
                            f"{self._word('eno', index)} {self._word('dve', index)} "
                            f"{self._word('tri', index)} {self._word('sti', index)}."
                        ),
                    )
                )
        return rows

    def test_selects_exactly_360_new_rows_per_cell(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        inherited = self._inherited_rows()
        selected, summary = select_new_rows(self._new_rows(), inherited_rows=inherited, config=config)
        self.assertEqual(len(selected), 14400)
        self.assertTrue(all(count == 360 for count in summary["new_rows_per_cell"].values()))
        selected_again, _ = select_new_rows(list(reversed(self._new_rows())), inherited_rows=inherited, config=config)
        self.assertEqual([row["candidate_id"] for row in selected], [row["candidate_id"] for row in selected_again])

    def test_new_row_surplus_shortfall_fails(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        with self.assertRaisesRegex(RuntimeError, "surplus shortfall"):
            select_new_rows(self._new_rows(per_cell=399), inherited_rows=self._inherited_rows(), config=config)

    def test_build_combined_rows_preserves_nesting_counts(self) -> None:
        config = load_scale2000_generation_config(CONFIG_PATH)
        combined, summary = build_combined_rows(self._inherited_rows(), self._new_rows(), config=config)
        self.assertEqual(len(combined), 16000)
        self.assertEqual(summary["inherited_rows"], 1600)
        self.assertEqual(summary["new_rows"], 14400)
        self.assertTrue(all(count == 400 for count in counts_by_cell(combined).values()))
        inherited_ids = {row["candidate_id"] for row in self._inherited_rows()}
        self.assertTrue(inherited_ids.issubset({row["candidate_id"] for row in combined}))

    def test_multiplier_table(self) -> None:
        table = scale2000_multiplier_table()
        self.assertEqual(table["scale2000_semantic_items"], 16000)
        self.assertEqual(table["total_view_records"], 320000)
        self.assertEqual(table["optimizer_steps"], 40000)
        self.assertIn("not independent linguistic information", table["interpretation"])


if __name__ == "__main__":
    unittest.main()
