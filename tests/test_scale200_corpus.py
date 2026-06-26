from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slaif_asr.scale200_corpus import (
    build_prompt,
    build_record,
    extract_utterance_lines,
    filter_records,
    load_generation_config,
    prompt_contains_forbidden_identifier,
    select_fixed_rows,
)


CONFIG_PATH = Path("configs/generation/gams_corpus_v3_1600_v1.json")


class Scale200CorpusTests(unittest.TestCase):
    def test_config_has_forty_cells_and_safe_prompts(self) -> None:
        config = load_generation_config(CONFIG_PATH)
        self.assertEqual(len(config["prompt_cells"]), 40)
        for cell in config["prompt_cells"]:
            prompt = build_prompt(cell, requested_rows=cell["requested_rows"])
            self.assertFalse(prompt_contains_forbidden_identifier(prompt))
            self.assertIn("Brez oštevilčenja", prompt)
            self.assertNotIn(str(cell["source_family_id"]), prompt)

    def test_parse_lines_removes_numbering_and_rejects_markup(self) -> None:
        lines, rejected = extract_utterance_lines(
            "1. Lahko prideš jutri?\n```json\n{\"x\": 1}\n```\n- Prinesi mapo.",
            cell_id="cell01",
            attempt_id="cell01-attempt-00",
        )
        self.assertEqual([line.text for line in lines], ["Lahko prideš jutri?", "Prinesi mapo."])
        self.assertGreaterEqual(len(rejected), 2)

    def test_build_record_uses_schema_2_without_text_ids(self) -> None:
        config = load_generation_config(CONFIG_PATH)
        cell = config["prompt_cells"][0]
        row = build_record(config=config, cell=cell, attempt_index=0, output_ordinal=7, text="Danes grem po kruh.")
        self.assertEqual(row["schema_version"], "2.0")
        self.assertEqual(row["partition_role"], "selected_training")
        self.assertEqual(row["spoken_text"], row["target_text"])
        self.assertNotIn(row["candidate_id"], row["spoken_text"])
        self.assertIsNone(row["template_family_id"])
        self.assertEqual(row["utterance_family_id"], row["candidate_id"])

    def test_filter_rejects_duplicates_and_metadata(self) -> None:
        config = load_generation_config(CONFIG_PATH)
        cell = config["prompt_cells"][0]
        rows = [
            build_record(config=config, cell=cell, attempt_index=0, output_ordinal=1, text="Danes grem po kruh."),
            build_record(config=config, cell=cell, attempt_index=0, output_ordinal=2, text="Danes grem po kruh."),
            build_record(config=config, cell=cell, attempt_index=0, output_ordinal=3, text="Kandidat 17 gre v trgovino."),
        ]
        retained, rejected, summary = filter_records(rows, config=config)
        self.assertEqual(len(retained), 1)
        self.assertEqual(summary["reason_counts"]["surface_duplicate"], 1)
        self.assertEqual(summary["reason_counts"]["metadata_leak"], 1)
        self.assertEqual(len(rejected), 2)

    def test_selects_exactly_40_per_cell_deterministically(self) -> None:
        config = load_generation_config(CONFIG_PATH)
        rows = []
        for cell in config["prompt_cells"]:
            for ordinal in range(1, 46):
                rows.append(
                    build_record(
                        config=config,
                        cell=cell,
                        attempt_index=0,
                        output_ordinal=ordinal,
                        text=f"To je varna testna poved {cell['cell_id']} {ordinal}.",
                    )
                )
        first, summary = select_fixed_rows(rows, config=config)
        second, _summary = select_fixed_rows(list(reversed(rows)), config=config)
        self.assertEqual(len(first), 1600)
        self.assertEqual(summary["fixed_rows"], 1600)
        self.assertEqual([row["candidate_id"] for row in first], [row["candidate_id"] for row in second])

    def test_select_shortfall_fails(self) -> None:
        config = load_generation_config(CONFIG_PATH)
        rows = []
        for cell in config["prompt_cells"]:
            limit = 39 if cell["cell_id"] == "cell01" else 40
            for ordinal in range(1, limit + 1):
                rows.append(
                    build_record(
                        config=config,
                        cell=cell,
                        attempt_index=0,
                        output_ordinal=ordinal,
                        text=f"To je kratka testna poved {cell['cell_id']} {ordinal}.",
                    )
                )
        with self.assertRaisesRegex(RuntimeError, "shortfall"):
            select_fixed_rows(rows, config=config)


if __name__ == "__main__":
    unittest.main()
