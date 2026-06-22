from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.gams import (
    build_candidates_from_text_lines,
    extract_candidate_text_lines,
    load_generation_config,
    parse_strict_json_candidates,
    protected_hash,
    validate_candidate_batch,
    validate_gams_candidate,
)


class GamsCandidateTests(unittest.TestCase):
    def row(self) -> dict:
        return {
            "candidate_id": "round1-0001",
            "spoken_text": "Čez cesto gre danes moder avtomobil.",
            "target_text": "Čez cesto gre danes moder avtomobil.",
            "language": "sl-SI",
            "phenomena": ["diacritic:č", "ordinary"],
            "source_error_clusters": [],
            "generation_seed": 1234,
        }

    def test_strict_json_generation_parsing(self) -> None:
        rows = parse_strict_json_candidates(json.dumps([self.row()], ensure_ascii=False))
        self.assertEqual(rows[0]["candidate_id"], "round1-0001")
        with self.assertRaisesRegex(ValueError, "Markdown"):
            parse_strict_json_candidates("```json\n[]\n```")

    def test_text_line_harness_extracts_numbered_sentences(self) -> None:
        raw = """
        Tukaj so stavki:
        1. Danes je v Ljubljani miren večer.
        2) Prosim, preveri čisto škatlo na mizi.
        - To ni seznam v obliki JSON, ampak je uporaben slovenski stavek.
        """
        lines = extract_candidate_text_lines(raw)
        self.assertEqual(lines[0], "Danes je v Ljubljani miren večer.")
        valid, rejected = build_candidates_from_text_lines(lines, round_id="round1", generation_seed=1234)
        self.assertGreaterEqual(len(valid), 2)
        self.assertFalse(rejected)
        self.assertEqual(valid[0].candidate_id, "round1-0001")
        self.assertEqual(valid[0].spoken_text, valid[0].target_text)

    def test_text_line_harness_rejects_duplicate_lines(self) -> None:
        lines = [
            "Danes je v Ljubljani miren večer.",
            "Danes je v Ljubljani miren večer.",
        ]
        valid, rejected = build_candidates_from_text_lines(lines, round_id="round1", generation_seed=1234)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("duplicate", rejected[0])

    def test_valid_candidate_passes(self) -> None:
        candidate = validate_gams_candidate(self.row())
        self.assertEqual(candidate.language, "sl-SI")

    def test_malformed_candidate_rejected(self) -> None:
        row = self.row()
        row["language"] = "sl"
        with self.assertRaisesRegex(ValueError, "language must be sl-SI"):
            validate_gams_candidate(row)

    def test_protected_real_gate_overlap_rejected(self) -> None:
        row = self.row()
        with self.assertRaisesRegex(ValueError, "protected evaluation"):
            validate_gams_candidate(row, protected_hashes={protected_hash(row["spoken_text"])})

    def test_duplicate_and_near_duplicate_rejection(self) -> None:
        first = self.row()
        second = self.row()
        second["candidate_id"] = "round1-0002"
        second["spoken_text"] = "Čez cesto gre danes moder avtomobil!"
        second["target_text"] = "Čez cesto gre danes moder avtomobil!"
        valid, rejected = validate_candidate_batch([first, second])
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("near-duplicate", rejected[0])

    def test_generation_config_is_pinned_and_gpu_only(self) -> None:
        config = load_generation_config(Path("configs/generation/gams_prompt_curriculum.json"))
        self.assertEqual(config["primary_model"]["revision"], "1d0b27af5748784482600d24779409e7e1dc9adc")
        self.assertTrue(config["quantization"]["load_in_4bit"])
        self.assertFalse(config["device_policy"]["cpu_offload"])

    def test_generation_config_rejects_cpu_offload(self) -> None:
        config = json.loads(Path("configs/generation/gams_prompt_curriculum.json").read_text(encoding="utf-8"))
        config["device_policy"]["cpu_offload"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CPU offload"):
                load_generation_config(path)


if __name__ == "__main__":
    unittest.main()
