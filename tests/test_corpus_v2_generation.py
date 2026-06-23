from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.corpus_v2_generation import (
    build_prompt,
    build_record,
    config_sha256,
    extract_utterance_lines,
    filter_records,
    load_config,
    prompt_contains_forbidden_identifier,
    review_template_rows,
    write_public_reports,
    write_review_outputs,
)
from slaif_asr.data_quality import atomic_write_jsonl, load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/generation/slovenian_corpus_v2_candidate_reservoir.json"


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "left"
    chat_template = None

    def __call__(self, prompts, **kwargs):
        max_len = max(len(prompt.split()) for prompt in prompts)
        input_ids = []
        attention_mask = []
        for prompt in prompts:
            length = len(prompt.split())
            pad = max_len - length
            input_ids.append([0] * pad + list(range(10, 10 + length)))
            attention_mask.append([0] * pad + [1] * length)
        return {
            "input_ids": FakeTensor(input_ids),
            "attention_mask": FakeTensor(attention_mask),
        }


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def to(self, _device):
        return self


class CorpusV2GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.cell = self.config["prompt_cells"][0]

    def test_prompt_forbids_metadata_identifiers(self) -> None:
        prompt = build_prompt(self.cell, requested_rows=40)
        self.assertIn("Ne uporabljaj oštevilčenja", prompt)
        self.assertIn("postaj", prompt)
        self.assertFalse(prompt_contains_forbidden_identifier(prompt))
        self.assertNotIn(self.cell["cell_id"], prompt)
        self.assertNotIn(self.cell["source_family_id"], prompt)

    def test_schema_20_row_construction_and_ids_out_of_text(self) -> None:
        text = "Danes se dobimo pred knjižnico."
        row = build_record(
            config=self.config,
            cell=self.cell,
            attempt_index=0,
            output_ordinal=3,
            text=text,
            extraction_mode="line",
        )
        self.assertEqual(row["schema_version"], "2.0")
        self.assertEqual(row["language"], "sl-SI")
        self.assertEqual(row["partition_role"], "synthetic_candidate")
        self.assertEqual(row["source_type"], "generated_text")
        self.assertEqual(row["spoken_text"], row["target_text"])
        self.assertEqual(row["template_family_id"], None)
        self.assertEqual(row["utterance_family_id"], row["candidate_id"])
        self.assertEqual(row["minimal_pair"], None)
        self.assertNotIn(row["candidate_id"], row["spoken_text"])
        self.assertNotIn(str(row["generation"]["prompt_cell"]), row["spoken_text"])

    def test_deterministic_candidate_and_family_ids(self) -> None:
        left = build_record(config=self.config, cell=self.cell, attempt_index=1, output_ordinal=7, text="V parku je danes mirno.", extraction_mode="line")
        right = build_record(config=self.config, cell=self.cell, attempt_index=1, output_ordinal=7, text="V parku je danes mirno.", extraction_mode="line")
        other = build_record(config=self.config, cell=self.cell, attempt_index=1, output_ordinal=8, text="V parku je danes mirno.", extraction_mode="line")
        self.assertEqual(left["candidate_id"], right["candidate_id"])
        self.assertNotEqual(left["candidate_id"], other["candidate_id"])
        self.assertEqual(left["source_family_id"], self.cell["source_family_id"])

    def test_batched_prompt_tokenization_has_attention_mask_padding(self) -> None:
        from scripts.generate_gams_corpus_v2 import encode_prompts

        tokenizer = FakeTokenizer()
        encoded = encode_prompts(tokenizer, ["kratek poziv", "daljši testni poziv"], max_context_tokens=32)
        self.assertIn("attention_mask", encoded)
        self.assertEqual(encoded["input_ids"].shape, encoded["attention_mask"].shape)
        self.assertEqual(encoded["attention_mask"].rows[0][0], 0)
        self.assertEqual(encoded["attention_mask"].rows[1][0], 1)

    def test_output_parser_removes_numbering_and_keeps_plain_lines(self) -> None:
        raw = "Dober dan vsem skupaj.\n1. To je oštevilčeno.\n- Tudi to je alineja.\nJutri bo sestanek v sejni sobi."
        lines, rejected = extract_utterance_lines(raw, cell_id="cell01", attempt_id="cell01-attempt-00")
        self.assertEqual(
            [line.text for line in lines],
            [
                "Dober dan vsem skupaj.",
                "To je oštevilčeno.",
                "Tudi to je alineja.",
                "Jutri bo sestanek v sejni sobi.",
            ],
        )
        self.assertEqual(sum(1 for item in rejected if item.reason == "parser_numbering_or_bullet"), 0)

    def test_duplicate_and_metadata_leak_rejection(self) -> None:
        records = [
            build_record(config=self.config, cell=self.cell, attempt_index=0, output_ordinal=1, text="Danes je lepo vreme.", extraction_mode="line"),
            build_record(config=self.config, cell=self.cell, attempt_index=0, output_ordinal=2, text="Danes je lepo vreme.", extraction_mode="line"),
            build_record(config=self.config, cell=self.cell, attempt_index=0, output_ordinal=3, text="Skupina 7 pride jutri.", extraction_mode="line"),
        ]
        retained, rejected, _ = filter_records(records, config=self.config)
        self.assertEqual(len(retained), 1)
        reasons = {item.reason for item in rejected}
        self.assertIn("surface_duplicate", reasons)
        self.assertIn("metadata_leak", reasons)

    def test_repeated_ngram_and_fuzzy_rejection(self) -> None:
        config = dict(self.config)
        config["filtering"] = {
            **self.config["filtering"],
            "max_repeated_token_ngram_count": 1,
            "token_jaccard_reject_threshold": 0.72,
            "character_jaccard_reject_threshold": 0.8,
        }
        records = [
            build_record(config=config, cell=self.cell, attempt_index=0, output_ordinal=1, text="Ali lahko danes odpreš okno?", extraction_mode="line"),
            build_record(config=config, cell=self.cell, attempt_index=0, output_ordinal=2, text="Ali lahko jutri odpreš vrata?", extraction_mode="line"),
            build_record(config=config, cell=self.cell, attempt_index=0, output_ordinal=3, text="V mestu je danes miren večer.", extraction_mode="line"),
            build_record(config=config, cell=self.cell, attempt_index=0, output_ordinal=4, text="V mestu je danes mirno jutro.", extraction_mode="line"),
        ]
        retained, rejected, _ = filter_records(records, config=config)
        self.assertLess(len(retained), len(records))
        reasons = {item.reason for item in rejected}
        self.assertTrue({"token_ngram_concentration", "fuzzy_similarity_candidate"} & reasons)

    def test_bounded_retry_shortfall_is_visible(self) -> None:
        from scripts.generate_gams_corpus_v2 import planned_prompts_for_attempt

        accepted = {"cell01": 0}
        attempt0 = planned_prompts_for_attempt(self.config, accepted, 0)
        self.assertEqual(len(attempt0), 12)
        accepted = {cell["cell_id"]: 40 for cell in self.config["prompt_cells"]}
        self.assertEqual(planned_prompts_for_attempt(self.config, accepted, 1), [])

    def test_review_template_has_one_row_per_candidate_and_no_accept(self) -> None:
        rows = [
            build_record(config=self.config, cell=self.cell, attempt_index=0, output_ordinal=1, text="Danes je lepo vreme.", extraction_mode="line"),
            build_record(config=self.config, cell=self.cell, attempt_index=0, output_ordinal=2, text="Jutri pride nova pošiljka.", extraction_mode="line"),
        ]
        template = review_template_rows(rows)
        self.assertEqual(len(template), 2)
        self.assertTrue(all(row["outcome"] == "" for row in template))
        self.assertTrue(all(row["minimal_pair_approved"] is False for row in template))

    def test_public_summary_contains_no_raw_text_or_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = dict(self.config)
            config["run_directory"] = str(tmp / "run")
            config["public_reports"] = {
                "json": str(tmp / "report.json"),
                "markdown": str(tmp / "report.md"),
            }
            run_dir = Path(config["run_directory"])
            run_dir.mkdir(parents=True)
            rows = [
                build_record(config=config, cell=self.cell, attempt_index=0, output_ordinal=1, text="Danes je lep dan.", extraction_mode="line")
            ]
            atomic_write_jsonl(run_dir / "generated-all.local.jsonl", rows)
            atomic_write_jsonl(run_dir / "pre-review-candidates.local.jsonl", rows)
            write_review_outputs(rows, config)
            payload = write_public_reports(config)
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("Danes je lep dan", serialized)
            self.assertNotIn(str(tmp), serialized)

    def test_old_line_harness_behavior_remains_available(self) -> None:
        from slaif_asr.gams import extract_candidate_text_lines

        self.assertEqual(extract_candidate_text_lines("1. Prvi stavek.\n2. Drugi stavek."), ["Prvi stavek.", "Drugi stavek."])

    def test_config_hash_is_deterministic(self) -> None:
        self.assertEqual(config_sha256(self.config), config_sha256(load_config(CONFIG_PATH)))


if __name__ == "__main__":
    unittest.main()
