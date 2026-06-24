from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slaif_asr.corpus_v2_scoring import (
    CorpusScoringRecord,
    aggregate_scoring_summary,
    assert_public_scoring_payload_safe,
    build_per_row_scores,
    verify_scoring_authorization,
)
from slaif_asr.corpus_v2_selection import (
    SELECTED_CERTIFICATE_STATUS,
    SelectionCandidate,
    assert_public_selection_payload_safe,
    local_selected_rows,
    select_controls,
    select_hard_examples,
)
from slaif_asr.data_quality import atomic_write_json, sha256_file


def scoring_record(sample_id: str, reference: str, *, index: int = 0) -> CorpusScoringRecord:
    return CorpusScoringRecord(
        sample_id=sample_id,
        audio_filepath=f"/ignored/{sample_id}.wav",
        duration=1.0 + index,
        reference=reference,
        original_index=index,
        text_sha256=f"text-{sample_id}",
        audio_sha256=f"audio-{sample_id}",
        source_id=f"source-{sample_id}",
        source_family_id=f"source-family-{index}",
        utterance_family_id=f"utterance-{sample_id}",
        discovered_template_family=f"dtf-{sample_id}",
        domain="fixture-domain",
        phenomena=("fixture",),
        prompt_cell=f"cell{index % 4}",
        row={},
        audio_row={"candidate_id": sample_id, "audio_filepath": f"/ignored/{sample_id}.wav"},
    )


def candidate(
    sample_id: str,
    *,
    wer: float,
    cer: float,
    empty: bool = False,
    duration: float = 1.0,
    domain: str = "domain-a",
    source_family: str | None = None,
    utterance_family: str | None = None,
    discovered_family: str | None = None,
    prompt_cell: str = "cell01",
) -> SelectionCandidate:
    return SelectionCandidate(
        sample_id=sample_id,
        reference=f"referenca {sample_id}",
        audio_filepath=f"/ignored/{sample_id}.wav",
        duration_seconds=duration,
        domain=domain,
        phenomena=("ordinary",),
        prompt_cell=prompt_cell,
        source_id=f"source-{sample_id}",
        source_family_id=source_family or f"source-family-{sample_id}",
        utterance_family_id=utterance_family or f"utterance-{sample_id}",
        discovered_template_family=discovered_family or f"dtf-{sample_id}",
        text_sha256=f"text-{sample_id}",
        audio_sha256=f"audio-{sample_id}",
        normalized_wer=wer,
        normalized_cer=cer,
        empty_hypothesis=empty,
        deletion_rate=0.0,
        row={"normalized_wer": wer, "normalized_cer": cer, "duration_seconds": duration},
        audio_row={"candidate_id": sample_id, "audio_filepath": f"/ignored/{sample_id}.wav"},
    )


class CorpusV2ScoringSelectionTests(unittest.TestCase):
    def test_scoring_authorization_requires_expected_hash_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "auth.json"
            atomic_write_json(path, {"certificate_id": "sl-corpus-v2-scoring-authorization-v1", "status": "SCORING_AUTHORIZED"})
            digest = sha256_file(path)
            with mock.patch("slaif_asr.corpus_v2_scoring.SCORING_AUTHORIZATION_PATH", path), mock.patch(
                "slaif_asr.corpus_v2_scoring.EXPECTED_SCORING_AUTHORIZATION_SHA256", digest
            ):
                self.assertEqual(verify_scoring_authorization(require_status="SCORING_AUTHORIZED")["sha256"], digest)
            with mock.patch("slaif_asr.corpus_v2_scoring.SCORING_AUTHORIZATION_PATH", path), mock.patch(
                "slaif_asr.corpus_v2_scoring.EXPECTED_SCORING_AUTHORIZATION_SHA256", "0" * 64
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    verify_scoring_authorization(require_status="SCORING_AUTHORIZED")

    def test_wrong_authorization_status_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "auth.json"
            atomic_write_json(path, {"certificate_id": "sl-corpus-v2-scoring-authorization-v1", "status": "DRAFT"})
            digest = sha256_file(path)
            with mock.patch("slaif_asr.corpus_v2_scoring.SCORING_AUTHORIZATION_PATH", path), mock.patch(
                "slaif_asr.corpus_v2_scoring.EXPECTED_SCORING_AUTHORIZATION_SHA256", digest
            ):
                with self.assertRaisesRegex(RuntimeError, "status"):
                    verify_scoring_authorization(require_status="SCORING_AUTHORIZED")

    def test_prediction_association_rejects_missing_prediction(self) -> None:
        records = [scoring_record("row-a", "ena dva", index=0), scoring_record("row-b", "tri štiri", index=1)]
        with self.assertRaisesRegex(RuntimeError, "prediction ID mismatch"):
            build_per_row_scores(records, {"row-a": "ena"})

    def test_prediction_association_rejects_unexpected_prediction(self) -> None:
        records = [scoring_record("row-a", "ena dva", index=0)]
        with self.assertRaisesRegex(RuntimeError, "prediction ID mismatch"):
            build_per_row_scores(records, {"row-a": "ena dva", "row-x": "odveč"})

    def test_corpus_metrics_use_summed_edit_counts(self) -> None:
        records = [
            scoring_record("short", "ena", index=0),
            scoring_record("long", "ena dva tri štiri pet", index=1),
        ]
        per_rows = build_per_row_scores(records, {"short": "", "long": "ena dva tri štiri pet"})
        summary = aggregate_scoring_summary(per_rows)
        self.assertEqual(summary["metrics"]["raw"]["corpus_wer"], 16.667)
        self.assertEqual(summary["metrics"]["raw"]["mean_utterance_wer"], 50.0)

    def test_empty_hypothesis_has_hard_score_priority(self) -> None:
        easy_empty = candidate("empty", wer=0.0, cer=0.0, empty=True)
        hard_nonempty = candidate("noisy", wer=300.0, cer=200.0, empty=False)
        hard, _attempts = select_hard_examples([easy_empty, hard_nonempty], target=1)
        self.assertEqual(hard[0].sample_id, "empty")

    def test_hard_score_tie_breaking_uses_cer_and_duration(self) -> None:
        left = candidate("left", wer=50.0, cer=30.0, duration=2.0)
        right = candidate("right", wer=50.0, cer=40.0, duration=1.0)
        hard, _attempts = select_hard_examples([left, right], target=1)
        self.assertEqual(hard[0].sample_id, "right")

    def test_discovered_family_cap(self) -> None:
        rows = [
            candidate("a", wer=100.0, cer=10.0, discovered_family="same"),
            candidate("b", wer=99.0, cer=9.0, discovered_family="same"),
            candidate("c", wer=98.0, cer=8.0, discovered_family="other"),
        ]
        hard, _attempts = select_hard_examples(rows, target=2)
        self.assertEqual({item.sample_id for item in hard}, {"a", "c"})

    def test_domain_cap_relaxation_is_recorded(self) -> None:
        rows = [
            candidate(f"row-{index:03d}", wer=100.0 - index / 100, cer=50.0, domain="only-domain", prompt_cell=f"cell{index % 8}")
            for index in range(80)
        ]
        hard, attempts = select_hard_examples(rows, target=40)
        self.assertEqual(len(hard), 40)
        self.assertTrue(any(attempt["relax_domain_cap"] for attempt in attempts))

    def test_deterministic_control_selection(self) -> None:
        rows = [
            candidate(f"row-{index:03d}", wer=float(index), cer=float(index), domain=f"domain-{index % 3}", prompt_cell=f"cell{index % 4}")
            for index in range(30)
        ]
        hard = rows[:5]
        first = [item.sample_id for item in select_controls(rows, hard, target=10)]
        second = [item.sample_id for item in select_controls(list(reversed(rows)), hard, target=10)]
        self.assertEqual(first, second)
        self.assertFalse(set(first) & {item.sample_id for item in hard})

    def test_local_selected_manifest_is_stable(self) -> None:
        hard = [candidate(f"hard-{index}", wer=100 - index, cer=20, prompt_cell=f"cell{index}") for index in range(2)]
        controls = [candidate(f"control-{index}", wer=10, cer=5, prompt_cell=f"cell{index}") for index in range(2)]
        first_manifest, first_audio = local_selected_rows(hard, controls)
        second_manifest, second_audio = local_selected_rows(list(hard), list(controls))
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_audio, second_audio)
        self.assertEqual(first_manifest[0]["role"], "selected_training")

    def test_public_scoring_report_rejects_raw_text_and_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            assert_public_scoring_payload_safe({"candidate_id": "gamsv2-row"})
        with self.assertRaisesRegex(ValueError, "row IDs"):
            assert_public_scoring_payload_safe({"safe": "gamsv2-row"})

    def test_public_selection_certificate_rejects_local_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            assert_public_selection_payload_safe({"hypothesis": "besedilo"})
        with self.assertRaisesRegex(ValueError, "local paths"):
            assert_public_selection_payload_safe({"safe": "/home/example/file"})

    def test_selected_status_constant_is_not_training_eligible(self) -> None:
        self.assertEqual(SELECTED_CERTIFICATE_STATUS, "SELECTED_TRAINING_MANIFEST_READY")
        self.assertNotEqual(SELECTED_CERTIFICATE_STATUS, "TRAINING_ELIGIBLE")


if __name__ == "__main__":
    unittest.main()
