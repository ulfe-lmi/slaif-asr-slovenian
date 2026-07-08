from __future__ import annotations

import hashlib
import unittest

from slaif_asr.artur_controller_dev import (
    ProtectedIndex,
    assert_public_payload_safe,
    gate_audio_hashes,
    gate_recording_ids,
    is_segment_eligible,
    select_controller_dev_segments,
)
from slaif_asr.real_eval import ArturSegment


def segment(recording_id: str, text: str = "Dober dan vsem poslušalcem.", start: float = 0.0) -> ArturSegment:
    return ArturSegment(
        sample_id=f"artur-j-{recording_id}-0000-0000",
        recording_id=recording_id,
        start=start,
        end=start + 3.0,
        text=text,
        transcript_path="fixture.trs",
    )


def protected() -> ProtectedIndex:
    return ProtectedIndex("fixture", set(), set(), set())


class ArturControllerDevTests(unittest.TestCase):
    def test_builder_excludes_gate_recording_identities(self) -> None:
        rows = [{"recording_id": "rec-a"}, {"recording_id": "rec-b"}]
        self.assertEqual(gate_recording_ids({"selected": rows}), {"rec-a", "rec-b"})
        candidates = [segment("rec-a"), segment("rec-c")]
        selected = select_controller_dev_segments(
            candidates,
            excluded_recordings={"rec-a"},
            protected=protected(),
            required_count=1,
        )
        self.assertEqual(selected[0].recording_id, "rec-c")


    def test_audio_hash_overlap_can_be_detected(self) -> None:
        metadata = {"selected": [{"audio_sha256": "abc"}, {"audio_sha256": "def"}]}
        self.assertEqual({"abc"} & gate_audio_hashes(metadata), {"abc"})


    def test_protected_text_surface_hits_are_fatal(self) -> None:
        text = "Dober dan vsem poslušalcem."
        surface_hash = hashlib.sha256("dober dan vsem poslušalcem".encode("utf-8")).hexdigest()
        index = ProtectedIndex("fixture", {surface_hash}, set(), set())
        self.assertFalse(
            is_segment_eligible(
                segment("rec-c", text),
                excluded_recordings=set(),
                protected=index,
                duration_min=2.0,
                duration_max=15.0,
            )
        )


    def test_certificate_schema_allows_hashes_but_rejects_paths_and_raw_keys(self) -> None:
        assert_public_payload_safe({"normalized_reference_hash_set_sha256": "abc", "audio_hash_set_sha256": "def"})
        with self.assertRaises(ValueError):
            assert_public_payload_safe({"reference": "raw text"})
        with self.assertRaises(ValueError):
            assert_public_payload_safe({"manifest": "/home/user/file.wav"})


if __name__ == "__main__":
    unittest.main()
