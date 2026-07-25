from __future__ import annotations

import copy
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import (
    CleanAudioRecord,
    prepare_virtual_sample,
    run_determinism_checks_for_exposures,
    virtual_augmentation_spec,
)
from slaif_asr.scale200_corpus import load_augmentation_config
from slaif_asr.scale8000_otf_surface08 import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    CORPUS_ID,
    Scale8000CleanPool,
    build_round_tasks,
    diversity_statistics,
    exposure_record,
    load_config,
    validate_config,
    validate_loader_telemetry,
    validate_public_report,
)
from scripts.run_surface08_scale8000_otf_augmented import cumulative_fill_rate


CONFIG_PATH = REPO_ROOT / "configs/experiments/surface08-scale8000-otf-augmented.json"
PROFILE_PATH = REPO_ROOT / "configs/augmentation/scale200_transcript_preserving_v1.json"


def write_wav(path: Path) -> None:
    values = (
        np.sin(np.linspace(0.0, 24.0, 3200, endpoint=False)) * 8000.0
    ).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(values.tobytes())


def fake_pool(root: Path, rows: int = 16) -> Scale8000CleanPool:
    records = {}
    order = []
    voices = ("piper-artur", "F1", "F2", "F3", "F4", "M1", "M2", "M3", "M4")
    for index in range(rows):
        key = f"semantic-{index:04d}"
        order.append(key)
        path = root / f"{key}.wav"
        write_wav(path)
        records[key] = tuple(
            CleanAudioRecord(
                semantic_key=key,
                voice=voice,
                source_audio_sha256=f"{index:064x}"[-64:],
                audio_path=path,
                duration_seconds=0.2,
                transcript=f"synthetic transcript {index}",
            )
            for voice in voices
        )
    return Scale8000CleanPool(
        records_by_semantic_key=records,
        semantic_order=tuple(order),
        fixed_text_sha256="a" * 64,
        piper_manifest_sha256="b" * 64,
        supertonic_manifest_sha256="c" * 64,
    )


class Surface08Scale8000OtfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.profiles = list(load_augmentation_config(PROFILE_PATH)["augmentation_profiles"])

    def test_committed_config_is_valid_and_uses_untouched_base(self) -> None:
        loaded = load_config(CONFIG_PATH)
        self.assertEqual(loaded["model"]["initialization"], "untouched_base")
        self.assertEqual(loaded["data"]["training_source"], "scale8000_clean_otf_only")
        self.assertEqual(loaded["otf_augmentation"]["augmentation_key"], AUGMENTATION_KEY)
        self.assertFalse(loaded["otf_augmentation"]["disk_output"])

    def test_governance_records_named_scale8000_surface08_exception(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        adr = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Work Order 0045", agents)
        self.assertIn("Phase 6 / Work Order 0045", adr)
        self.assertIn("not general authorization for scale-8000 full-encoder training", adr)

    def test_scale2000_s6tts_and_real_training_sources_are_rejected(self) -> None:
        for marker in ("scale2000_training", "s6tts", "real_speech"):
            changed = copy.deepcopy(self.config)
            changed["data"]["unexpected_source"] = marker
            with self.assertRaisesRegex(ValueError, "forbidden training sources"):
                validate_config(changed)

    def test_prior_checkpoint_initialization_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["model"]["initialization"] = "surface08_scale2000"
        with self.assertRaisesRegex(ValueError, "prior adapted checkpoint"):
            validate_config(changed)

    def test_schedule_visits_every_semantic_row_before_repeating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pool = fake_pool(Path(directory), rows=16)
            first = [exposure_record(pool, index).semantic_key for index in range(16)]
            second = [exposure_record(pool, index).semantic_key for index in range(16, 32)]
        self.assertEqual(len(set(first)), 16)
        self.assertEqual(set(first), set(second))
        self.assertTrue(
            all(
                exposure_record(pool, index).voice
                != exposure_record(pool, index + 16).voice
                for index in range(16)
            )
        )

    def test_round_tasks_are_deterministic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pool = fake_pool(Path(directory), rows=16)
            first = build_round_tasks(
                pool,
                1,
                physical_microbatch=2,
                effective_batch=8,
                round_size=16,
                duration_bucket_size=16,
            )
            second = build_round_tasks(
                pool,
                1,
                physical_microbatch=2,
                effective_batch=8,
                round_size=16,
                duration_bucket_size=16,
            )
        first_ids = [exposure for task in first for _record, exposure in task.exposures]
        second_ids = [exposure for task in second for _record, exposure in task.exposures]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(set(first_ids), set(range(16)))

    def test_virtual_spec_binds_scale8000_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pool = fake_pool(Path(directory), rows=1)
            record = exposure_record(pool, 0)
            first = virtual_augmentation_spec(
                record,
                7,
                self.profiles,
                corpus_id=CORPUS_ID,
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
            )
            second = virtual_augmentation_spec(
                record,
                7,
                self.profiles,
                corpus_id=CORPUS_ID,
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
            )
        self.assertEqual(first, second)

    def test_waveform_replays_across_workers_and_writes_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = fake_pool(root, rows=2)
            exposures = [(exposure_record(pool, index), index) for index in range(2)]
            before = set(root.iterdir())
            result = run_determinism_checks_for_exposures(
                exposures,
                self.profiles,
                corpus_id=CORPUS_ID,
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
                num_workers=3,
            )
            sample = prepare_virtual_sample(
                exposures[0][0],
                exposures[0][1],
                self.profiles,
                corpus_id=CORPUS_ID,
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
            )
            after = set(root.iterdir())
        self.assertEqual(result["status"], "PASSED")
        self.assertTrue(result["worker_count_independent"])
        self.assertTrue(result["worker_order_independent"])
        self.assertTrue(result["same_virtual_exposure_after_restart"])
        self.assertEqual(sample.transcript, exposures[0][0].transcript)
        self.assertEqual(before, after)

    def test_diversity_stats_report_unique_first_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pool = fake_pool(Path(directory), rows=16)
            stats = diversity_statistics(pool, 20, profile_counts={"profile": 20})
        self.assertEqual(stats["unique_semantic_rows_seen"], 16)
        self.assertEqual(stats["mean_exposures_per_semantic_row"], 1.25)
        self.assertEqual(stats["p95_exposures_per_semantic_row"], 2)

    def test_loader_telemetry_requires_three_workers(self) -> None:
        payload = {
            "round": 1,
            "examples_per_second": 12.8,
            "otf_fill_rate": 0.999,
            "consumer_wait_percent": 0.1,
            "queue_p50": 12,
            "queue_p95": 16,
            "worker_count": 3,
            "microbatch_count": 8000,
        }
        validate_loader_telemetry(payload)
        payload["worker_count"] = 2
        with self.assertRaisesRegex(ValueError, "exactly three"):
            validate_loader_telemetry(payload)

    def test_live_fill_rate_accepts_dictionary_events(self) -> None:
        events = [
            {"consumer_wait_seconds": 0.1},
            {"consumer_wait_seconds": 0.2},
        ]
        self.assertAlmostEqual(cumulative_fill_rate(events, 10.0), 0.97)

    def test_public_report_rejects_paths_and_raw_fields(self) -> None:
        validate_public_report({"classification": "DIAGNOSTIC_ONLY", "rows": 64_000})
        with self.assertRaises(ValueError):
            validate_public_report({"local_path": "/private/data"})
        with self.assertRaises(ValueError):
            validate_public_report({"reference": "raw text"})


if __name__ == "__main__":
    unittest.main()
