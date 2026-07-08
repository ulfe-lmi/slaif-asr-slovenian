from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from slaif_asr.s6tts_tts import (
    PINNED_REVISION,
    build_s6_command,
    local_path,
    summarize_local_view,
    validate_public_payload,
    validate_s6_config,
)
from slaif_asr.tts import write_jsonl
from slaif_asr.s6tts_tts import load_s6_config, s6_paths
from slaif_asr.tts import validate_wav


def write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes((b"\x01\x00" * channels) * frames)


class S6TtsTests(unittest.TestCase):
    def test_config_validator_accepts_pinned_config(self):
        config = load_s6_config()
        self.assertEqual(config["engine"]["revision"], PINNED_REVISION)

    def test_config_validator_rejects_wrong_revision(self):
        config = load_s6_config()
        config["engine"] = dict(config["engine"])
        config["engine"]["revision"] = "bad"
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_s6_config(config)

    def test_command_builder_uses_argv_and_ini_output(self):
        command = build_s6_command(
            cli_path=Path(".external/s6tts/build/s6cli"),
            ini_path=Path(".external/s6tts/data/sl-si-s6/sint.ini"),
            text="Živjo.",
            output_file=Path("out.wav"),
        )
        self.assertEqual(command[:2], [".external/s6tts/build/s6cli", "--ini"])
        self.assertIn("--text", command)
        self.assertIn("-o", command)
        self.assertNotIn("shell=True", command)

    def test_runs_root_override_redirects_runs_paths(self):
        with mock.patch.dict("os.environ", {"SLAIF_ASR_RUNS_ROOT": "/tmp/slaif-s6-runs"}):
            self.assertEqual(
                local_path("runs/data-quality/sl-corpus-v4-s6tts-vintage-clean-view-v1"),
                Path("/tmp/slaif-s6-runs/data-quality/sl-corpus-v4-s6tts-vintage-clean-view-v1"),
            )

    def test_no_shell_invocation(self):
        command = build_s6_command(cli_path=Path("s6cli"), ini_path=Path("sint.ini"), text="Živjo.", output_file=Path("out.wav"))
        with mock.patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(command, 0, "", "")
            from slaif_asr.s6tts_tts import run_s6_command

            run_s6_command(command)
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_wav_validator_accepts_mono_16bit_16khz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.wav"
            write_wav(path)
            self.assertEqual(validate_wav(path, sample_rate=16000).sample_width, 2)

    def test_wav_validator_rejects_wrong_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong_rate = Path(tmp) / "wrong-rate.wav"
            write_wav(wrong_rate, sample_rate=8000)
            with self.assertRaisesRegex(ValueError, "expected 16000 Hz"):
                validate_wav(wrong_rate, sample_rate=16000)
            stereo = Path(tmp) / "stereo.wav"
            write_wav(stereo, channels=2)
            with self.assertRaisesRegex(ValueError, "expected 1 channel"):
                validate_wav(stereo, sample_rate=16000)
            width = Path(tmp) / "width.wav"
            write_wav(width, sample_width=1)
            with self.assertRaisesRegex(ValueError, "expected 2-byte"):
                validate_wav(width, sample_rate=16000)

    def test_public_payload_rejects_raw_text_and_paths(self):
        with self.assertRaisesRegex(ValueError, "raw text"):
            validate_public_payload({"text": "Živjo."})
        with self.assertRaisesRegex(ValueError, "local absolute"):
            validate_public_payload({"audio": "/home/janezp/file.wav"})

    def test_summary_schema_requires_zero_failures_for_acceptance(self):
        config = load_s6_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "s6tts"
            (source / "data" / "sl-si-s6").mkdir(parents=True)
            (source / "data" / "sl-si-s6" / "sint.ini").write_text("fixture", encoding="utf-8")
            paths = s6_paths(config)
            paths = paths.__class__(
                source_dir=source,
                build_dir=root / "build",
                cli_path=root / "s6cli",
                runtime_ini=source / "data" / "sl-si-s6" / "sint.ini",
                run_root=root / "run",
                audio_manifest=root / "run" / "audio-manifest.local.jsonl",
                provenance_manifest=root / "run" / "provenance.local.jsonl",
                validation=root / "run" / "audio-validation.local.json",
                summary=root / "run" / "summary.local.json",
                logs_dir=root / "run" / "logs",
            )
            row = {
                "audio_relative_path": "wav/a.wav",
                "audio_sha256": "a" * 64,
                "duration_seconds": 1.0,
                "peak_ratio": 0.5,
            }
            write_jsonl(paths.audio_manifest, [row])
            write_jsonl(paths.provenance_manifest, [row])
            summary = summarize_local_view(config, paths)
            self.assertEqual(summary["status"], "S6TTS_REJECTED_SYNTHESIS_QUALITY")
            self.assertEqual(summary["actual_clean_files"], 1)
            self.assertIn("TRAINING_ELIGIBLE", summary["prohibited_statuses"])

    def test_no_tracked_wav_or_s6_external_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(["git", "ls-files"], cwd=root, text=True, stdout=subprocess.PIPE, check=True)
        tracked = completed.stdout.splitlines()
        self.assertFalse([path for path in tracked if path.endswith((".wav", ".onnx", ".nemo", ".ckpt", ".pt", ".pth"))])
        self.assertFalse([path for path in tracked if path.startswith(".external/s6tts")])


if __name__ == "__main__":
    unittest.main()
