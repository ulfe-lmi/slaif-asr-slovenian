from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from slaif_asr.tts import (
    build_piper_command,
    inspect_wav,
    load_candidates,
    load_tts_config,
    render_candidates,
    run_piper_command,
    sha256_file,
    validate_candidate_record,
    validate_wav,
    write_jsonl,
)


def write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((b"\x01\x00" * channels) * frames)


class TtsPipelineTests(unittest.TestCase):
    def valid_record(self) -> dict:
        return {
            "schema_version": "1.0",
            "candidate_id": "piper-smoke-0001",
            "spoken_text": "Čez cesto pelje moder avtomobil.",
            "target_text": "Čez cesto pelje moder avtomobil.",
            "language": "sl-SI",
            "partition_role": "synthetic_smoke",
            "phenomena": ["diacritic:č"],
            "generation": {"system": "manual-fixture", "model": None, "revision": None, "seed": 1},
        }

    def test_valid_candidate_record(self):
        candidate = validate_candidate_record(self.valid_record())
        self.assertEqual(candidate.language, "sl-SI")

    def test_duplicate_candidate_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            row = json.dumps(self.valid_record(), ensure_ascii=False)
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
                load_candidates(path)

    def test_invalid_language_rejected(self):
        row = self.valid_record()
        row["language"] = "sl"
        with self.assertRaisesRegex(ValueError, "language must be sl-SI"):
            validate_candidate_record(row)

    def test_unsafe_candidate_id_rejected(self):
        row = self.valid_record()
        row["candidate_id"] = "../bad"
        with self.assertRaisesRegex(ValueError, "unsafe candidate_id"):
            validate_candidate_record(row)

    def test_target_spoken_mismatch_rejected(self):
        row = self.valid_record()
        row["target_text"] = "Drugo besedilo."
        with self.assertRaisesRegex(ValueError, "spoken_text must equal target_text"):
            validate_candidate_record(row)

    def test_valid_native_and_final_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            native = Path(tmp) / "native.wav"
            final = Path(tmp) / "final.wav"
            write_wav(native, sample_rate=22050)
            write_wav(final, sample_rate=16000)
            self.assertEqual(validate_wav(native, sample_rate=22050).channels, 1)
            self.assertEqual(validate_wav(final, sample_rate=16000).sample_width, 2)

    def test_wrong_sample_rate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.wav"
            write_wav(path, sample_rate=22050)
            with self.assertRaisesRegex(ValueError, "expected 16000 Hz"):
                validate_wav(path, sample_rate=16000)

    def test_stereo_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            write_wav(path, channels=2)
            with self.assertRaisesRegex(ValueError, "expected 1 channel"):
                validate_wav(path, sample_rate=16000)

    def test_empty_audio_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.wav"
            write_wav(path, frames=0)
            with self.assertRaisesRegex(ValueError, "empty audio"):
                inspect_wav(path)

    def test_sha256_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_deterministic_manifest_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = [{"b": 2, "a": "č"}]
            write_jsonl(path, rows)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"a": "č", "b": 2}\n')

    def test_argv_construction_and_no_shell(self):
        command = build_piper_command(
            piper_python=Path(".venv-piper/bin/python"),
            model_path=Path("voice.onnx"),
            config_path=Path("voice.onnx.json"),
            output_file=Path("out.wav"),
            text="Živjo.",
        )
        self.assertIn("--cuda", command)
        self.assertIn("--debug", command)
        self.assertEqual(command[-1], "Živjo.")
        self.assertNotIn("--", command)
        self.assertNotIn("shell=True", command)

    def test_no_shell_invocation(self):
        with mock.patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(["python"], 0, "ok", "")
            run_piper_command(["python", "-m", "piper"], env={})
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_failed_piper_subprocess_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.minimal_config(root)
            candidates = [validate_candidate_record(self.valid_record())]
            self.prepare_runtime_files(root)
            with mock.patch("slaif_asr.tts.sox_version", return_value="SoX fixture"), mock.patch(
                "slaif_asr.tts.run_piper_command"
            ) as run:
                run.return_value = subprocess.CompletedProcess(["piper"], 1, "boom", "")
                with self.assertRaisesRegex(RuntimeError, "Piper failed"):
                    render_candidates(candidates=candidates, config=cfg, output_root=root / "out")

    def test_missing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.minimal_config(root)
            candidates = [validate_candidate_record(self.valid_record())]
            self.prepare_runtime_files(root)
            with mock.patch("slaif_asr.tts.sox_version", return_value="SoX fixture"), mock.patch(
                "slaif_asr.tts.run_piper_command"
            ) as run:
                run.return_value = subprocess.CompletedProcess(["piper"], 0, "DEBUG Using CUDA\n", "")
                with self.assertRaises(FileNotFoundError):
                    render_candidates(candidates=candidates, config=cfg, output_root=root / "out")

    def test_cuda_provider_fallback_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.minimal_config(root)
            candidates = [validate_candidate_record(self.valid_record())]
            self.prepare_runtime_files(root)
            with mock.patch("slaif_asr.tts.sox_version", return_value="SoX fixture"), mock.patch(
                "slaif_asr.tts.run_piper_command"
            ) as run:
                run.return_value = subprocess.CompletedProcess(
                    ["piper"],
                    0,
                    "DEBUG:piper.voice:Using CUDA\nFailed to create CUDAExecutionProvider\n",
                    "",
                )
                with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
                    render_candidates(candidates=candidates, config=cfg, output_root=root / "out")

    def test_render_metadata_uses_deterministic_conversion_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.minimal_config(root)
            candidates = [validate_candidate_record(self.valid_record())]
            self.prepare_runtime_files(root)

            def fake_run(command, *, env):
                output = Path(command[command.index("--output-file") + 1])
                write_wav(output, sample_rate=22050)
                return subprocess.CompletedProcess(command, 0, "DEBUG Using CUDA\n", "")

            with mock.patch("slaif_asr.tts.run_piper_command", side_effect=fake_run), mock.patch(
                "slaif_asr.tts.convert_to_16k_pcm", side_effect=lambda native, final: write_wav(final, sample_rate=16000)
            ), mock.patch("slaif_asr.tts.sox_version", return_value="SoX fixture"):
                result = render_candidates(candidates=candidates, config=cfg, output_root=root / "out")

            self.assertEqual(result["candidate_count"], 1)
            manifest = (root / "out" / "nemo-manifest.jsonl").read_text(encoding="utf-8")
            self.assertIn('"target_lang": "sl-SI"', manifest)
            provenance = json.loads((root / "out" / "rendered-records.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(provenance["audio_validation"]["conversion"]["version"], "SoX fixture")

    def test_license_configuration_completeness(self):
        cfg = load_tts_config()
        text = json.dumps(cfg, ensure_ascii=False)
        self.assertEqual(cfg["engine"]["license"], "GPL-3.0-or-later")
        self.assertEqual(cfg["voice"]["revision"], "217ddc79818708b078d0d14a8fae9608b9d77141")
        self.assertIn("CC BY-SA 4.0", text)
        self.assertIn("11356/1776", text)
        self.assertFalse(cfg["license"]["generated_audio_public_release_permitted_by_this_pr"])

    def test_no_project_package_imports_piper(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "slaif_asr").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("import piper", text)
            self.assertNotIn("from piper", text)

    def test_no_tracked_voice_model_or_audio_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(["git", "ls-files"], cwd=root, text=True, stdout=subprocess.PIPE, check=True)
        tracked = completed.stdout.splitlines()
        forbidden_suffixes = (".onnx", ".wav", ".flac", ".mp3", ".nemo", ".ckpt")
        self.assertFalse([path for path in tracked if path.endswith(forbidden_suffixes)])
        self.assertFalse([path for path in tracked if path.startswith(".external/piper1-gpl")])

    def test_engine_and_voice_download_paths_are_ignored(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            ".external/piper1-gpl/README.md",
            ".external/piper-voices/sl/sl_SI/artur/medium/sl_SI-artur-medium.onnx",
            "runs/tts/piper/final-16000/piper-smoke-0001.wav",
        ]
        ignored = []
        for path in paths:
            completed = subprocess.run(
                ["git", "check-ignore", path],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode == 0:
                ignored.append(completed.stdout.strip())
                continue
            # Some development hosts make .external an ignored symlink to a
            # large local cache. Git refuses to traverse beyond that symlink,
            # so checking the ignored parent is the equivalent safety proof.
            if path.startswith(".external/"):
                parent = subprocess.run(
                    ["git", "check-ignore", ".external"],
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    check=True,
                )
                ignored.append(path if parent.stdout.strip() == ".external" else "")
        self.assertEqual(ignored, paths)

    def minimal_config(self, root: Path) -> dict:
        return {
            "engine": {
                "repository_name": "OHF-Voice/piper1-gpl",
                "revision": "engine-rev",
                "license": "GPL-3.0-or-later",
                "environment": str(root / ".venv-piper"),
            },
            "voice": {
                "local_storage_dir": str(root / "voice"),
                "files": [
                    {"role": "model", "path": "voice.onnx"},
                    {"role": "config", "path": "voice.onnx.json"},
                ],
                "native_sample_rate": 22050,
                "final_asr_sample_rate": 16000,
                "name": "sl_SI-artur-medium",
                "repository": "rhasspy/piper-voices",
                "revision": "voice-rev",
            },
            "runtime": {"required_execution_provider": "CUDAExecutionProvider", "physical_gpu": 0},
            "local_artifacts": {},
        }

    def prepare_runtime_files(self, root: Path) -> None:
        (root / ".venv-piper" / "bin").mkdir(parents=True)
        (root / ".venv-piper" / "bin" / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
        (root / "voice").mkdir()
        (root / "voice" / "voice.onnx").write_bytes(b"onnx")
        (root / "voice" / "voice.onnx.json").write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
