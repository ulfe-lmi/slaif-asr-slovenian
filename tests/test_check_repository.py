from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_repository import validate_repository


class RepositoryValidationTests(unittest.TestCase):
    def test_valid_fixture_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("[Doc](docs/guide.md)\n", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
            (root / "config.json").write_text('{"ok": true}\n', encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname = \"fixture\"\n", encoding="utf-8")

            issues = validate_repository(
                root,
                ["README.md", "docs/guide.md", "config.json", "pyproject.toml"],
                max_file_bytes=1024,
            )

            self.assertEqual(issues, [])

    def test_forbidden_model_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.nemo").write_text("placeholder\n", encoding="utf-8")

            issues = validate_repository(root, ["model.nemo"])

            self.assertTrue(any("model artifact" in issue.message for issue in issues))

    def test_audio_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.wav").write_text("placeholder\n", encoding="utf-8")

            issues = validate_repository(root, ["sample.wav"])

            self.assertTrue(any("audio artifact" in issue.message for issue in issues))

    def test_oversized_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("x" * 8, encoding="utf-8")

            issues = validate_repository(root, ["large.txt"], max_file_bytes=4)

            self.assertTrue(any("exceeds" in issue.message for issue in issues))

    def test_broken_relative_markdown_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("[Missing](docs/missing.md)\n", encoding="utf-8")

            issues = validate_repository(root, ["README.md"])

            self.assertTrue(any("broken relative Markdown link" in issue.message for issue in issues))

    def test_untracked_runtime_artifacts_are_not_inspected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# OK\n", encoding="utf-8")
            (root / "runs").mkdir()
            (root / "runs" / "ignored.wav").write_text("placeholder\n", encoding="utf-8")
            (root / "models").mkdir()
            (root / "models" / "ignored.nemo").write_text("placeholder\n", encoding="utf-8")

            issues = validate_repository(root, ["README.md"])

            self.assertEqual(issues, [])

    def test_retired_registry_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "configs" / "data_quality" / "retired_corpora.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"schema_version":"1.0","retired_corpora":[{"artifact":"candidate_pool","sha256":"bad"}]}\n',
                encoding="utf-8",
            )

            issues = validate_repository(root, ["configs/data_quality/retired_corpora.json"])

            self.assertTrue(any("retired registry hashes" in issue.message for issue in issues))

    def test_training_eligible_certificate_requires_sections_and_no_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "docs" / "data-certificates" / "bad.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"schema_version":"1.0","corpus_id":"x","status":"TRAINING_ELIGIBLE","text":"raw"}\n',
                encoding="utf-8",
            )

            issues = validate_repository(root, ["docs/data-certificates/bad.json"])

            self.assertTrue(any("forbidden key: text" in issue.message for issue in issues))
            self.assertTrue(any("TRAINING_ELIGIBLE certificate missing" in issue.message for issue in issues))


if __name__ == "__main__":
    unittest.main()
