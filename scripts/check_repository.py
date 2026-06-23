#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_FILE_BYTES = 1_048_576
FORBIDDEN_MODEL_SUFFIXES = (
    ".nemo",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".engine",
    ".plan",
)
FORBIDDEN_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus")
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)?PRIVATE KEY-----"),
    re.compile(r"\bAWS_ACCESS_KEY_ID\s*="),
    re.compile(r"\bAWS_SECRET_ACCESS_KEY\s*="),
    re.compile(r"\b(?:HF_TOKEN|HUGGINGFACE_TOKEN|GITHUB_TOKEN|GH_TOKEN)\s*="),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
)
LOCAL_PATH_PATTERNS = (
    re.compile("/" + "mnt" + "/" + "data" + r"\b"),
    re.compile("/" + "home" + "/" + "co" + "dex" + r"\b"),
    re.compile("/" + "Users" + r"/[A-Za-z0-9._-]+"),
)
MARKDOWN_INLINE_LINK = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
MARKDOWN_REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class RepositoryIssue:
    path: str
    message: str


def git_tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git ls-files failed")
    return [line for line in completed.stdout.splitlines() if line]


def validate_repository(
    root: Path,
    tracked_files: list[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[RepositoryIssue]:
    root = root.resolve()
    issues: list[RepositoryIssue] = []
    tracked_set = set(tracked_files)

    for relative in sorted(tracked_set):
        path = root / relative
        suffix = path.suffix.lower()
        lower_name = path.name.lower()

        if suffix in FORBIDDEN_MODEL_SUFFIXES:
            issues.append(RepositoryIssue(relative, "forbidden model artifact filename"))
        if suffix in FORBIDDEN_AUDIO_SUFFIXES:
            issues.append(RepositoryIssue(relative, "forbidden audio artifact filename"))
        if lower_name in {"credentials.json"} or lower_name.endswith(".pem") or lower_name.endswith(".key"):
            issues.append(RepositoryIssue(relative, "forbidden credential filename"))

        try:
            size = path.stat().st_size
        except FileNotFoundError:
            issues.append(RepositoryIssue(relative, "tracked file is missing"))
            continue
        if size > max_file_bytes:
            issues.append(RepositoryIssue(relative, f"tracked file exceeds {max_file_bytes} bytes"))

        text = read_text_if_applicable(path)
        if text is None:
            continue

        issues.extend(validate_text_file(relative, text))
        if suffix == ".json":
            issues.extend(validate_json(relative, text))
            if relative.startswith("docs/evaluation-gates/") and relative.endswith(".metadata.json"):
                issues.extend(validate_real_gate_metadata(relative, text))
        elif suffix == ".jsonl":
            issues.extend(validate_jsonl(relative, text))
        elif suffix == ".toml":
            issues.extend(validate_toml(relative, text))
        elif suffix == ".md":
            issues.extend(validate_markdown_links(root, relative, text, tracked_set))

    return issues


def read_text_if_applicable(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".editorconfig", ".gitattributes", ".gitignore"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def validate_text_file(relative: str, text: str) -> list[RepositoryIssue]:
    issues: list[RepositoryIssue] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if line.endswith((" ", "\t")):
            issues.append(RepositoryIssue(relative, f"trailing whitespace on line {index}"))
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            issues.append(RepositoryIssue(relative, "possible credential or private-key material"))
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            issues.append(RepositoryIssue(relative, "local or private absolute path reference"))
    return issues


def validate_json(relative: str, text: str) -> list[RepositoryIssue]:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return [RepositoryIssue(relative, f"invalid JSON: line {exc.lineno} column {exc.colno}")]
    return []


def validate_real_gate_metadata(relative: str, text: str) -> list[RepositoryIssue]:
    issues: list[RepositoryIssue] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [RepositoryIssue(relative, f"invalid JSON: line {exc.lineno} column {exc.colno}")]
    if not isinstance(payload, dict):
        return [RepositoryIssue(relative, "gate metadata must be a JSON object")]

    forbidden_keys = {"text", "reference", "raw_reference", "raw_transcription", "transcription", "audio_filepath"}

    def walk(value: object, key: str = "") -> None:
        if key in forbidden_keys:
            issues.append(RepositoryIssue(relative, f"gate metadata contains forbidden key: {key}"))
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            if re.search(r"(^|[\"'\s])/(?:home|mnt/data)/", value):
                issues.append(RepositoryIssue(relative, "gate metadata contains a local absolute path"))

    walk(payload)

    if payload.get("gate_id") == "fleurs-sl-si-test-full-v2":
        expected_rows = 834
        selected = payload.get("selected")
        if payload.get("construction_algorithm") != "fleurs-sl-si-test-full-v2":
            issues.append(RepositoryIssue(relative, "FLEURS v2 metadata has wrong construction algorithm"))
        if payload.get("rows") != expected_rows:
            issues.append(RepositoryIssue(relative, "FLEURS v2 metadata must declare 834 rows"))
        if not isinstance(selected, list):
            issues.append(RepositoryIssue(relative, "FLEURS v2 metadata selected must be a list"))
            return issues
        if len(selected) != expected_rows:
            issues.append(RepositoryIssue(relative, f"FLEURS v2 metadata selected count is {len(selected)}, expected 834"))
        allowed_keys = {"source_row_index", "source_id", "sample_id", "audio_sha256", "reference_sha256", "duration_seconds", "gender"}
        indexes: list[int] = []
        sample_ids: list[str] = []
        for item in selected:
            if not isinstance(item, dict):
                issues.append(RepositoryIssue(relative, "FLEURS v2 selected entry must be an object"))
                continue
            extra_keys = set(item) - allowed_keys
            missing_keys = allowed_keys - set(item)
            if extra_keys:
                issues.append(RepositoryIssue(relative, f"FLEURS v2 selected entry has unexpected keys: {sorted(extra_keys)}"))
            if missing_keys:
                issues.append(RepositoryIssue(relative, f"FLEURS v2 selected entry is missing keys: {sorted(missing_keys)}"))
            try:
                source_row_index = int(item["source_row_index"])
            except (KeyError, TypeError, ValueError):
                issues.append(RepositoryIssue(relative, "FLEURS v2 selected entry has invalid source_row_index"))
                continue
            indexes.append(source_row_index)
            sample_id = str(item.get("sample_id", ""))
            sample_ids.append(sample_id)
            expected_sample_id = f"fleurs-sl-si-test-occ-{source_row_index:05d}"
            if sample_id != expected_sample_id:
                issues.append(RepositoryIssue(relative, f"FLEURS v2 sample_id mismatch for source_row_index {source_row_index}"))
        if sorted(indexes) != list(range(expected_rows)):
            issues.append(RepositoryIssue(relative, "FLEURS v2 source_row_index values must be exactly 0..833"))
        if len(sample_ids) != len(set(sample_ids)):
            issues.append(RepositoryIssue(relative, "FLEURS v2 metadata has duplicate sample_id values"))
    return issues


def validate_jsonl(relative: str, text: str) -> list[RepositoryIssue]:
    issues: list[RepositoryIssue] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(RepositoryIssue(relative, f"invalid JSONL on line {index}: column {exc.colno}"))
    return issues


def validate_toml(relative: str, text: str) -> list[RepositoryIssue]:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [RepositoryIssue(relative, f"invalid TOML: {exc}")]
    return []


def validate_markdown_links(root: Path, relative: str, text: str, tracked_files: set[str]) -> list[RepositoryIssue]:
    issues: list[RepositoryIssue] = []
    base_dir = (root / relative).parent
    targets = [match.group(1) for match in MARKDOWN_INLINE_LINK.finditer(text)]
    targets.extend(match.group(1) for match in MARKDOWN_REFERENCE_LINK.finditer(text))
    for raw_target in targets:
        target = normalize_markdown_target(raw_target)
        if should_skip_markdown_target(target):
            continue
        path_part = target.split("#", 1)[0]
        resolved = (base_dir / path_part).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            issues.append(RepositoryIssue(relative, f"Markdown link escapes repository: {raw_target}"))
            continue
        if resolved.exists():
            continue
        repository_relative = os.path.relpath(resolved, root).replace(os.sep, "/")
        if repository_relative not in tracked_files:
            issues.append(RepositoryIssue(relative, f"broken relative Markdown link: {raw_target}"))
    return issues


def normalize_markdown_target(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    return target.strip("'\"")


def should_skip_markdown_target(target: str) -> bool:
    if not target or target.startswith("#"):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tracked repository files for CPU-only CI.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args()

    tracked_files = git_tracked_files(args.root)
    issues = validate_repository(args.root, tracked_files, max_file_bytes=args.max_file_bytes)
    if issues:
        for issue in issues:
            print(f"{issue.path}: {issue.message}", file=sys.stderr)
        return 1
    print(f"Repository validation passed for {len(tracked_files)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
