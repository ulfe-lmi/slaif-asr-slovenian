#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.tts import load_tts_config, repo_resolve, sha256_file


def api_tree_url(repository: str, revision: str, parent: str) -> str:
    return f"https://huggingface.co/api/models/{repository}/tree/{revision}/{parent}"


def resolve_url(repository: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repository}/resolve/{revision}/{path}"


def fetch_hf_metadata(repository: str, revision: str, path: str) -> dict:
    parent = path.rsplit("/", 1)[0]
    with urllib.request.urlopen(api_tree_url(repository, revision, parent), timeout=30) as response:
        rows = json.load(response)
    for row in rows:
        if row.get("path") == path:
            return row
    raise FileNotFoundError(f"{path} not found in pinned Hugging Face tree")


def valid_existing(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    return path.exists() and path.stat().st_size == expected_size and sha256_file(path) == expected_sha256


def download_atomic(url: str, output_path: Path) -> str | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response:
        etag = response.headers.get("ETag")
        with part_path.open("wb") as fp:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fp.write(chunk)
    os.replace(part_path, output_path)
    return etag.strip('"') if etag else None


def main() -> int:
    cfg = load_tts_config()
    repository = cfg["voice"]["repository"]
    revision = cfg["voice"]["revision"]
    root = repo_resolve(cfg["voice"]["local_storage_dir"])
    for item in cfg["voice"]["files"]:
        relative = item["path"]
        output = root / relative
        metadata = fetch_hf_metadata(repository, revision, relative)
        if "lfs" in metadata:
            lfs_sha = metadata["lfs"].get("oid")
            if item.get("hf_lfs_sha256") != lfs_sha:
                raise ValueError(f"{relative}: LFS SHA256 mismatch: {lfs_sha}")
        expected_size = int(item["byte_size"])
        expected_sha256 = item["sha256"]
        etag = None
        if not valid_existing(output, expected_size=expected_size, expected_sha256=expected_sha256):
            etag = download_atomic(resolve_url(repository, revision, relative), output)
        actual_size = output.stat().st_size
        actual_sha256 = sha256_file(output)
        if actual_size != expected_size:
            raise ValueError(f"{relative}: expected {expected_size} bytes, got {actual_size}")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{relative}: expected SHA256 {expected_sha256}, got {actual_sha256}")
        sidecar = output.with_suffix(output.suffix + ".sha256")
        sidecar.write_text(f"{actual_sha256}  {output.name}\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "path": str(output),
                    "revision": revision,
                    "bytes": actual_size,
                    "sha256": actual_sha256,
                    "hf_lfs_sha256": item.get("hf_lfs_sha256"),
                    "http_etag": etag,
                    "sidecar": str(sidecar),
                    "result": "PASSED",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
