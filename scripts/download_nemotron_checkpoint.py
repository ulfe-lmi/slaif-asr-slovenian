#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned official Nemotron 3.5 ASR checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=repo_path("local_artifacts.checkpoint_dir"))
    parser.add_argument("--force", action="store_true", help="Re-download even when the target exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate URL metadata without downloading the file.")
    args = parser.parse_args()

    cfg = load_runtime_config()["base_model"]
    if not re.fullmatch(r"[0-9a-f]{64}", cfg["sha256"]):
        print(f"Configured SHA256 is not a 64-character lowercase hex digest: {cfg['sha256']}", file=sys.stderr)
        return 2
    url = f"https://huggingface.co/{cfg['repository']}/resolve/{cfg['revision']}/{cfg['filename']}"
    target = args.output_dir / cfg["filename"]
    checksum_file = target.with_suffix(target.suffix + ".sha256")

    if args.dry_run:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            print(f"url={url}")
            print(f"status={response.status}")
            print(f"content_length={response.headers.get('Content-Length')}")
            print(f"etag={response.headers.get('ETag')}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.force:
        size, digest = file_size_and_sha256(target)
        if digest != cfg["sha256"]:
            print(f"Checksum mismatch for existing file: {digest} != {cfg['sha256']}", file=sys.stderr)
            return 2
        if int(cfg.get("byte_size", size)) != size:
            print(f"Byte-size mismatch for existing file: {size} != {cfg['byte_size']}", file=sys.stderr)
            return 2
        checksum_file.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        print(f"Verified existing checkpoint: {target}")
        print(f"bytes={size}")
        print(f"sha256={digest}")
        return 0

    tmp_target = target.with_suffix(target.suffix + ".part")
    tmp_target.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with urllib.request.urlopen(url, timeout=60) as response, tmp_target.open("wb") as fp:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            fp.write(chunk)
            byte_count += len(chunk)
    actual = digest.hexdigest()
    if actual != cfg["sha256"]:
        tmp_target.unlink(missing_ok=True)
        print(f"Checksum mismatch for downloaded file: {actual} != {cfg['sha256']}", file=sys.stderr)
        return 2
    if int(cfg.get("byte_size", byte_count)) != byte_count:
        tmp_target.unlink(missing_ok=True)
        print(f"Byte-size mismatch for downloaded file: {byte_count} != {cfg['byte_size']}", file=sys.stderr)
        return 2
    tmp_target.replace(target)
    checksum_file.write_text(f"{actual}  {target.name}\n", encoding="utf-8")
    print(f"Downloaded and verified checkpoint: {target}")
    print(f"bytes={byte_count}")
    print(f"sha256={actual}")
    return 0


def file_size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
