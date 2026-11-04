"""Restore byte-identical study snapshots from the frozen source manifests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


MANIFESTS = (
    "data/raw_manifest.json",
    "cross_year_2024/data/raw_manifest.json",
    "cross_year_2024/data_harmonized_2023/raw_manifest.json",
)


def verify(path, metadata):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Snapshot differs from the frozen manifest: {path}")


def restore(path, metadata, verify_only):
    if path.exists() or verify_only:
        verify(path, metadata)
        return "verified"

    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            with urlopen(metadata["url"], timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            verify(temporary, metadata)
            temporary.replace(path)
            return "downloaded and verified"
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def manifest_entries(study_root):
    for relative in MANIFESTS:
        manifest = study_root / relative
        content = json.loads(manifest.read_text(encoding="utf-8"))
        if content.get("errors") or not content.get("files"):
            raise ValueError(f"Incomplete source manifest: {manifest}")
        for name, metadata in content["files"].items():
            if name in {"", ".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe snapshot filename in {manifest}: {name}")
            if urlsplit(metadata["url"]).scheme != "https" or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]):
                raise ValueError(f"Invalid source URL or SHA-256 for {name}")
            yield manifest.parent / "raw" / name, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--verify-only", action="store_true", help="Verify local snapshots without requesting network data")
    args = parser.parse_args()
    root = args.study_root.resolve()
    entries = list(manifest_entries(root))
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = {pool.submit(restore, path, metadata, args.verify_only): path for path, metadata in entries}
        for future in as_completed(requests):
            path = requests[future].relative_to(root)
            try:
                print(f"{future.result()} {path}")
            except Exception as exc:
                failures.append(path)
                print(f"ERROR {path}: {exc}")
    print(f"{len(entries) - len(failures)}/{len(entries)} snapshots verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
