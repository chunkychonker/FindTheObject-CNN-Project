"""File digests and atomic JSON writes, for scripts that record results tied to the exact inputs behind them."""

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB chunks. Raises FileNotFoundError if missing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write to <path>.tmp, then rename over path, so a failed run leaves no partial file. Creates parent
    dirs. Raises OSError on write failure, TypeError if obj holds something JSON cannot encode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)
