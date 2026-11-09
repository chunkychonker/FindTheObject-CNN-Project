"""CLI: join committed P6 labels to the sheet's key and grade P6 (docs/v3a-spec.md §2.9 step 3).

Usage:
    python scripts/p6_join.py --labels results/v3a/p6_labels.json --key outputs/eval/<run>/p6_key.json

The labels file must be committed and unchanged since HEAD before the key is read: the commit order is
what shows the labels were made blind to the scores. The labels name the sheet they were made from by
SHA-256, and it must be the key's sheet. Writes results/v3a/p6_result.json next to the labels.

labels file: {"sheet_sha256": "...", "labels": {"1": "A" | "B" | "C" | "D", ... "40": ...}, "notes": {...}}
Raises SystemExit if the labels are not committed and clean; ValueError for a sheet mismatch or a label
set that does not cover the 40 tiles exactly; FileNotFoundError for a missing file.
"""

import argparse
import json
import subprocess
from pathlib import Path

from findobject.eval.p6_sheet import Tile
from findobject.eval.p6_tally import parse_labels, tally
from findobject.utils.files import sha256_file, write_json_atomic


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args(argv)

    require_committed(args.labels)
    labels_doc = json.loads(args.labels.read_text())
    key = json.loads(args.key.read_text())  # only now
    if labels_doc["sheet_sha256"] != key["sheet_sha256"]:
        raise ValueError(f"the labels were made from sheet {labels_doc['sheet_sha256'][:12]}…, "
                         f"but this key is for sheet {key['sheet_sha256'][:12]}…")
    tiles = [Tile(t["number"], t["row"], t["top"]) for t in key["tiles"]]
    labels = parse_labels(labels_doc["labels"], len(tiles))
    result = tally(labels, tiles)
    result |= {"labels_file": str(args.labels), "labels_sha256": sha256_file(args.labels),
               "labels_commit": _git("log", "-1", "--format=%H", "--", str(args.labels)), "key_file": str(args.key),
               "sheet_sha256": key["sheet_sha256"],
               "tiles": [{**t, "label": labels[t["number"]].value} for t in key["tiles"]]}
    out = args.labels.with_name("p6_result.json")
    write_json_atomic(out, result)
    for group in ("top", "control"):
        g = result[group]
        print(f"{group}: {g['sign']}/{g['n']} A or B  {g['counts']}")
    print(f"P6 {'holds' if result['p6_holds'] else 'fails'}; wrote {out}")


def require_committed(path: Path) -> None:
    """Exit unless path is tracked by git and identical to its version at HEAD."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0
    if not (tracked and clean):
        raise SystemExit(f"{path} must be committed, with no changes since, before the key is opened")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


if __name__ == "__main__":
    main()
