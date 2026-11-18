"""CLI: draw the v3b hand-audit sheet of a run's top dev background false positives at τ_W (docs/v3b-spec.md §2.6).

Usage:
    python scripts/fp_sheet.py --run peak_decoy-20261112-011953

Reads outputs/eval/<run>/detect_dev.json. Writes:
  results/v3b/fp_sheet.png          the sheet, committed: numbered crops, no scores
  results/v3b/fp_audit.csv          number,label,note with empty labels, to fill in from the sheet alone
                                    with one of LABELS (P6's four, docs/v3a-spec.md §4 step 2)
  outputs/eval/<run>/fp_key.json    which photo, point, scale and score each number is. Don't open it
                                    before the labels are committed.

Refuses (ValueError) to overwrite an fp_audit.csv that already holds a label, so no labeling is lost.
Raises FileNotFoundError for a missing detect_dev.json or photo; ValueError from choose_fps. Every file is
written atomically.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from findobject.data.gtsdb import load_photo, read_gtsdb
from findobject.eval.detect_report import TAU_W
from findobject.eval.fp_sheet import N_AUDIT, choose_fps, render_fp_sheet
from findobject.utils.files import sha256_file, write_json_atomic

EVAL_ROOT = Path("outputs/eval")
OUT_DIR = Path("results/v3b")
GTSDB_DIR = Path("data/raw/gtsdb/FullIJCNN2013")
FP_SEED = 0
# P6's categories, applied to the object at the marked point. When unsure between two, the later letter:
# it removes fewer false positives, so the estimate of FPPI without unlabeled signs stays conservative.
LABELS = {"A": "an unlabeled sign outside the 43 classes",
          "B": "an unlabeled sign of one of the 43 classes",
          "C": "a sign-like non-sign: round, triangular or octagonal red/blue/white object, traffic light, "
               "vehicle lights, logo",
          "D": "nothing sign-like"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--gtsdb", type=Path, default=GTSDB_DIR)
    args = parser.parse_args(argv)

    dev_path = EVAL_ROOT / args.run / "detect_dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"no result at {dev_path}: run scripts/detect_photos.py --split dev first")
    csv_path = OUT_DIR / "fp_audit.csv"
    refuse_labeled(csv_path)
    dev = json.loads(dev_path.read_text())
    threshold = dev["thresholds"][TAU_W]
    tiles = choose_fps(dev["photos"], threshold, np.random.default_rng(FP_SEED))

    by_index = {p.index: p for p in read_gtsdb(args.gtsdb)}
    images = {i: load_photo(by_index[i]) for i in sorted({t.photo_index for t in tiles})}
    sheet_path = OUT_DIR / "fp_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sheet_path.with_name("fp_sheet.tmp.png")
    render_fp_sheet(images, tiles).save(tmp)
    tmp.replace(sheet_path)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["number", "label", "note"])
    w.writerows([[n, "", ""] for n in range(1, N_AUDIT + 1)])
    tmp = csv_path.with_name("fp_audit.tmp.csv")
    tmp.write_text(buf.getvalue())
    tmp.replace(csv_path)

    key_path = EVAL_ROOT / args.run / "fp_key.json"
    write_json_atomic(key_path, {
        "run_id": args.run, "detect_dev": str(dev_path), "detect_dev_sha256": sha256_file(dev_path),
        "checkpoint_sha256": dev["checkpoint_sha256"], "threshold": threshold, "seed": FP_SEED,
        "sheet_sha256": sha256_file(sheet_path), "labels": LABELS,
        "tiles": [t._asdict() for t in tiles],
    })
    print(f"wrote {sheet_path}, {csv_path} (label each number {'/'.join(LABELS)})\n"
          f"wrote {key_path} (don't open it before the labels are committed)")


def refuse_labeled(csv_path: Path) -> None:
    """Raises ValueError if csv_path exists and any row has a non-empty label."""
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        if any(row.get("label", "").strip() for row in csv.DictReader(f)):
            raise ValueError(f"{csv_path} already holds labels; move it aside to draw a new sheet")


if __name__ == "__main__":
    main()
