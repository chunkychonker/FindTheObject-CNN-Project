"""CLI: draw the v3b hand-audit sheet of a run's top dev background false positives at τ_W (docs/v3b-spec.md §2.6).

Usage:
    python scripts/fp_sheet.py --run peak_decoy-20261112-011953

Reads outputs/eval/<run>/detect_dev.json. Writes:
  results/v3b/fp_sheet.png          the sheet, committed: numbered crops, no scores
  results/v3b/fp_audit.csv          number,label with empty labels, to fill in by hand with one of
                                    unlabeled_sign, sign_like, other
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
LABELS = ("unlabeled_sign", "sign_like", "other")


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
    w.writerow(["number", "label"])
    w.writerows([[n, ""] for n in range(1, N_AUDIT + 1)])
    tmp = csv_path.with_name("fp_audit.tmp.csv")
    tmp.write_text(buf.getvalue())
    tmp.replace(csv_path)

    key_path = EVAL_ROOT / args.run / "fp_key.json"
    write_json_atomic(key_path, {
        "run_id": args.run, "detect_dev": str(dev_path), "detect_dev_sha256": sha256_file(dev_path),
        "checkpoint_sha256": dev["checkpoint_sha256"], "threshold": threshold, "seed": FP_SEED,
        "sheet_sha256": sha256_file(sheet_path), "labels": list(LABELS),
        "tiles": [t._asdict() for t in tiles],
    })
    print(f"wrote {sheet_path}, {csv_path} (label each number: {', '.join(LABELS)})\n"
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
