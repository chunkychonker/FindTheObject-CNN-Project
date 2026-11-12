"""CLI: the seam report for one run on real_dev (docs/v3a1-spec.md §2.3).

Usage:
    python scripts/seam_report.py --run <run_id>

Reads outputs/eval/<run>/presence_real_dev.json (written by evaluate_presence.py) and the boxes of the
real_dev and hybrid_dev files it was scored on. Writes outputs/eval/<run>/seam_real_dev.json.

Raises FileNotFoundError if the result or a data file is missing; ValueError if a data file's SHA-256 is
not the one the result was scored on (the rows would not line up), or as seam_report raises.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from findobject.eval.seam_report import box_long_side, seam_report
from findobject.utils.files import sha256_file, write_json_atomic

EVAL_ROOT = Path("outputs/eval")
REAL, HYBRID = "real_dev", "hybrid_dev"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run id under outputs/eval")
    args = parser.parse_args(argv)

    result = json.loads((EVAL_ROOT / args.run / f"presence_{REAL}.json").read_text())
    boxes = {stem: _boxes(result["data"][stem]) for stem in (REAL, HYBRID)}
    ps = result["per_scene"]
    out = seam_report(np.asarray(ps["positives"]["presence"], np.float32), np.asarray(ps["hybrid"]["presence"], np.float32),
                      box_long_side(boxes[REAL]), box_long_side(boxes[HYBRID]),
                      np.asarray(ps["positives"]["photo_index"]), np.float32(result["tau"]))
    out = {"run_id": args.run, "tau": result["tau"], "data": {s: result["data"][s] for s in (REAL, HYBRID)}} | out
    path = EVAL_ROOT / args.run / f"seam_{REAL}.json"
    write_json_atomic(path, out)

    for name, g in out["median_gap"].items():
        print(f"median gap {name:>14} (n={g['n']:3d}): {g['value']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}]")
    s = out["share_hybrid_higher"]
    print(f"share hybrid > real: {s['value']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")
    print(f"median real {out['median_real']['value']:.3f}, median hybrid {out['median_hybrid']['value']:.3f}")
    print(f"wrote {path}")


def _boxes(entry: dict) -> np.ndarray:
    """The boxes of the data file a result names, after checking it is the same file byte for byte."""
    path = Path(entry["file"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{path} is not the file this result was scored on (SHA-256 differs)")
    with np.load(path) as z:
        return z["boxes"]


if __name__ == "__main__":
    main()
