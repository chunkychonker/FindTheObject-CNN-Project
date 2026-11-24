"""CLI: grade v3b's frozen predictions from the three seeds' whole-photo results (docs/v3b-spec.md §2.6, §3).

Usage:
    python scripts/grade_v3b.py --runs <seed0_run> <seed1_run> <seed2_run> --split dev    # W1-W5
    python scripts/grade_v3b.py --runs <seed0_run> <seed1_run> <seed2_run> --split test   # W6, after test

Dev reads outputs/eval/<run>/detect_dev.json for each run, and seed 0's detect_dev_scale1.json (W3). Test
reads each run's detect_dev.json and detect_test.json. Writes results/v3b/grades_<split>.json: the grades
with the numbers behind each, a per-seed summary of what is reported but not graded, and the SHA-256 of
every file read, so the grade names exactly what it graded.

Raises FileNotFoundError for a missing result; ValueError if a run's files were written for different
checkpoints, a dev file wasn't made at PYRAMID_SCALES, or detect_grade refuses the reports.
"""

import argparse
import json
from pathlib import Path

from findobject.eval.detect_grade import N_SEEDS, grade_dev, grade_test
from findobject.inference.detection import PYRAMID_SCALES
from findobject.utils.files import sha256_file, write_json_atomic

EVAL_ROOT = Path("outputs/eval")
OUT_DIR = Path("results/v3b")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=N_SEEDS, required=True, metavar="RUN", help="seed 0 first")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args(argv)

    files: dict[str, dict] = {}
    dev = [_read(EVAL_ROOT / r / "detect_dev.json", files) for r in args.runs]
    for r, d in zip(args.runs, dev):
        if tuple(d["scales"]) != PYRAMID_SCALES:
            raise ValueError(f"{r}'s detect_dev.json used scales {d['scales']}, not {list(PYRAMID_SCALES)}")
    if args.split == "dev":
        scale1 = _read(EVAL_ROOT / args.runs[0] / "detect_dev_scale1.json", files)
        _same_checkpoint(dev[0], scale1, args.runs[0])
        grades = grade_dev([d["report"] for d in dev], scale1["report"])
        summary = {r: _summary(d) for r, d in zip(args.runs, dev)} | {f"{args.runs[0]} (scale 1)": _summary(scale1)}
    else:
        test = [_read(EVAL_ROOT / r / "detect_test.json", files) for r in args.runs]
        for r, d, t in zip(args.runs, dev, test):
            _same_checkpoint(d, t, r)
        grades = grade_test([d["report"] for d in dev], [t["report"] for t in test])
        summary = {r: _summary(t) for r, t in zip(args.runs, test)}

    out = OUT_DIR / f"grades_{args.split}.json"
    write_json_atomic(out, {"split": args.split, "runs": args.runs, "grades": grades, "summary": summary,
                            "files": files})
    print(f"wrote {out}")
    for w, g in grades.items():
        print(f"  {w}: {'PASS' if g['pass'] else 'FAIL'}  " +
              ", ".join(f"{k} {_fmt(v)}" for k, v in g.items() if k != "pass"))


def _read(path: Path, files: dict[str, dict]) -> dict:
    """Load one result and record its SHA-256 in files. Raises FileNotFoundError if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"no result at {path}: run scripts/detect_photos.py for it first")
    files[str(path)] = {"sha256": sha256_file(path)}
    return json.loads(path.read_text())


def _same_checkpoint(a: dict, b: dict, run: str) -> None:
    if a["checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise ValueError(f"{run}'s result files were written for different checkpoints")


def _summary(result: dict) -> dict:
    """Reported, not graded: per threshold, found by group and FPPI; FP kinds and scales; AP_point."""
    r = result["report"]
    return {"scales": result["scales"], "seconds_per_photo": result["seconds_per_photo"],
            "ap_point": r["ap_point"], "n_photos": r["n_photos"], "n_signs": r["n_signs"],
            "at": {name: {"threshold": at["threshold"],
                          "found": {g: {k: e["found"][k] for k in ("k", "n", "value", "ci")}
                                    for g, e in at["groups"].items()},
                          "located": {g: e["located"]["value"] for g, e in at["groups"].items()},
                          "located_before_suppression": {g: e["located_before_suppression"]["value"]
                                                         for g, e in at["groups"].items()},
                          "class_acc_given_located": {g: e["class_acc_given_located"]["value"]
                                                      for g, e in at["groups"].items()},
                          "differences": at["differences"], "fppi": at["fppi"],
                          "fp_by_kind": at["fp_by_kind"], "fp_by_scale": at["fp_by_scale"],
                          **({"windowed_vs_whole": at["windowed_vs_whole"]} if "windowed_vs_whole" in at else {})}
                   for name, at in r["at"].items()}}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


if __name__ == "__main__":
    main()
