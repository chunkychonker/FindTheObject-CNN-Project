"""CLI: compare two checkpoints' presence on the same scenes: paired ΔAUROC and matched-false-alarm found rates.

Usage:
    python scripts/compare_presence.py --a <run_a> --b <run_b> --split real_dev   # or val

Reads outputs/eval/<run>/presence_<split>.json for both runs (written by evaluate_presence.py) and writes
outputs/eval/compare/<run_a>__<run_b>__<split>.json. ΔAUROC = AUROC_a − AUROC_b with a paired bootstrap
interval (both models scored on each resample): photo-clustered on real_dev, per scene on val. Found at
matched false-alarm rates is descriptive (docs/v3a-spec.md §2.6).

Raises FileNotFoundError if a result is missing; ValueError if the two results were scored on different
data (any data file's SHA-256 differs, A15), so a difference can only come from the models.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from findobject.eval.presence_report import PhotoIntervals, SceneIntervals, auroc_difference, matched_found
from findobject.eval.stats import auroc
from findobject.utils.files import write_json_atomic

EVAL_ROOT = Path("outputs/eval")
SPLITS = ("val", "real_dev")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="run id of model a")
    parser.add_argument("--b", required=True, help="run id of model b")
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args(argv)

    a, b = (json.loads((EVAL_ROOT / run / f"presence_{args.split}.json").read_text()) for run in (args.a, args.b))
    check_same_data(a, b)
    out = compare(a, b, args.split)
    out |= {"a": args.a, "b": args.b, "split": args.split}
    path = EVAL_ROOT / "compare" / f"{args.a}__{args.b}__{args.split}.json"
    write_json_atomic(path, out)

    d = out["delta_auroc"]
    print(f"{args.split}: AUROC a {out['auroc_a']:.4f}, b {out['auroc_b']:.4f}; "
          f"Δ (a − b) {d['value']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    for rate in out["matched_a"]:
        print(f"  found at {float(rate):.0%} false alarms: a {out['matched_a'][rate]['found']:.2%}, "
              f"b {out['matched_b'][rate]['found']:.2%}")
    print(f"wrote {path}")


def check_same_data(a: dict, b: dict) -> None:
    """Raises ValueError unless both results were scored on the same split and the same data files, byte for byte."""
    if a["split"] != b["split"]:
        raise ValueError(f"different splits: {a['split']} vs {b['split']}")
    sha_a = {stem: r["sha256"] for stem, r in a["data"].items()}
    sha_b = {stem: r["sha256"] for stem, r in b["data"].items()}
    if sha_a != sha_b:
        differ = sorted(s for s in sha_a.keys() | sha_b.keys() if sha_a.get(s) != sha_b.get(s))
        raise ValueError(f"refusing to compare: the two results were scored on different data ({', '.join(differ)})")


def compare(a: dict, b: dict, split: str) -> dict:
    """Pure: the comparison of two same-data presence results (check_same_data first)."""
    arrays = {}
    for name, r in (("a", a), ("b", b)):
        ps = r["per_scene"]
        arrays[name] = (np.asarray(ps["positives"]["presence"], np.float32),
                        np.asarray(ps["negatives"]["presence"], np.float32),
                        np.asarray(ps["positives"]["correct"], bool) & np.asarray(ps["positives"]["hit"], bool))
    (a_pos, a_neg, a_joint), (b_pos, b_neg, b_joint) = arrays["a"], arrays["b"]
    if split == "val":
        intervals = SceneIntervals(len(a_pos), len(a_neg))
    else:
        intervals = PhotoIntervals(np.asarray(a["per_scene"]["positives"]["photo_index"]),
                                   np.asarray(a["per_scene"]["negatives"]["photo_index"]))
    delta = auroc_difference(a_pos, a_neg, b_pos, b_neg, intervals)
    return {
        "auroc_a": auroc(a_pos, a_neg),
        "auroc_b": auroc(b_pos, b_neg),
        "delta_auroc": delta,
        "matched_a": matched_found(a_pos, a_joint, a_neg),
        "matched_b": matched_found(b_pos, b_joint, b_neg),
    }


if __name__ == "__main__":
    main()
