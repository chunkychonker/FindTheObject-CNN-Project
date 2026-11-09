"""CLI: the v2 report for one run on real-photo windows vs their paired hybrid scenes (docs/v2-spec.md §2.5).

Usage: python scripts/report_real.py --run peak_fresh_aug-20261108-024220 --split dev

Needs outputs/eval/<run>/best_real_<split>.json and best_hybrid_<split>.json (from scripts/evaluate.py), the
npz files they were scored on, and the overlap flags (scripts/check_gtsdb_overlap.py). Prints the report
and writes outputs/eval/<run>/real_<split>_report.json.

Subsets (§2.6): all windows; unseen = no exact-frame match in GTSRB train; unflagged = neither exact nor
similarity-flagged; seen = exact-frame match. Every rate has a Wilson 95% interval; real vs hybrid is an
exact McNemar test on joint accuracy, paired by sign.

Raises FileNotFoundError for a missing input; ValueError if an eval JSON was scored on a different file
than the npz now on disk (SHA-256 mismatch), if real and hybrid rows are not the same signs, or if a
window has no overlap record.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from findobject.eval.real_report import SizeBucket, discordant, rates_by, size_bucket
from findobject.eval.stats import Rate, mcnemar_exact, wilson

EVAL_ROOT = Path("outputs/eval")
OVERLAP = Path("results/v2-gtsdb-overlap/overlap.json")
TIERS = ("real", "hybrid")
MIN_CLASS_N = 10  # per-class rates shown only with at least this many windows
WORST_SHOWN = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--overlap", type=Path, default=OVERLAP)
    args = parser.parse_args()

    meta, outcome = None, {}
    for tier in TIERS:
        npz = args.processed_dir / f"{tier}_{args.split}.npz"
        ev = json.loads((EVAL_ROOT / args.run / f"best_{tier}_{args.split}.json").read_text())
        if ev["data_sha256"] != _sha256(npz):
            raise ValueError(f"{tier}: eval JSON was scored on a different {npz} than the one on disk")
        with np.load(npz) as d:
            m = {k: d[k].astype(np.int64) for k in ("photo_index", "sign_index", "class_ids", "native_long_side")}
        if meta is None:
            meta = m
        elif any(not np.array_equal(meta[k], m[k]) for k in meta):
            raise ValueError("real and hybrid rows are not the same signs")
        ps = ev["per_scene"]
        correct, hit = np.array(ps["correct"], bool), np.array(ps["hit"], bool)
        outcome[tier] = {"joint": correct & hit, "class": correct, "pointing": hit, "y_pred": np.array(ps["y_pred"])}

    flags = {(r["photo_index"], r["sign_index"]): r for r in json.loads(args.overlap.read_text())["signs"]}
    keys = list(zip(meta["photo_index"].tolist(), meta["sign_index"].tolist()))
    missing = [k for k in keys if k not in flags]
    if missing:
        raise ValueError(f"{len(missing)} windows have no overlap record, e.g. {missing[0]}")
    exact = np.array([flags[k]["exact_frame"] for k in keys])
    sim = np.array([flags[k]["sim_flag"] for k in keys])
    subsets = {"all": np.ones(len(keys), bool), "unseen": ~exact, "unflagged": ~exact & ~sim, "seen": exact}
    sizes = [size_bucket(L).value for L in meta["native_long_side"].tolist()]

    report = {"run": args.run, "split": args.split, "n_windows": len(keys), "subsets": {}}
    for name, mask in subsets.items():
        block = {"n": int(mask.sum())}
        for tier in TIERS:
            block[tier] = {m: _rate(outcome[tier][m][mask]) for m in ("joint", "class", "pointing")}
            block[tier]["joint_by_size"] = {
                k: _jsonable_rate(r) for k, r in rates_by(outcome[tier]["joint"][mask],
                                                          [s for s, keep in zip(sizes, mask) if keep]).items()}
        only_real, only_hybrid = discordant(outcome["real"]["joint"][mask], outcome["hybrid"]["joint"][mask])
        block["mcnemar_real_vs_hybrid"] = {"only_real": only_real, "only_hybrid": only_hybrid,
                                           "p": mcnemar_exact(only_real, only_hybrid)}
        report["subsets"][name] = block

    cls = rates_by(outcome["real"]["joint"], meta["class_ids"].tolist(), min_n=MIN_CLASS_N)
    report["real_joint_by_class"] = {str(k): _jsonable_rate(r) for k, r in sorted(cls.items())}
    manifest = json.loads((args.processed_dir / "real_manifest.json").read_text())["splits"][args.split]
    report["signs"], report["skipped"] = manifest["signs"], manifest["skipped"]

    out = EVAL_ROOT / args.run / f"real_{args.split}_report.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, out)
    _print(report, cls)
    print(f"\nwrote {out}")


def _rate(ok: np.ndarray) -> dict:
    return _jsonable_rate(wilson(int(ok.sum()), len(ok)))


def _jsonable_rate(r: Rate) -> dict:
    return {"k": r.k, "n": r.n, "value": r.value, "lo": r.lo, "hi": r.hi}


def _fmt(r: dict) -> str:
    return f"{100 * r['value']:5.1f}% [{100 * r['lo']:4.1f}, {100 * r['hi']:5.1f}] (n={r['n']})"


def _print(report: dict, cls: dict) -> None:
    print(f"{report['run']} on {report['split']}: {report['signs']} signs -> {report['n_windows']} windows, "
          f"skipped {report['skipped']}")
    for name, b in report["subsets"].items():
        mc = b["mcnemar_real_vs_hybrid"]
        print(f"\n[{name}] n={b['n']}")
        for tier in TIERS:
            t = b[tier]
            print(f"  {tier:6s} joint {_fmt(t['joint'])} | class {100 * t['class']['value']:.1f}% "
                  f"| pointing {100 * t['pointing']['value']:.1f}%")
        print(f"  real vs hybrid joint: only real {mc['only_real']}, only hybrid {mc['only_hybrid']}, "
              f"McNemar p = {mc['p']:.2g}")
        for tier in TIERS:
            by = b[tier]["joint_by_size"]
            print(f"  {tier:6s} by size: " + " | ".join(f"{s.value} {_fmt(by[s.value])}" for s in SizeBucket
                                                        if s.value in by))
    worst = sorted(cls.items(), key=lambda kv: kv[1].value)[:WORST_SHOWN]
    print("\nworst classes on real (id: joint, n): "
          + ", ".join(f"{k}: {100 * r.value:.0f}% n={r.n}" for k, r in worst))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
