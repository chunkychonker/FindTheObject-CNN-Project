"""CLI: find every sign in whole GTSDB photos with a trained run, and record the v3b report (docs/v3b-spec.md §3).

Usage:
    python scripts/detect_photos.py --checkpoint outputs/checkpoints/<run>/best.pt --split dev
    python scripts/detect_photos.py --checkpoint ... --split dev --scales 1      # W3's scale-1-only comparison
    python scripts/detect_photos.py --checkpoint ... --split test                # once per run, after dev

  split  photos    scales                     thresholds                               writes
  dev    0-599     PYRAMID_SCALES, or 1 alone chosen here: τ_W etc. from dev FPs       detect_dev.json / detect_dev_scale1.json
  test   600-899   PYRAMID_SCALES only        read from this run's detect_dev.json     detect_test.json

Writes outputs/eval/<run>/<file>: every kept detection with its match, every sign's best raw score, the
§2.4 report, the thresholds, the scales, the checkpoint's SHA-256, the device and seconds per photo.

Dev reads τ_syn from presence_val.json and each sign's windowed outcome from presence_real_dev.json (both
must carry this checkpoint's SHA-256), and chooses τ_W (dev FPPI <= 0.5) and the FPPI 0.1 / 1.0 points from
its own false positives. Test (A12) reads every threshold from detect_dev.json, which must carry the same
SHA-256 and have used PYRAMID_SCALES, takes no threshold or scale argument, and exits with code 2 if
detect_test.json already exists. There is no --force: whole test photos are scored once per run. No test
number is printed before the file is written.

Raises FileNotFoundError for a missing checkpoint, photo or stored result; ValueError for a stored result
from another checkpoint, a dev file made at other scales, or windowed rows that don't line up with
real_dev.npz. Nothing is written on any failure; the JSON is written atomically.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from findobject.data.gtsdb import RealPhoto, RealSplit, load_photo, photos_in, read_gtsdb
from findobject.eval.detect_report import TAU_SYN, TAU_W, PhotoRecord, detect_report
from findobject.eval.froc import threshold_at_fppi
from findobject.eval.photo_match import best_raw_score, match_photo
from findobject.eval.photo_truth import photo_truth
from findobject.eval.presence_report import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from findobject.eval.test_guard import scored_results
from findobject.inference.detect import detect
from findobject.inference.detection import PYRAMID_SCALES
from findobject.training.checkpoint import load_checkpoint_info, load_model
from findobject.utils.device import pick_device
from findobject.utils.files import sha256_file, write_json_atomic

EVAL_ROOT = Path("outputs/eval")
GTSDB_DIR = Path("data/raw/gtsdb/FullIJCNN2013")
CHECKPOINT_STEM = "best"
SCALE1 = (1.0,)
SPLITS = {"dev": RealSplit.DEV, "test": RealSplit.TEST}
TAU_W_FPPI = 0.5  # §2.5: τ_W is the smallest threshold with dev FPPI <= 0.5
REPORTED_FPPI = (0.1, 1.0)  # reported, never graded
OUT_FILES = {("dev", PYRAMID_SCALES): "detect_dev.json", ("dev", SCALE1): "detect_dev_scale1.json",
             ("test", PYRAMID_SCALES): "detect_test.json"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=None,
                        help=f"dev only: {' '.join(map(str, PYRAMID_SCALES))} (the default) or 1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gtsdb", type=Path, default=GTSDB_DIR)
    args = parser.parse_args(argv)
    if args.checkpoint.stem != CHECKPOINT_STEM:
        parser.error(f"only {CHECKPOINT_STEM}.pt is scored (the result file names no checkpoint), got {args.checkpoint}")
    if args.split == "test" and args.scales is not None:
        parser.error("--scales is dev only: test uses PYRAMID_SCALES, as detect_dev.json did")
    scales = PYRAMID_SCALES if args.scales is None else tuple(args.scales)
    if (args.split, scales) not in OUT_FILES:
        parser.error(f"--scales must be {' '.join(map(str, PYRAMID_SCALES))} or 1, got {' '.join(map(str, scales))}")

    run_dir = EVAL_ROOT / args.checkpoint.parent.name
    out_path = run_dir / OUT_FILES[(args.split, scales)]
    if args.split == "test":
        refuse_test_rerun(run_dir)
    ckpt_sha = sha256_file(args.checkpoint)
    # Everything that can refuse runs before the slow part.
    if args.split == "dev":
        tau_syn = read_tau_syn(run_dir, ckpt_sha)
        windowed = read_windowed(run_dir, ckpt_sha, tau_syn)
        fixed_thresholds, thresholds_from = None, None
    else:
        windowed = {}
        fixed_thresholds, thresholds_from = read_dev_thresholds(run_dir, ckpt_sha)
    photos = photos_in(read_gtsdb(args.gtsdb), SPLITS[args.split])

    info = load_checkpoint_info(args.checkpoint)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    print(f"detecting with {args.checkpoint} (epoch {info.epoch}) on {len(photos)} {args.split} photos, "
          f"scales {scales}, {device}")
    records, seconds_per_photo = score_photos(model, photos, scales)

    thresholds = dev_thresholds(records, tau_syn) if fixed_thresholds is None else fixed_thresholds
    report = detect_report(records, thresholds, windowed, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES)
    write_json_atomic(out_path, {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": info.epoch,
        "run_id": args.checkpoint.parent.name,
        "scales": list(scales),
        "thresholds": thresholds,
        "thresholds_from": thresholds_from,  # null on dev: chosen here
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "report": report,
        "photos": [_photo_json(r) for r in records],
        "seconds_per_photo": seconds_per_photo,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"wrote {out_path}")  # A12: before any number, so a crash can't leave a seen-but-unrecorded test score
    print_summary(report, seconds_per_photo)


def refuse_test_rerun(run_dir: Path) -> None:
    """Exit with code 2 if this run's whole test photos were already scored (eval/test_guard.py)."""
    existing = scored_results(run_dir, "photos_test")
    if existing:
        print(f"refusing: whole test photos already scored for this run: {', '.join(map(str, existing))}\n"
              f"the test photos are scored once per checkpoint", file=sys.stderr)
        raise SystemExit(2)


def _read_result(path: Path, checkpoint_sha256: str, hint: str) -> dict:
    """Raises FileNotFoundError if path is missing, ValueError if it holds another checkpoint's result."""
    if not path.exists():
        raise FileNotFoundError(f"no stored result at {path}: {hint}")
    result = json.loads(path.read_text())
    if result["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"{path} was written for a different checkpoint (SHA-256 "
                         f"{result['checkpoint_sha256'][:12]}… vs {checkpoint_sha256[:12]}…)")
    return result


def read_tau_syn(run_dir: Path, checkpoint_sha256: str) -> np.float32:
    """The run's window τ from presence_val.json. Raises as _read_result."""
    val = _read_result(run_dir / "presence_val.json", checkpoint_sha256,
                       "run scripts/evaluate_presence.py --split val with this checkpoint first")
    return np.float32(val["tau"])


def read_windowed(run_dir: Path, checkpoint_sha256: str, tau_syn: np.float32) -> dict[tuple[int, int], bool]:
    """Each real_dev sign's windowed found outcome (correct & hit & presence >= τ_syn), keyed by
    (photo index, gt row index) through real_dev.npz's photo_index and sign_index. Raises as _read_result;
    also ValueError if presence_real_dev.json used another τ, if real_dev.npz changed since it was scored
    (SHA-256), or if their rows don't line up."""
    real = _read_result(run_dir / "presence_real_dev.json", checkpoint_sha256,
                        "run scripts/evaluate_presence.py --split real_dev with this checkpoint first")
    if np.float32(real["tau"]) != tau_syn:
        raise ValueError(f"presence_real_dev.json used τ {real['tau']}, presence_val.json has {float(tau_syn)}")
    data = real["data"]["real_dev"]
    if sha256_file(Path(data["file"])) != data["sha256"]:
        raise ValueError(f"{data['file']} changed since presence_real_dev.json was scored")
    with np.load(data["file"]) as z:
        photo, sign = z["photo_index"].astype(int), z["sign_index"].astype(int)
    pos = real["per_scene"]["positives"]
    if pos["photo_index"] != photo.tolist():
        raise ValueError("presence_real_dev.json's positives don't line up with real_dev.npz row for row")
    found = (np.asarray(pos["correct"], bool) & np.asarray(pos["hit"], bool)
             & (np.asarray(pos["presence"], np.float32) >= tau_syn))
    return {(int(p), int(s)): bool(f) for p, s, f in zip(photo, sign, found)}


def read_dev_thresholds(run_dir: Path, checkpoint_sha256: str) -> tuple[dict[str, float], dict]:
    """Every threshold from this run's detect_dev.json, and where it came from. Raises as _read_result;
    also ValueError if that file was made at scales other than PYRAMID_SCALES."""
    path = run_dir / OUT_FILES[("dev", PYRAMID_SCALES)]
    dev = _read_result(path, checkpoint_sha256, "score --split dev with this checkpoint first")
    if tuple(dev["scales"]) != PYRAMID_SCALES:
        raise ValueError(f"{path} used scales {dev['scales']}, not {list(PYRAMID_SCALES)}")
    return {k: float(v) for k, v in dev["thresholds"].items()}, {"file": str(path), "sha256": sha256_file(path)}


def dev_thresholds(records: list[PhotoRecord], tau_syn: np.float32) -> dict[str, float]:
    """τ_syn, τ_W and the reported FPPI points, from the false positives' scores alone (§2.5)."""
    fp = np.array([d.score for r in records for d, g in zip(r.kept, r.match.found) if g is None], np.float32)
    out = {TAU_SYN: float(tau_syn), TAU_W: float(threshold_at_fppi(fp, len(records), TAU_W_FPPI))}
    out |= {f"fppi_{f}": float(threshold_at_fppi(fp, len(records), f)) for f in REPORTED_FPPI}
    return out


def score_photos(model: torch.nn.Module, photos: list[RealPhoto],
                 scales: tuple[float, ...]) -> tuple[list[PhotoRecord], float]:
    """Detect, match and keep each sign's best raw score, photo by photo; also the mean seconds per photo
    for detect alone. Raises as load_photo / detect; nothing is returned on a partial failure."""
    records, seconds = [], 0.0
    for i, photo in enumerate(photos):
        img = load_photo(photo)
        start = time.perf_counter()
        result = detect(model, img, scales)  # ends on .cpu(), so the GPU has finished when this returns
        seconds += time.perf_counter() - start
        truth = photo_truth(photo)
        records.append(PhotoRecord(index=photo.index, truth=truth, kept=result.kept,
                                   match=match_photo(result.kept, truth), best_raw=best_raw_score(result.raw, truth)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(photos)} photos")
    return records, seconds / len(photos)


def _photo_json(r: PhotoRecord) -> dict:
    return {
        "index": r.index,
        "signs": [{"gt_index": s.gt_index, "box": list(s.box), "class_id": s.class_id, "long_side": s.long_side,
                   "stacked": s.stacked} for s in r.truth],
        "kept": [{"x": d.x, "y": d.y, "score": d.score, "class_id": d.class_id, "class_prob": d.class_prob,
                  "scale": d.scale, "found": f, "located": loc, "fp_kind": k.value if k is not None else None}
                 for d, f, loc, k in zip(r.kept, r.match.found, r.match.located, r.match.fp_kind)],
        "best_raw": {str(g): s for g, s in r.best_raw.items()},
    }


def print_summary(report: dict, seconds_per_photo: float) -> None:
    print(f"{report['n_photos']} photos, {report['n_signs']} signs, {seconds_per_photo:.2f} s per photo, "
          f"AP_point {report['ap_point']:.3f}")
    for name, at in report["at"].items():
        f = at["groups"]["all"]["found"]
        ci = "" if f["ci"] is None else f" [{100 * f['ci'][0]:.1f}, {100 * f['ci'][1]:.1f}]"
        print(f"  {name:>9} = {at['threshold']:.4f}: found {f['k']}/{f['n']} = {100 * f['value']:.1f}%{ci}, "
              f"FPPI {at['fppi']['value']:.2f}")


if __name__ == "__main__":
    main()
