"""CLI: score a trained checkpoint on one split and record the result as JSON.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>/best.pt --split val
    python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>/best.pt --split test   # once per checkpoint

Writes outputs/eval/<run>/<checkpoint>_<split>.json with all metrics, per-class accuracy, the confusion
matrix, and the SHA-256 of the data file (so a number stays tied to the exact data that produced it).
Data dir and loss weight come from the config stored in the checkpoint, not from configs/.

Test guard: if a test result for this checkpoint already exists, exits with code 2 before evaluating,
unless --force. Val results are overwritten freely. Raises FileNotFoundError for a missing checkpoint or
data file, ValueError for malformed data, RuntimeError if the weights don't fit the model. The JSON is
written atomically, so a failed run leaves no partial result.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from findobject.data.dataset import SignScenes
from findobject.data.gtsrb import NUM_CLASSES
from findobject.eval.evaluate import evaluate
from findobject.eval.report import class_report
from findobject.training.checkpoint import load_checkpoint_info, load_model
from findobject.utils.device import pick_device

EVAL_ROOT = Path("outputs/eval")
BATCH_SIZE = 256
WORST_SHOWN = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--force", action="store_true", help="re-run test even if a result already exists")
    args = parser.parse_args()

    out_path = result_path(args.checkpoint, args.split)
    refuse_test_rerun(out_path, args.split, args.force)

    info = load_checkpoint_info(args.checkpoint)
    data_path = Path(info.config["data"]["processed_dir"]) / f"{args.split}.npz"
    device = pick_device("auto")
    model = load_model(args.checkpoint, device)
    loader = DataLoader(SignScenes(data_path), batch_size=BATCH_SIZE, num_workers=0)
    print(f"evaluating {args.checkpoint} (epoch {info.epoch}) on {data_path} ({len(loader.dataset)} scenes), {device}")

    result = evaluate(model, loader, device, info.config["train"]["heatmap_loss_weight"])
    report = class_report(result.y_true, result.y_pred, NUM_CLASSES)

    write_json_atomic(out_path, {
        "split": args.split,
        "n_scenes": len(result.y_true),
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": info.epoch,
        "run_id": args.checkpoint.parent.name,
        "metrics": result.scalars(),
        "val_metrics_at_save": info.metrics,
        "per_class": {"counts": report.counts, "accuracy": report.accuracy},
        "confusion": report.confusion,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })

    m = result.scalars()
    print(f"\n{args.split}: joint {m['joint_acc']:.4f} | class {m['class_acc']:.4f} | pointing {m['pointing_acc']:.4f} "
          f"| center err median {m['center_err_median']:.2f}px mean {m['center_err_mean']:.2f}px")
    present = [(acc, k, n) for k, (acc, n) in enumerate(zip(report.accuracy, report.counts)) if acc is not None]
    worst = ", ".join(f"{k} ({acc:.2f}, n={n})" for acc, k, n in sorted(present)[:WORST_SHOWN])
    print(f"worst classes (id, acc, n): {worst}")
    print(f"wrote {out_path}")


def result_path(checkpoint: Path, split: str) -> Path:
    """outputs/eval/<run_id>/<checkpoint stem>_<split>.json, with run_id = the checkpoint's folder name."""
    return EVAL_ROOT / checkpoint.parent.name / f"{checkpoint.stem}_{split}.json"


def refuse_test_rerun(path: Path, split: str, force: bool) -> None:
    """Exit with code 2 if this would overwrite an existing test result without --force.

    Test is meant to be looked at once; re-running it while tweaking turns it into a second val set.
    This is a guard against accidents, not a security boundary.
    """
    if split == "test" and path.exists() and not force:
        print(f"refusing: test result already exists at {path}\n"
              f"the test set is scored once per checkpoint; pass --force only if you mean to re-score it",
              file=sys.stderr)
        raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write to <path>.tmp, then rename over path. Creates parent dirs. Raises OSError on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
