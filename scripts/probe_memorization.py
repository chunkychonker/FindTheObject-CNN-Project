"""CLI: split a checkpoint's train/val gap into scene memorization and sign memorization.

Usage:
    python scripts/probe_memorization.py --checkpoint outputs/checkpoints/<run>/best.pt [--n 5000] [--seed 987654]

Scores the checkpoint on two sets of scenes it has never seen:
    fresh  --n new scenes built from TRAINING signs and backgrounds (FreshScenes, epoch 0, --seed;
           no sign augmentation, so they look like the stored splits)
    val    the stored val.npz: unseen signs AND unseen backgrounds
Train accuracy minus fresh = scene memorization (it learned the exact training images).
Fresh minus val ("sign gap") = sign memorization (it knows these particular physical signs).
Weak classes (<= 12 training tracks) are reported separately: few tracks, most to memorize.

Writes outputs/eval/<run>/<checkpoint stem>_probe.json atomically (overwrites; val-side only, so
rerunning is harmless). Never touches test. Data settings come from the config stored in the
checkpoint. Deterministic for a given checkpoint, --n and --seed (up to MPS float noise).
Raises FileNotFoundError / ValueError / RuntimeError for a bad checkpoint or data (as load_model,
SignScenes and split_sources do).
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from findobject.data.data_config import synth_config
from findobject.data.dataset import SignScenes
from findobject.data.fresh import FreshScenes
from findobject.data.splits import split_sources
from findobject.eval.evaluate import evaluate
from findobject.eval.memorization import WEAK_MAX_TRACKS, subset_class_acc, tracks_per_class, weak_classes
from findobject.training.checkpoint import load_checkpoint_info, load_model
from findobject.utils.device import pick_device

EVAL_ROOT = Path("outputs/eval")
BATCH_SIZE = 256
PROBE_EPOCH = 0  # training epochs are 1-based, so epoch 0 never appears in a training run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=987654)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.n < 1 or args.seed < 0:
        parser.error("--n must be >= 1 and --seed >= 0")

    info = load_checkpoint_info(args.checkpoint)
    data_cfg = info.config["data"]
    weight = info.config["train"]["heatmap_loss_weight"]
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)

    train_source = split_sources(data_cfg)["train"]
    fresh_ds = FreshScenes(train_source, synth_config(data_cfg), args.n, args.seed, PROBE_EPOCH)
    val_path = Path(data_cfg["processed_dir"]) / "val.npz"
    print(f"probing {args.checkpoint} (epoch {info.epoch}) on {args.n} fresh training-sign scenes and {val_path}")

    fresh = evaluate(model, DataLoader(fresh_ds, batch_size=BATCH_SIZE), device, weight)
    val = evaluate(model, DataLoader(SignScenes(val_path), batch_size=BATCH_SIZE), device, weight)
    weak = weak_classes(train_source.signs)
    tracks = tracks_per_class(train_source.signs)

    summary = {
        "fresh_train_signs": fresh.scalars(),
        "val": val.scalars(),
        "sign_gap_joint": fresh.joint_acc - val.joint_acc,
        "sign_gap_class": fresh.class_acc - val.class_acc,
        "weak_classes": {str(k): tracks[k] for k in weak},
        "weak_class_acc": {"fresh": subset_class_acc(fresh.y_true, fresh.y_pred, weak),
                           "val": subset_class_acc(val.y_true, val.y_pred, weak)},
    }
    out_path = EVAL_ROOT / args.checkpoint.parent.name / f"{args.checkpoint.stem}_probe.json"
    write_json_atomic(out_path, {
        **summary,
        "n_fresh": args.n, "fresh_seed": args.seed, "fresh_epoch": PROBE_EPOCH, "weak_max_tracks": WEAK_MAX_TRACKS,
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": info.epoch, "run_id": args.checkpoint.parent.name,
        "device": str(device), "torch_version": str(torch.__version__),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })

    print(f"fresh (training signs): joint {fresh.joint_acc:.4f} | class {fresh.class_acc:.4f}")
    print(f"val   (unseen signs):   joint {val.joint_acc:.4f} | class {val.class_acc:.4f}")
    print(f"sign gap (fresh - val): joint {summary['sign_gap_joint'] * 100:+.2f} pts | "
          f"class {summary['sign_gap_class'] * 100:+.2f} pts")
    print(f"weak classes {weak} (<= {WEAK_MAX_TRACKS} tracks): class acc fresh "
          f"{summary['weak_class_acc']['fresh']:.4f} | val {summary['weak_class_acc']['val']:.4f}")
    print(f"wrote {out_path}")


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write to <path>.tmp, then rename over path. Creates parent dirs. Raises OSError on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
