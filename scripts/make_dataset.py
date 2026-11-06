"""CLI: generate the seeded train/val/test synthetic scenes and write data/processed/{train,val,test}.npz.

Usage: python scripts/make_dataset.py --config configs/baseline.yaml

Each .npz holds: images uint8 [N,S,S,3], class_ids int64 [N], boxes int16 [N,4] (x1,y1,x2,y2,
x2/y2 exclusive), heatmaps float16 [N,H,H]. Re-running overwrites with identical output.
Signs: train and val split the GTSRB training set by whole track (no near-duplicate frames
across them); test uses only the GTSRB test set. Backgrounds: the three splits use disjoint
COCO images.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from findobject.data.coco import list_backgrounds
from findobject.data.gtsrb import SignSample, load_sign, read_test_index, read_train_index, split_by_track
from findobject.data.synth import SynthConfig, make_scene

# Seed offset per split, keyed by name so adding a split never changes another split's output.
SPLIT_SEED_OFFSET = {"train": 0, "test": 1, "val": 2}


def generate(signs: list[SignSample], backgrounds: list[Path], n: int, seed: int, cfg: SynthConfig) -> dict:
    rng = np.random.default_rng(seed)
    s, h = cfg.scene_size, cfg.heatmap_size
    out = {
        "images": np.empty((n, s, s, 3), np.uint8),
        "class_ids": np.empty(n, np.int64),
        "boxes": np.empty((n, 4), np.int16),
        "heatmaps": np.empty((n, h, h), np.float16),
    }
    for i in range(n):
        sign = signs[rng.integers(len(signs))]
        with Image.open(backgrounds[rng.integers(len(backgrounds))]) as bg:
            scene = make_scene(load_sign(sign), bg, rng, cfg)
        out["images"][i] = scene.image
        out["class_ids"][i] = sign.class_id
        out["boxes"][i] = scene.box
        out["heatmaps"][i] = scene.heatmap
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}")
    return out


def save_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.config.read_text())["data"]
    synth_cfg = data_cfg["synth"]
    cfg = SynthConfig(**{**synth_cfg, "brightness": tuple(synth_cfg["brightness"])})
    raw = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bgs = list_backgrounds(raw / "coco/val2017", raw / "coco/annotations/instances_val2017.json")
    every = data_cfg["test_background_every"]
    train_signs, val_signs = split_by_track(
        read_train_index(raw / "gtsrb"), data_cfg["holdout_fraction"], seed=data_cfg["seed"]
    )
    splits = {
        "train": (train_signs, [b for i, b in enumerate(bgs) if i % every not in (0, 1)], data_cfg["n_train"]),
        "val": (val_signs, bgs[1::every], data_cfg["n_val"]),
        "test": (read_test_index(raw / "gtsrb"), bgs[::every], data_cfg["n_test"]),
    }
    print(f"{len(bgs)} backgrounds")
    for name, (signs, split_bgs, n) in splits.items():
        print(f"{name}: {n} scenes from {len(signs)} signs, {len(split_bgs)} backgrounds")
        arrays = generate(signs, split_bgs, n, seed=data_cfg["seed"] * 1000 + SPLIT_SEED_OFFSET[name], cfg=cfg)
        save_atomic(out_dir / f"{name}.npz", arrays)
        print(f"  wrote {out_dir / name}.npz")


if __name__ == "__main__":
    main()
