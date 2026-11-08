"""CLI: draw what a config's training scenes look like, augmentation included, for an eyeball check.

Usage:
    python scripts/show_train_scenes.py --config configs/peak_fresh_aug.yaml [--epoch 1] [--out outputs/figures]

Writes two PNGs to --out (overwriting earlier ones), named after train.run_name:
    train_scenes_<run>.png   the first 24 training scenes of --epoch, box in green, class name on top
    sign_variants_<run>.png  one training sign drawn 12 times with the config's augmentation: if a
                             warp looks implausible or the box stops hugging the sign, it shows here

Fresh mode only: fixed-mode scenes are already visible with show_data.py. Same config and epoch ->
same pictures as that epoch of training sees (FreshScenes is seeded by (train.seed, epoch, i)).
Raises ValueError for a config whose data.train_scenes isn't fresh; FileNotFoundError for missing raw data.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no window backend needed
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from findobject.data.class_names import CLASS_NAMES
from findobject.data.data_config import train_synth_config
from findobject.data.gtsrb import load_sign
from findobject.data.sign_augment import augment_sign
from findobject.data.splits import split_sources
from findobject.training.train_data import TrainScenes, epoch_datasets

ROWS, COLS = 4, 6
VARIANTS = 12
SIGN_INDEX = 0  # which training sign to show in the variants figure
DPI = 110


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, run = cfg["data"], cfg["train"]["run_name"]
    if TrainScenes(data_cfg.get("train_scenes", TrainScenes.FIXED.value)) is not TrainScenes.FRESH:
        raise ValueError("show_train_scenes needs data.train_scenes: fresh; use show_data.py for stored scenes")
    dataset = epoch_datasets(data_cfg, cfg["train"]["seed"])(args.epoch)
    synth = train_synth_config(data_cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    fig = scene_grid([dataset[i] for i in range(ROWS * COLS)])
    fig.suptitle(f"{run}: first {ROWS * COLS} training scenes of epoch {args.epoch} (green = box)", fontsize=14)
    _save(fig, args.out / f"train_scenes_{run}.png")

    sign = split_sources(data_cfg)["train"].signs[SIGN_INDEX]
    fig = sign_variants(load_sign(sign), synth.sign_augment)
    fig.suptitle(f"{run}: one training sign ({CLASS_NAMES[sign.class_id]}), original + {VARIANTS - 1} augmented draws",
                 fontsize=12)
    _save(fig, args.out / f"sign_variants_{run}.png")


def scene_grid(samples: list) -> plt.Figure:
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3.3))
    for a, s in zip(axes.flat, samples):
        a.imshow(s.image.permute(1, 2, 0).numpy())
        x1, y1, x2, y2 = s.box.tolist()
        a.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.5))
        a.set_title(f"{int(s.class_id)}: {CLASS_NAMES[int(s.class_id)]}", fontsize=9)
        a.axis("off")
    return fig


def sign_variants(sign: Image.Image, aug) -> plt.Figure:
    """Each variant on a mid-grey square so the mask edge (where the background would show) is visible."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, VARIANTS // 2, figsize=(VARIANTS // 2 * 2.2, 2 * 2.4))
    for k, a in enumerate(axes.flat):
        rgb, mask = (sign.convert("RGB"), Image.new("L", sign.size, 255)) if k == 0 else augment_sign(sign, rng, aug)
        tile = Image.new("RGB", rgb.size, (128, 128, 128))
        tile.paste(rgb, (0, 0), mask)
        a.imshow(tile, interpolation="nearest")
        a.set_title("original" if k == 0 else f"draw {k}", fontsize=8)
        a.axis("off")
    return fig


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
