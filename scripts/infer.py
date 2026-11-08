"""CLI: run a trained checkpoint on image files; print what sign it sees and where, and save overlays.

Usage:
    python scripts/infer.py --checkpoint outputs/checkpoints/<run>/best.pt photo1.jpg photo2.png
    python scripts/infer.py --checkpoint ... --size 256 big_photo.jpg   # larger scene, smaller-looking sign

Each image is center-cropped to a square and resized to --size x --size (default 128, the training
scene size), then scored. Prints one line per image and writes <out>/<image stem>_overlay.png.

The model always names one of the 43 classes, even when there is no sign; peak_prob (how strongly the
heatmap fires at its peak) is the hint for "probably nothing here". It was trained on signs of 16-48 px
in a 128 px scene, so a far-away sign in a shrunk-down photo may be too small to find.

Fails fast: every image is opened and the output names checked before anything runs, so a missing or
unreadable file, a non-multiple-of-4 --size, or two inputs with the same file stem stop the run with
nothing written. Raises FileNotFoundError / ValueError / RuntimeError for a bad checkpoint, as load_model.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no window backend needed
import matplotlib.pyplot as plt
import torch
from PIL import Image

from findobject.data.class_names import CLASS_NAMES
from findobject.inference.predict import predict
from findobject.inference.preprocess import to_scene
from findobject.training.checkpoint import load_model
from findobject.utils.device import pick_device
from findobject.viz.overlay import draw_prediction

DEFAULT_OUT = Path("outputs/infer")
BATCH_SIZE = 64
FIGURE_INCHES = 4
DPI = 128


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--size", type=int, default=128, help="scene side in px; multiple of 4")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args(argv)

    out_paths = overlay_paths(args.images, args.out)
    scenes = torch.stack([load_scene(p, args.size) for p in args.images])  # all inputs checked before any work

    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    preds = [p for batch in scenes.split(BATCH_SIZE) for p in predict(model, batch.to(device))]

    args.out.mkdir(parents=True, exist_ok=True)
    for path, scene, pred, out_path in zip(args.images, scenes, preds, out_paths):
        x, y = pred.peak_xy
        print(f"{path}: class {pred.class_id} {CLASS_NAMES[pred.class_id]!r} ({pred.class_prob:.1%}) "
              f"| peak at ({x:.0f}, {y:.0f}) of {args.size}px, peak_prob {pred.peak_prob:.2f}")
        save_overlay(scene, pred, out_path)
    print(f"wrote {len(out_paths)} overlay(s) to {args.out}")


def overlay_paths(images: list[Path], out_dir: Path) -> list[Path]:
    """<out_dir>/<stem>_overlay.png per input. Raises ValueError if two inputs share a stem
    (their overlays would overwrite each other)."""
    paths = [out_dir / f"{p.stem}_overlay.png" for p in images]
    if len(set(paths)) != len(paths):
        raise ValueError("two input images share a file name stem; their overlays would collide")
    return paths


def load_scene(path: Path, size: int) -> torch.Tensor:
    """Open one image as a [3, size, size] scene. Raises FileNotFoundError if missing,
    PIL.UnidentifiedImageError if not an image, ValueError for a bad size (see to_scene)."""
    with Image.open(path) as img:
        return to_scene(img, size)


def save_overlay(scene: torch.Tensor, pred, out_path: Path) -> None:
    """Render one scene's overlay to out_path (overwriting it). Raises OSError on write failure."""
    fig, ax = plt.subplots(figsize=(FIGURE_INCHES, FIGURE_INCHES + 0.4))
    draw_prediction(ax, scene, pred)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
