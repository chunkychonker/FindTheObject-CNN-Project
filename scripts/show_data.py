"""CLI: render PNG figures of a processed split so you can see what the model trains on.

Usage: python scripts/show_data.py [--split train|val] [--out outputs/figures] [--seed 0]

Writes four figures to --out (overwriting earlier ones):
    data_anatomy_<split>.png     one scene up close: image, box, heatmap target, overlay, sign pixels
    data_samples_<split>.png     24 random scenes with box and class name
    data_classes_<split>.png     one typical sign per class, cropped at the model's resolution
    data_difficulty_<split>.png  signs sorted by brightness and by size

test is deliberately not offered: looking at test scenes would let them influence decisions.
Read-only on data. Raises FileNotFoundError if the .npz is missing.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no window backend needed
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

CLASS_NAMES = (
    "speed 20", "speed 30", "speed 50", "speed 60", "speed 70", "speed 80", "end speed 80", "speed 100",
    "speed 120", "no passing", "no passing trucks", "right-of-way", "priority road", "yield", "stop",
    "no vehicles", "trucks prohibited", "no entry", "general caution", "curve left", "curve right",
    "double curve", "bumpy road", "slippery road", "road narrows right", "road work", "traffic signals",
    "pedestrians", "children crossing", "bicycles crossing", "ice/snow", "wild animals", "end of limits",
    "turn right ahead", "turn left ahead", "ahead only", "straight or right", "straight or left",
    "keep right", "keep left", "roundabout", "end no passing", "end no passing trucks",
)
STRIP = 10  # signs per row in the difficulty figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    processed = Path(yaml.safe_load(args.config.read_text())["data"]["processed_dir"])
    with np.load(processed / f"{args.split}.npz") as z:
        images, class_ids, boxes, heatmaps = z["images"], z["class_ids"], z["boxes"].astype(int), z["heatmaps"]
    sign_brightness = np.array([_crop(images[i], boxes[i]).mean() for i in range(len(images))])
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    figures = {
        "anatomy": anatomy(images, class_ids, boxes, heatmaps, sign_brightness),
        "samples": samples(images, class_ids, boxes, rng),
        "classes": classes(images, class_ids, boxes, sign_brightness),
        "difficulty": difficulty(images, boxes, sign_brightness),
    }
    for name, fig in figures.items():
        path = args.out / f"data_{name}_{args.split}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def anatomy(images, class_ids, boxes, heatmaps, sign_brightness):
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    i = int(np.flatnonzero((side >= 32) & (side <= 40) & (sign_brightness > 100))[0])  # a clear, mid-size sign
    img, box, hm = images[i], boxes[i], heatmaps[i].astype(np.float32)
    s, h = img.shape[0], hm.shape[0]
    r, c = np.unravel_index(hm.argmax(), hm.shape)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.6))
    fig.suptitle(f"scene #{i}: class {class_ids[i]} ({CLASS_NAMES[class_ids[i]]})", fontsize=14)
    ax[0].imshow(img)
    ax[0].set_title(f"1. image the model gets\nuint8 [{s}, {s}, 3]")
    ax[1].imshow(img)
    ax[1].add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1],
                              fill=False, color="lime", lw=2))
    ax[1].set_title(f"2. box (x1, y1, x2, y2)\n{tuple(box.tolist())}")
    ax[2].imshow(hm, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title(f"3. heatmap target [{h}, {h}]\npeak {hm.max():.2f} at row {r}, col {c}")
    ax[3].imshow(img)
    ax[3].imshow(hm, cmap="magma", vmin=0, vmax=1, alpha=0.6, extent=(-0.5, s - 0.5, s - 0.5, -0.5))
    ax[3].set_title(f"4. target over image\neach cell = {s // h}x{s // h} px")
    ax[4].imshow(_crop(img, box), interpolation="nearest")
    ax[4].set_title(f"5. the sign, actual pixels\n{box[2] - box[0]}x{box[3] - box[1]} px")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    return fig


def samples(images, class_ids, boxes, rng, rows=4, cols=6):
    idx = rng.choice(len(images), rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for a, i in zip(axes.flat, idx):
        a.imshow(images[i])
        b = boxes[i]
        a.add_patch(Rectangle((b[0] - 0.5, b[1] - 0.5), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=1.5))
        a.set_title(f"#{i}  {class_ids[i]}: {CLASS_NAMES[class_ids[i]]}", fontsize=9)
        a.axis("off")
    fig.suptitle("random scenes (green = sign box)", fontsize=14)
    return fig


def classes(images, class_ids, boxes, sign_brightness, cols=8):
    """Per class, the scene whose sign brightness is the class median: a typical, not a best-case, example."""
    n = len(CLASS_NAMES)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    for k, a in enumerate(axes.flat):
        a.axis("off")
        if k >= n:
            continue
        members = np.flatnonzero(class_ids == k)
        if len(members) == 0:
            a.set_title(f"{k}: none", fontsize=8)
            continue
        i = members[np.argsort(sign_brightness[members])[len(members) // 2]]
        a.imshow(_crop(images[i], boxes[i]), interpolation="nearest")
        a.set_title(f"{k}: {CLASS_NAMES[k]}\nn={len(members)}", fontsize=8)
    fig.suptitle("one typical sign per class, at the pixel size the model sees", fontsize=14)
    return fig


def difficulty(images, boxes, sign_brightness):
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    fig, axes = plt.subplots(2, STRIP, figsize=(STRIP * 1.9, 4.8))
    for row, (key, label, fmt) in enumerate([(sign_brightness, "brightness", "{:.0f}"), (side, "size", "{} px")]):
        order = np.argsort(key, kind="stable")
        picks = order[np.linspace(0, len(order) - 1, STRIP).astype(int)]  # evenly spaced percentiles
        for a, i, pct in zip(axes[row], picks, np.linspace(0, 100, STRIP)):
            a.imshow(_crop(images[i], boxes[i]), interpolation="nearest")
            a.set_title(f"p{pct:.0f}: " + fmt.format(key[i]), fontsize=8)
            a.axis("off")
        axes[row][0].text(-0.15, 0.5, label, transform=axes[row][0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=11)
    fig.suptitle("signs at evenly spaced percentiles: darkest -> brightest (top), smallest -> largest (bottom)",
                 fontsize=12)
    return fig


def _crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    return img[box[1]:box[3], box[0]:box[2]]


if __name__ == "__main__":
    main()
