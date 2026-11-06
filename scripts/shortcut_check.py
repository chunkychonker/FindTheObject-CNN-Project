"""Diagnostic: does the heatmap head detect *signs*, or *any pasted rectangle*?

Usage: python scripts/shortcut_check.py --checkpoint outputs/checkpoints/<run>/best.pt [--n 1000]

Builds fresh scenes from the val ingredients (held-out GTSRB tracks, val COCO backgrounds) with
the same size/brightness/blur/noise recipe as data/synth.py. Every background is rendered in
paired variants that differ only in what is pasted on it:

    empty            nothing pasted
    sign             one real sign (control: should match val pointing accuracy)
    decoy:<kind>     one non-sign rectangle, pasted exactly like a sign
    sign+<kind>      the sign and the decoy, non-overlapping

Decoy kinds: bg_patch (crop of a different COCO photo), shuffled_sign (a sign's pixels permuted:
same colors, no shape), flat (solid patch of a sign's mean color), other_sign (a second real
sign; positive control, expected to fire).

Prints a table and writes an example panel PNG. Read-only on data and checkpoints; deterministic
for a given --seed up to MPS float noise. Raises FileNotFoundError if the checkpoint or raw data
is missing.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from findobject.data.coco import list_backgrounds
from findobject.data.gtsrb import SignSample, load_sign, read_train_index, split_by_track
from findobject.data.synth import SynthConfig
from findobject.eval.heatmap_metrics import peak_centers
from findobject.training.checkpoint import load_model
from findobject.utils.device import pick_device

DECOYS = ("bg_patch", "shuffled_sign", "flat", "other_sign")
VARIANTS = ("empty", "sign", *(f"decoy:{d}" for d in DECOYS), *(f"sign+{d}" for d in DECOYS))
FIRE_THRESHOLD = 0.5  # a heatmap peak above this counts as "the model says there is a sign here"
BATCH = 256
PANEL_ROWS = 6

Box = tuple[int, int, int, int]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--n", type=int, default=1000, help="backgrounds (each rendered in every variant)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--panel-out", type=Path, default=Path("outputs/shortcut_check.png"))
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.config.read_text())["data"]
    synth = data_cfg["synth"]
    cfg = SynthConfig(**{**synth, "brightness": tuple(synth["brightness"])})
    raw = Path(data_cfg["raw_dir"])
    _, val_signs = split_by_track(read_train_index(raw / "gtsrb"), data_cfg["holdout_fraction"], seed=data_cfg["seed"])
    bgs = list_backgrounds(raw / "coco/val2017", raw / "coco/annotations/instances_val2017.json")
    val_bgs = bgs[1::data_cfg["test_background_every"]]  # same rule as make_dataset.py

    print(f"building {args.n} backgrounds x {len(VARIANTS)} variants from {len(val_signs)} val signs")
    images, sign_boxes, decoy_boxes = build(val_signs, val_bgs, args.n, np.random.default_rng(args.seed), cfg)

    device = pick_device("auto")
    model = load_model(args.checkpoint, device)
    probs, confs = {}, {}
    for v in VARIANTS:
        probs[v], confs[v] = predict(model, images[v], device)

    report(probs, confs, sign_boxes, decoy_boxes, cfg.scene_size)
    args.panel_out.parent.mkdir(parents=True, exist_ok=True)
    panel(images, probs, sign_boxes, decoy_boxes).save(args.panel_out)
    print(f"\nexample panel: {args.panel_out}")


# --- scene building (mirrors synth.make_scene, but with control over what gets pasted) ---

def build(signs: list[SignSample], bgs: list[Path], n: int, rng: np.random.Generator, cfg: SynthConfig):
    s = cfg.scene_size
    images = {v: np.empty((n, s, s, 3), np.uint8) for v in VARIANTS}
    sign_boxes = np.empty((n, 4), np.int64)
    decoy_boxes = np.empty((n, 4), np.int64)
    for i in range(n):
        canvas = _square(bgs[rng.integers(len(bgs))], s, rng)
        a = signs[rng.integers(len(signs))]
        b = signs[rng.integers(len(signs))]
        while b.class_id == a.class_id:
            b = signs[rng.integers(len(signs))]

        sign = _prep(load_sign(a), rng, cfg)
        other = _prep(load_sign(b), rng, cfg)  # decoys take this patch's size and colors
        w, h = other.size
        decoys = {
            "bg_patch": ImageEnhance.Brightness(
                _square(bgs[rng.integers(len(bgs))], max(w, h), rng).crop((0, 0, w, h))
            ).enhance(rng.uniform(*cfg.brightness)),
            "shuffled_sign": _shuffle_pixels(other, rng),
            "flat": Image.new("RGB", (w, h), tuple(int(c) for c in np.asarray(other).reshape(-1, 3).mean(0))),
            "other_sign": other,
        }
        sb, db = _two_boxes(sign.size, (w, h), s, rng)
        sign_boxes[i], decoy_boxes[i] = sb, db

        blur = rng.uniform(0, cfg.blur_max_radius) if rng.random() < cfg.blur_prob else None
        noise = rng.normal(0.0, rng.uniform(0, cfg.noise_max_std), (s, s, 3))  # shared across variants
        finish = lambda pastes: _finish(canvas, pastes, blur, noise)
        images["empty"][i] = finish([])
        images["sign"][i] = finish([(sign, sb)])
        for d, patch in decoys.items():
            images[f"decoy:{d}"][i] = finish([(patch, db)])
            images[f"sign+{d}"][i] = finish([(sign, sb), (patch, db)])
    return images, sign_boxes, decoy_boxes


def _square(path: Path, size: int, rng: np.random.Generator) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
    side = min(img.size)
    x, y = int(rng.integers(0, img.width - side + 1)), int(rng.integers(0, img.height - side + 1))
    return img.crop((x, y, x + side, y + side)).resize((size, size), Image.BILINEAR)


def _prep(sign: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Image.Image:
    long_side = int(rng.integers(cfg.sign_min, cfg.sign_max + 1))
    scale = long_side / max(sign.size)
    sign = sign.resize((max(1, round(sign.width * scale)), max(1, round(sign.height * scale))), Image.BILINEAR)
    return ImageEnhance.Brightness(sign).enhance(rng.uniform(*cfg.brightness))


def _shuffle_pixels(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img)
    flat = rng.permutation(arr.reshape(-1, 3))
    return Image.fromarray(flat.reshape(arr.shape))


def _two_boxes(size_a, size_b, s: int, rng: np.random.Generator) -> tuple[Box, Box]:
    """Random non-overlapping placements. Always terminates: two patches <= 48 px fit in 128 px."""
    def place(w, h):
        x, y = int(rng.integers(0, s - w + 1)), int(rng.integers(0, s - h + 1))
        return (x, y, x + w, y + h)

    a = place(*size_a)
    while True:
        b = place(*size_b)
        if b[0] >= a[2] or a[0] >= b[2] or b[1] >= a[3] or a[1] >= b[3]:
            return a, b


def _finish(canvas: Image.Image, pastes, blur, noise) -> np.ndarray:
    img = canvas.copy()
    for patch, box in pastes:
        img.paste(patch, box[:2])
    if blur is not None:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)


# --- model + metrics ---

@torch.no_grad()
def predict(model, images: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """-> heatmap probs [N, H, W] and top-class softmax confidence [N], on CPU."""
    probs, confs = [], []
    for i in range(0, len(images), BATCH):
        x = torch.from_numpy(images[i:i + BATCH]).to(device).permute(0, 3, 1, 2).float().div_(255)
        out = model(x)
        probs.append(torch.sigmoid(out.heatmap_logits).squeeze(1).cpu())
        confs.append(torch.softmax(out.class_logits, dim=1).max(dim=1).values.cpu())
    return torch.cat(probs), torch.cat(confs)


def _in_box(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    return (points[:, 0] >= boxes[:, 0]) & (points[:, 0] < boxes[:, 2]) & \
           (points[:, 1] >= boxes[:, 1]) & (points[:, 1] < boxes[:, 3])


def _box_max(probs: torch.Tensor, boxes: torch.Tensor, scene_size: int) -> torch.Tensor:
    """[N] highest heatmap prob among cells whose centers fall inside each box."""
    h = probs.shape[-1]
    centers = (torch.arange(h) + 0.5) * (scene_size / h)
    mx = (centers[None] >= boxes[:, 0:1]) & (centers[None] < boxes[:, 2:3])  # [N, W]
    my = (centers[None] >= boxes[:, 1:2]) & (centers[None] < boxes[:, 3:4])  # [N, H]
    mask = my[:, :, None] & mx[:, None, :]
    return probs.masked_fill(~mask, 0).flatten(1).max(1).values


def report(probs, confs, sign_boxes, decoy_boxes, scene_size: int) -> None:
    sb, db = torch.from_numpy(sign_boxes), torch.from_numpy(decoy_boxes)
    print(f"\n{'variant':<22}{'peak p':>8}{'peak>0.5':>10}{'@sign':>8}{'@decoy':>8}"
          f"{'max p in sign':>15}{'max p in decoy':>16}{'cls conf':>10}")
    for v in VARIANTS:
        p = probs[v]
        peak = p.flatten(1).max(1).values
        where = peak_centers(p, scene_size)
        has_sign = v.startswith("sign")
        has_decoy = "decoy" in v or "+" in v
        col = lambda ok, x: f"{x:.3f}" if ok else "-"
        print(f"{v:<22}{peak.median().item():>8.3f}{(peak > FIRE_THRESHOLD).float().mean().item():>10.3f}"
              f"{col(has_sign, _in_box(where, sb).float().mean().item()):>8}"
              f"{col(has_decoy, _in_box(where, db).float().mean().item()):>8}"
              f"{col(has_sign, _box_max(p, sb, scene_size).median().item()):>15}"
              f"{col(has_decoy, _box_max(p, db, scene_size).median().item()):>16}"
              f"{confs[v].median().item():>10.3f}")
    print("\npeak p / max p / cls conf are medians over scenes; peak>0.5, @sign, @decoy are fractions of scenes.")


def panel(images, probs, sign_boxes, decoy_boxes) -> Image.Image:
    """Rows = scenes. Column pairs (image | heatmap overlay) for: sign+bg_patch, decoy:bg_patch, empty.
    Green box = real sign, red box = decoy."""
    cols = ("sign+bg_patch", "decoy:bg_patch", "empty")
    s, z = 128, 2
    out = Image.new("RGB", (len(cols) * 2 * s * z, PANEL_ROWS * s * z), "white")
    for r in range(PANEL_ROWS):
        for c, v in enumerate(cols):
            img = Image.fromarray(images[v][r]).resize((s * z, s * z), Image.NEAREST)
            heat = (probs[v][r].numpy() * 255).astype(np.uint8)
            heat = Image.fromarray(heat).resize((s * z, s * z), Image.NEAREST)
            overlay = Image.blend(img.convert("L").convert("RGB"),
                                  Image.merge("RGB", (heat, Image.new("L", heat.size), Image.new("L", heat.size))), 0.6)
            for tile in (img, overlay):
                d = ImageDraw.Draw(tile)
                if v.startswith("sign"):
                    d.rectangle([x * z for x in sign_boxes[r]], outline=(0, 255, 0), width=2)
                if "decoy" in v or "+" in v:
                    d.rectangle([x * z for x in decoy_boxes[r]], outline=(255, 0, 0), width=2)
            out.paste(img, ((2 * c) * s * z, r * s * z))
            out.paste(overlay, ((2 * c + 1) * s * z, r * s * z))
    return out


if __name__ == "__main__":
    main()
