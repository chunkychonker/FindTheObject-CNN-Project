"""Choose and draw the v3b hand-audit sheet of whole-photo false positives (docs/v3b-spec.md §2.6). Pure.

The sheet holds the N_AUDIT highest-scoring `background` false positives at a threshold, numbered in a
seeded random order, each a crop of its photo centered on the detection's point. No score, rank, class
or photo number is drawn: those are only in the key, so each tile is judged from the image alone.
"""

from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

N_AUDIT = 40
CROP_AT_SCALE1 = 96  # native px; a detection at scale s gets CROP_AT_SCALE1 / s, about what the model saw
TILE_PX = 192
COLUMNS = 5
LABEL_PX = 18  # strip above each tile holding its number
TICK = (0.30, 0.42)  # tick marks from 30% to 42% of the tile from its center: the point itself stays clear
PAD_GRAY = 128  # outside the photo


class FpTile(NamedTuple):
    number: int  # 1-based, as printed on the sheet
    photo_index: int
    x: float  # the detection's point, native px
    y: float
    scale: float
    score: float


def choose_fps(photos: list[dict], threshold: float, rng: np.random.Generator) -> list[FpTile]:
    """The N_AUDIT highest-scoring background false positives with score >= threshold, from detect_photos'
    per-photo records ({"index", "kept": [{x, y, score, scale, fp_kind}, ...]}). Ties go to the lower photo
    index, then the earlier detection. Numbered in a random order from rng.
    Raises ValueError if fewer than N_AUDIT qualify."""
    fps = [(-k["score"], p["index"], j, k) for p in photos for j, k in enumerate(p["kept"])
           if k["fp_kind"] == "background" and k["score"] >= threshold]
    if len(fps) < N_AUDIT:
        raise ValueError(f"need {N_AUDIT} background false positives at {threshold}, got {len(fps)}")
    top = sorted(fps, key=lambda f: f[:3])[:N_AUDIT]
    order = rng.permutation(N_AUDIT)
    return [FpTile(i + 1, top[j][1], top[j][3]["x"], top[j][3]["y"], top[j][3]["scale"], top[j][3]["score"])
            for i, j in enumerate(order)]


def crop_tile(photo: Image.Image, tile: FpTile) -> Image.Image:
    """A square crop of side CROP_AT_SCALE1 / scale centered on the point (gray outside the photo), resized
    to TILE_PX with bilinear, with four tick marks around the center."""
    side = CROP_AT_SCALE1 / tile.scale
    x0, y0 = round(tile.x - side / 2), round(tile.y - side / 2)
    canvas = Image.new("RGB", (round(side), round(side)), (PAD_GRAY,) * 3)
    canvas.paste(photo, (-x0, -y0))
    out = canvas.resize((TILE_PX, TILE_PX), Image.BILINEAR)
    draw, c = ImageDraw.Draw(out), TILE_PX / 2
    a, b = TICK[0] * TILE_PX, TICK[1] * TILE_PX
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        draw.line([(c + dx * a, c + dy * a), (c + dx * b, c + dy * b)], fill=(255, 0, 0), width=2)
    return out


def render_fp_sheet(photos: dict[int, Image.Image], tiles: list[FpTile]) -> Image.Image:
    """Grid of crop_tile tiles in number order, each with its number above it.
    Raises KeyError if a tile's photo is not in photos."""
    n_rows = -(-len(tiles) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * TILE_PX, n_rows * (TILE_PX + LABEL_PX)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, tile in enumerate(sorted(tiles, key=lambda t: t.number)):
        x, y = (k % COLUMNS) * TILE_PX, (k // COLUMNS) * (TILE_PX + LABEL_PX)
        draw.text((x + 4, y + 3), str(tile.number), fill="black")
        sheet.paste(crop_tile(photos[tile.photo_index], tile), (x, y + LABEL_PX))
    return sheet
