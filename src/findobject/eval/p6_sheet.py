"""Choose and draw the P6 contact sheet: numbered windows with no scores (docs/v3a-spec.md §2.9). Pure.

The sheet holds the N_TOP highest-presence negatives and N_CONTROL drawn uniformly from the rest, in a
seeded shuffled order. Nothing drawn on it reveals which group a tile is in; that is only in the key.
"""

from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

N_TOP = 20
N_CONTROL = 20
TILE_ZOOM = 2  # tiles are drawn at 2x so small signs are easier to judge
COLUMNS = 5
LABEL_PX = 18  # strip above each tile holding its number, so the number never covers a pixel


class Tile(NamedTuple):
    number: int  # 1-based, as printed on the sheet
    row: int  # index into the negatives file
    top: bool  # one of the N_TOP highest-presence windows


def choose_tiles(presence: np.ndarray, rng: np.random.Generator) -> list[Tile]:
    """The N_TOP highest-presence rows (ties broken by lower row index) plus N_CONTROL distinct rows drawn
    uniformly from the rest, numbered in a random order from rng.
    Raises ValueError if there are fewer than N_TOP + N_CONTROL rows or presence is not 1-D."""
    presence = np.asarray(presence)
    if presence.ndim != 1 or len(presence) < N_TOP + N_CONTROL:
        raise ValueError(f"need a 1-D array of at least {N_TOP + N_CONTROL} scores, got {presence.shape}")
    by_score = np.argsort(-presence.astype(np.float64), kind="stable")
    top, rest = by_score[:N_TOP], np.sort(by_score[N_TOP:])
    control = rng.choice(rest, size=N_CONTROL, replace=False)
    rows = np.concatenate([top, control])
    is_top = np.arange(len(rows)) < N_TOP
    order = rng.permutation(len(rows))
    return [Tile(i + 1, int(rows[j]), bool(is_top[j])) for i, j in enumerate(order)]


def render_sheet(images: np.ndarray, tiles: list[Tile]) -> Image.Image:
    """Grid of tiles in number order, each zoomed TILE_ZOOM times with its number above it.
    images: uint8 [N, S, S, 3], the negatives file's images. Raises IndexError for a row outside images."""
    s = images.shape[1] * TILE_ZOOM
    n_rows = -(-len(tiles) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * s, n_rows * (s + LABEL_PX)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, tile in enumerate(sorted(tiles, key=lambda t: t.number)):
        x, y = (k % COLUMNS) * s, (k // COLUMNS) * (s + LABEL_PX)
        draw.text((x + 4, y + 3), str(tile.number), fill="black")
        sheet.paste(Image.fromarray(images[tile.row]).resize((s, s), Image.NEAREST), (x, y + LABEL_PX))
    return sheet
