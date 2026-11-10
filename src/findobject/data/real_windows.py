"""Cut a training-shaped window (one labeled sign, scene_size square) out of a real photo. Pure.

Geometry follows the v2 spec (docs/v2-spec.md §2.2): scale the photo so the target's longer side is at
most cfg.sign_max (never enlarge), then place a random square window that holds the whole target and
touches no other labeled sign.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from PIL import Image

from findobject.data.synth import Scene, SynthConfig, box_heatmap

PLACEMENT_TRIES = 50

Box = tuple[int, int, int, int]  # x1, y1, x2, y2; x2/y2 exclusive


class Skip(Enum):
    OTHER_SIGN_IN_VIEW = "other_sign_in_view"
    NO_CLEAR_SPOT = "no_clear_spot"  # negative window (real_negatives.py): every try touched a labeled sign


@dataclass(frozen=True)
class WindowPlan:
    scale: float  # applied to the whole photo
    x0: int  # window top-left, in scaled-photo px
    y0: int
    box: Box  # target sign in window px


def plan_window(photo_size: tuple[int, int], boxes: Sequence[Box], target: int,
                rng: np.random.Generator, cfg: SynthConfig) -> WindowPlan | Skip:
    """Choose the scale and window for sign `target` among all labeled `boxes` (photo px) of one photo.

    Tries up to PLACEMENT_TRIES random positions from rng and keeps the first whose window overlaps no
    other box; returns Skip.OTHER_SIGN_IN_VIEW if none does. With no other box nearby the first try wins,
    so rng is advanced by exactly two draws.
    Raises IndexError for a bad target, ValueError if a box is empty or the scaled photo is smaller than
    the window.
    """
    if not 0 <= target < len(boxes):
        raise IndexError(f"target {target} not in [0, {len(boxes)})")
    for b in boxes:
        if not (b[0] < b[2] and b[1] < b[3]):
            raise ValueError(f"empty box {b}")
    x1, y1, x2, y2 = boxes[target]
    scale = min(1.0, cfg.sign_max / max(x2 - x1, y2 - y1))
    width, height = scaled_size(photo_size, scale)
    s = cfg.scene_size
    if width < s or height < s:
        raise ValueError(f"photo {photo_size} at scale {scale:.3f} is smaller than the {s}px window")

    tx1, ty1, tx2, ty2 = _scale_box(boxes[target], scale, (width, height))
    others = [_scale_box(b, scale, (width, height)) for i, b in enumerate(boxes) if i != target]
    # The window [x0, x0+s) must contain [tx1, tx2) and lie inside [0, width).
    x_lo, x_hi = max(0, tx2 - s), min(tx1, width - s)
    y_lo, y_hi = max(0, ty2 - s), min(ty1, height - s)
    if x_lo > x_hi or y_lo > y_hi:
        raise ValueError(f"target box {boxes[target]} does not fit a {s}px window at scale {scale:.3f}")
    for _ in range(PLACEMENT_TRIES):
        x0 = int(rng.integers(x_lo, x_hi + 1))
        y0 = int(rng.integers(y_lo, y_hi + 1))
        window = (x0, y0, x0 + s, y0 + s)
        if not any(_overlap(window, o) for o in others):
            return WindowPlan(scale, x0, y0, (tx1 - x0, ty1 - y0, tx2 - x0, ty2 - y0))
    return Skip.OTHER_SIGN_IN_VIEW


def cut_window(photo: Image.Image, plan: WindowPlan, cfg: SynthConfig) -> Scene:
    """Scale the photo by plan.scale (bilinear, like synthetic scenes) and crop the window.

    No blur, noise or brightness change: the photo keeps its own. Does not mutate photo.
    Raises ValueError if the window does not lie inside the scaled photo (plan made for another photo).
    """
    size = scaled_size(photo.size, plan.scale)
    s = cfg.scene_size
    if not (0 <= plan.x0 <= size[0] - s and 0 <= plan.y0 <= size[1] - s):
        raise ValueError(f"window at ({plan.x0}, {plan.y0}) does not fit the scaled photo {size}")
    img = photo.convert("RGB")
    if plan.scale != 1.0:
        img = img.resize(size, Image.BILINEAR)
    window = img.crop((plan.x0, plan.y0, plan.x0 + s, plan.y0 + s))
    return Scene(image=np.asarray(window, dtype=np.uint8).copy(), box=plan.box, heatmap=box_heatmap(plan.box, cfg))


def scaled_size(photo_size: tuple[int, int], scale: float) -> tuple[int, int]:
    """(width, height) of the photo after scaling; exact at scale 1."""
    return round(photo_size[0] * scale), round(photo_size[1] * scale)


def _scale_box(box: Box, scale: float, bounds: tuple[int, int]) -> Box:
    """Scale a box outward (floor the start, ceil the end) so it still covers every sign pixel, then clip
    the end to the scaled photo (ceil can overshoot by one where the photo's own size was rounded down)."""
    x1, y1, x2, y2 = box
    w, h = bounds
    return (math.floor(x1 * scale), math.floor(y1 * scale),
            min(w, math.ceil(x2 * scale)), min(h, math.ceil(y2 * scale)))


def _overlap(a: Box, b: Box) -> bool:
    """True if the two half-open boxes share at least one pixel."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
