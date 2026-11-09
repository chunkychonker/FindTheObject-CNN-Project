"""Plan and cut real-photo windows that hold no labeled sign (docs/v3a-spec.md §2.4).

Each photo gets one window per Stratum, scaled like a positive window and kept clear of every labeled
box. Reuses v2's WindowPlan, cut_window and Skip so a negative is cut exactly the way a positive is.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from findobject.data.gtsdb import RealPhoto, load_photo
from findobject.data.real_windows import (PLACEMENT_TRIES, Box, Skip, WindowPlan, _overlap, _scale_box, cut_window,
                                          scaled_size)
from findobject.data.synth import SynthConfig

NEG_STREAM = 12  # RNG stream; WINDOW_STREAM (10) and HYBRID_STREAM (11) are v2's, in real_scenes.py
MARGIN_PX = 4  # labeled boxes are widened by this much per side, in photo px, before scaling
BAND_JITTER_PX = 16  # a band window's center row is the drawn row +- this, in scaled px (then clamped)


class Stratum(Enum):
    """Where in the photo a negative window may sit. The value is k in the RNG key."""

    UNIFORM = 0  # anywhere in the scaled photo
    BAND = 1  # at the height of a labeled sign: where signs actually are, not sky or road surface


@dataclass(frozen=True)
class RealNegatives:
    arrays: dict[str, np.ndarray]  # images, photo_index, stratum, scale, origin, heatmap_size
    skipped: dict[Stratum, int]  # windows with no clear spot after PLACEMENT_TRIES, per stratum


def plan_negative(photo_size: tuple[int, int], boxes: Sequence[Box], scale: float, stratum: Stratum,
                  band_row: float | None, rng: np.random.Generator, cfg: SynthConfig) -> WindowPlan | Skip:
    """Choose a window of the photo scaled by `scale` that touches none of the labeled `boxes` (photo px),
    each widened by MARGIN_PX per side (clipped to the photo) and then scaled outward.

    UNIFORM: x0 and y0 uniform over the scaled photo. BAND: band_row is a center row in scaled px; the
    window's center row is band_row + uniform(-BAND_JITTER_PX, +BAND_JITTER_PX), rounded to whole pixels
    and clamped so the window stays inside; x0 is uniform. Each try draws x0, then the row.
    Returns the first clear window, or Skip.NO_CLEAR_SPOT after PLACEMENT_TRIES tries. The plan's box is
    (0, 0, 0, 0): there is no sign.
    Raises ValueError if scale is not in (0, 1], band_row is given for UNIFORM or missing for BAND, a box
    is empty, or the scaled photo is smaller than the window.
    """
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    if (band_row is None) != (stratum is Stratum.UNIFORM):
        raise ValueError(f"band_row must be given for BAND and only for BAND, got {stratum} with {band_row}")
    for b in boxes:
        if not (b[0] < b[2] and b[1] < b[3]):
            raise ValueError(f"empty box {b}")
    width, height = scaled_size(photo_size, scale)
    s = cfg.scene_size
    if width < s or height < s:
        raise ValueError(f"photo {photo_size} at scale {scale:.3f} is smaller than the {s}px window")

    avoid = [_scale_box(widen(b, photo_size), scale, (width, height)) for b in boxes]
    for _ in range(PLACEMENT_TRIES):
        x0 = int(rng.integers(0, width - s + 1))
        if band_row is None:
            y0 = int(rng.integers(0, height - s + 1))
        else:
            center = band_row + rng.uniform(-BAND_JITTER_PX, BAND_JITTER_PX)
            y0 = int(np.clip(round(center - s / 2), 0, height - s))
        window = (x0, y0, x0 + s, y0 + s)
        if not any(_overlap(window, a) for a in avoid):
            return WindowPlan(scale, x0, y0, (0, 0, 0, 0))
    return Skip.NO_CLEAR_SPOT


def widen(box: Box, photo_size: tuple[int, int]) -> Box:
    """The box grown by MARGIN_PX on every side, clipped to the photo."""
    x1, y1, x2, y2 = box
    w, h = photo_size
    return max(0, x1 - MARGIN_PX), max(0, y1 - MARGIN_PX), min(w, x2 + MARGIN_PX), min(h, y2 + MARGIN_PX)


def build_real_negatives(photos: Sequence[RealPhoto], positive_scales: np.ndarray, sign_rows: np.ndarray,
                         seed: int, cfg: SynthConfig) -> RealNegatives:
    """One window per Stratum on every photo, with or without signs; a window with no clear spot is skipped.

    positive_scales: the split's positive-window scales, min(1, sign_max / native_long_side), one per
    real_<split>.npz row. sign_rows: the center row (photo px) of every labeled sign in the split.
    Window k (= stratum.value) of photo p uses its own RNG [seed, NEG_STREAM, p.index, k], which draws,
    in order: the scale (uniform over positive_scales), for BAND a sign row (uniform over sign_rows,
    multiplied by that scale), then plan_negative's tries. So output depends only on the inputs, not on
    processing order. Windows are stored photo by photo, UNIFORM before BAND.
    Raises ValueError if positive_scales or sign_rows is empty, or no window is placed; OSError /
    ValueError from load_photo (nothing is returned on a partial failure).
    """
    if not len(positive_scales) or not len(sign_rows):
        raise ValueError("positive_scales and sign_rows must be non-empty")
    images: list[np.ndarray] = []
    meta: list[tuple[int, int, float, int, int]] = []  # photo_index, stratum, scale, x0, y0
    skipped = {stratum: 0 for stratum in Stratum}
    for photo in photos:
        img = load_photo(photo)
        boxes = [sign.box for sign in photo.signs]
        for stratum in Stratum:
            rng = np.random.default_rng([seed, NEG_STREAM, photo.index, stratum.value])
            scale = float(positive_scales[rng.integers(len(positive_scales))])
            band_row = None if stratum is Stratum.UNIFORM else float(sign_rows[rng.integers(len(sign_rows))]) * scale
            plan = plan_negative(img.size, boxes, scale, stratum, band_row, rng, cfg)
            if isinstance(plan, Skip):
                skipped[stratum] += 1
                continue
            images.append(cut_window(img, plan, cfg).image)
            meta.append((photo.index, stratum.value, plan.scale, plan.x0, plan.y0))
    if not images:
        raise ValueError(f"no negative window placed on {len(photos)} photos")
    arrays = {
        "images": np.stack(images),
        "photo_index": np.asarray([m[0] for m in meta], np.int16),
        "stratum": np.asarray([m[1] for m in meta], np.int8),
        "scale": np.asarray([m[2] for m in meta], np.float64),
        "origin": np.asarray([(m[3], m[4]) for m in meta], np.int16),  # x0, y0 in scaled-photo px
        "heatmap_size": np.asarray(cfg.heatmap_size, np.int16),
    }
    return RealNegatives(arrays, skipped)
