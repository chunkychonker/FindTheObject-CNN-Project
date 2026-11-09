"""Build the paired real-window and hybrid scene arrays for a set of GTSDB photos (docs/v2-spec.md §2.2-2.4)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from findobject.data.gtsdb import RealPhoto, load_photo, load_sign_crop
from findobject.data.real_windows import Skip, cut_window, plan_window
from findobject.data.synth import Scene, SynthConfig, make_scene

# Independent RNG streams, so changing how hybrid scenes are drawn never moves a real window.
WINDOW_STREAM = 10
HYBRID_STREAM = 11


@dataclass(frozen=True)
class SkippedSign:
    photo_index: int
    sign_index: int
    reason: Skip


@dataclass(frozen=True)
class RealScenes:
    real: dict[str, np.ndarray]  # SignScenes format plus photo_index, sign_index, native_long_side
    hybrid: dict[str, np.ndarray]  # same keys; row i is the same physical sign as real row i
    n_signs: int
    skipped: tuple[SkippedSign, ...]


def build_real_scenes(photos: Sequence[RealPhoto], backgrounds: Sequence[Path], seed: int,
                      cfg: SynthConfig) -> RealScenes:
    """One real window and one hybrid scene per labeled sign that gets a window; the others are skipped.

    Sign j of photo p uses its own RNGs, seeded [seed, stream, p.index, j], so output depends only on
    (photos, backgrounds, seed, cfg), not on processing order. cfg should be the unaugmented val/test
    config (data_config.synth_config). Hybrid scenes paste the GTSDB crop with make_scene onto a random
    background from `backgrounds`.
    Raises ValueError if no sign gets a window or backgrounds is empty; OSError / ValueError from
    load_photo for an unreadable or wrong-size photo (nothing is returned on a partial failure).
    """
    if not backgrounds:
        raise ValueError("no backgrounds for hybrid scenes")
    real: list[Scene] = []
    hybrid: list[Scene] = []
    meta: list[tuple[int, int, int, int]] = []  # photo_index, sign_index, native_long_side, class_id
    skipped: list[SkippedSign] = []
    n_signs = 0
    for photo in photos:
        if not photo.signs:
            continue
        img = load_photo(photo)
        boxes = [s.box for s in photo.signs]
        for j, sign in enumerate(photo.signs):
            n_signs += 1
            plan = plan_window(img.size, boxes, j, np.random.default_rng([seed, WINDOW_STREAM, photo.index, j]), cfg)
            if isinstance(plan, Skip):
                skipped.append(SkippedSign(photo.index, j, plan))
                continue
            real.append(cut_window(img, plan, cfg))
            rng = np.random.default_rng([seed, HYBRID_STREAM, photo.index, j])
            with Image.open(backgrounds[rng.integers(len(backgrounds))]) as bg:
                hybrid.append(make_scene(load_sign_crop(img, sign), bg, rng, cfg))
            x1, y1, x2, y2 = sign.box
            meta.append((photo.index, j, max(x2 - x1, y2 - y1), sign.class_id))
    if not real:
        raise ValueError(f"no sign got a window ({n_signs} signs, {len(skipped)} skipped)")
    return RealScenes(_arrays(real, meta), _arrays(hybrid, meta), n_signs, tuple(skipped))


def _arrays(scenes: list[Scene], meta: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    m = np.asarray(meta, np.int64)
    return {
        "images": np.stack([s.image for s in scenes]),
        "class_ids": m[:, 3].copy(),
        "boxes": np.asarray([s.box for s in scenes], np.int16),
        "heatmaps": np.stack([s.heatmap for s in scenes]).astype(np.float16),
        "photo_index": m[:, 0].astype(np.int16),
        "sign_index": m[:, 1].astype(np.int16),
        "native_long_side": m[:, 2].astype(np.int16),
    }
