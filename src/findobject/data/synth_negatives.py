"""Build the synthetic scenes that hold no sign: plain empties and sign-shaped decoys (docs/v3a-spec.md §2.4).

Each scene i of a file has its own RNG [EMPTY_SEED, offset, i], so scene i never depends on the others.
Reads only the background images it is given; sign images are never opened (decoys use only ROI sizes).
"""

from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from findobject.data.gtsrb import SignSample
from findobject.data.synth import SynthConfig, make_empty_scene, make_scene

EMPTY_SEED = 3  # first word of every synthetic no-sign scene's RNG key; arbitrary, fixed once (v3a)


class SeedOffset(Enum):
    """Second word of the RNG key, one per kind of file."""

    VAL_EMPTY = 0
    TEST_EMPTY = 1
    # Both decoy files share it: decoy i of val_decoy_plain and of val_decoy_aug get the same background,
    # patch source and patch crop, so the two files differ only in the paste path.
    VAL_DECOY = 2


def build_empties(backgrounds: Sequence[Path], n: int, offset: SeedOffset, cfg: SynthConfig) -> dict[str, np.ndarray]:
    """n make_empty_scene scenes. Scene i draws its background index first, then make_empty_scene's draws.

    Returns images uint8 [n, S, S, 3], background_index int32 [n] (into `backgrounds`), background_name
    (each background's file name, so a check can confirm its split without the list), heatmap_size.
    Raises ValueError if n < 1 or backgrounds is empty; OSError for an unreadable background.
    """
    if n < 1 or not backgrounds:
        raise ValueError(f"need n >= 1 and some backgrounds, got n={n}, {len(backgrounds)} backgrounds")
    s = cfg.scene_size
    images = np.empty((n, s, s, 3), np.uint8)
    bg_index = np.empty(n, np.int32)
    for i in range(n):
        rng = np.random.default_rng([EMPTY_SEED, offset.value, i])
        bg_index[i] = rng.integers(len(backgrounds))
        with Image.open(backgrounds[bg_index[i]]) as bg:
            images[i] = make_empty_scene(bg, rng, cfg).image
    return {"images": images, "background_index": bg_index, "background_name": _names(backgrounds, bg_index),
            "heatmap_size": np.asarray(cfg.heatmap_size, np.int16)}


def build_decoys(signs: Sequence[SignSample], backgrounds: Sequence[Path], n: int, offset: SeedOffset,
                 cfg: SynthConfig) -> dict[str, np.ndarray]:
    """n decoys: make_scene called with a patch of another background in place of a sign, so the patch
    goes through exactly the paste path cfg gives a sign (resize to a 16-48 px long side, brightness, and
    cfg.sign_augment). The sign's heatmap and box are then dropped: there is no sign.

    Scene i draws, in order: the scene's background index; a sign index (its ROI size becomes the patch
    size; its pixels are unused); the patch's background index, never the scene's; the patch's top-left
    corner, uniform; then make_scene's draws.
    Returns images, background_index, patch_background_index (int32, into `backgrounds`), their
    background_name and patch_background_name, sign_index (int32, into `signs`), patch_size (int16 [n, 2],
    width and height), heatmap_size.
    Raises ValueError if n < 1, there are fewer than 2 backgrounds or no signs, or a patch background is
    smaller than its patch; OSError for an unreadable background.
    """
    if n < 1 or len(backgrounds) < 2 or not signs:
        raise ValueError(f"need n >= 1, >= 2 backgrounds and some signs, got n={n}, "
                         f"{len(backgrounds)} backgrounds, {len(signs)} signs")
    s = cfg.scene_size
    images = np.empty((n, s, s, 3), np.uint8)
    bg_index, patch_bg_index, sign_index = (np.empty(n, np.int32) for _ in range(3))
    patch_size = np.empty((n, 2), np.int16)
    for i in range(n):
        rng = np.random.default_rng([EMPTY_SEED, offset.value, i])
        bg_index[i] = rng.integers(len(backgrounds))
        sign_index[i] = rng.integers(len(signs))
        w, h = roi_size(signs[sign_index[i]])
        other = int(rng.integers(len(backgrounds) - 1))
        patch_bg_index[i] = other + (other >= bg_index[i])  # skip the scene's own background
        with Image.open(backgrounds[patch_bg_index[i]]) as src:
            if src.width < w or src.height < h:
                raise ValueError(f"{backgrounds[patch_bg_index[i]]}: {src.size} is smaller than a {w}x{h} patch")
            x, y = int(rng.integers(0, src.width - w + 1)), int(rng.integers(0, src.height - h + 1))
            patch = src.convert("RGB").crop((x, y, x + w, y + h))
        with Image.open(backgrounds[bg_index[i]]) as bg:
            images[i] = make_scene(patch, bg, rng, cfg).image
        patch_size[i] = (w, h)
    return {"images": images, "background_index": bg_index, "patch_background_index": patch_bg_index,
            "background_name": _names(backgrounds, bg_index),
            "patch_background_name": _names(backgrounds, patch_bg_index), "sign_index": sign_index, "patch_size": patch_size,
            "heatmap_size": np.asarray(cfg.heatmap_size, np.int16)}


def roi_size(sign: SignSample) -> tuple[int, int]:
    """(width, height) of the sign's ROI: the size load_sign crops to (GTSRB corners are inclusive)."""
    x1, y1, x2, y2 = sign.roi
    return x2 - x1 + 1, y2 - y1 + 1


def _names(backgrounds: Sequence[Path], index: np.ndarray) -> np.ndarray:
    return np.asarray([backgrounds[i].name for i in index])
