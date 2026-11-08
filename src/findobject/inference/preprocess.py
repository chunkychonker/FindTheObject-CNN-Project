"""Turns an arbitrary photo into a model-sized scene tensor."""

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from findobject.models.baseline import STRIDE


def to_scene(img: Image.Image, size: int) -> Tensor:
    """Center-crop to a square, resize to size x size -> float [3, size, size] in [0, 1].

    Uses bilinear resampling, as data/synth.py does for training scenes. An image that is already
    size x size is not resampled, so its pixels pass through exactly.

    Raises ValueError unless size > 0 and size is a multiple of STRIDE (heatmap cells must line up
    with scene pixels).
    """
    if size <= 0 or size % STRIDE:
        raise ValueError(f"size must be a positive multiple of {STRIDE}, got {size}")
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
