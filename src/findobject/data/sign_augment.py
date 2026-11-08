"""Warps and recolors one sign crop so the same physical sign looks different each time it is used.

Pure: no file I/O. Train-only by convention: SynthConfig's default SignAugment() turns all of it off,
so val/test generation is unaffected.

Deliberately absent: horizontal flips (they turn "turn left" into "turn right", "keep left" into
"keep right", and so on) and hue shifts (color carries class meaning: red warns or forbids, blue commands).
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class SignAugment:
    """Knobs for per-sign appearance changes. The default changes nothing.

    Raises ValueError on construction if a value is out of range.
    """

    max_rotation_deg: float = 0.0  # uniform in [-max, +max], about the sign's center
    max_perspective: float = 0.0  # each corner shifts by up to this fraction of the sign's longer side
    contrast: tuple[float, float] = (1.0, 1.0)  # multiplier range
    saturation: tuple[float, float] = (1.0, 1.0)  # multiplier range
    occlusion_prob: float = 0.0  # chance a background patch is pasted over part of the sign
    occlusion_area: tuple[float, float] = (0.10, 0.25)  # occluder area as a fraction of the sign's box

    def __post_init__(self) -> None:
        if not 0 <= self.max_rotation_deg <= 45:
            raise ValueError("max_rotation_deg must be in [0, 45]")
        if not 0 <= self.max_perspective <= 0.25:
            raise ValueError("max_perspective must be in [0, 0.25]")
        for name in ("contrast", "saturation"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be (lo, hi) with 0 < lo <= hi")
        if not 0 <= self.occlusion_prob <= 1:
            raise ValueError("occlusion_prob must be in [0, 1]")
        lo, hi = self.occlusion_area
        if not 0 < lo <= hi < 1:
            raise ValueError("occlusion_area must be (lo, hi) with 0 < lo <= hi < 1")

    @property
    def is_identity(self) -> bool:
        return self == SignAugment()


def augment_sign(sign: Image.Image, rng: np.random.Generator, aug: SignAugment) -> tuple[Image.Image, Image.Image]:
    """Recolor, then rotate and perspective-warp an RGB sign crop.

    Returns (rgb, mask): mask is mode "L", same size as rgb, 255 where the sign fully covers a pixel,
    0 where it does not (soft at the warped edges). Paste with canvas.paste(rgb, xy, mask) so the
    background shows around a tilted sign instead of a box-shaped patch.

    An identity `aug` returns the input pixels unchanged and a full mask, without drawing from `rng`.
    Occlusion is not applied here (it needs the background); see synth.make_scene.
    """
    sign = sign.convert("RGB")
    if aug.is_identity:
        return sign.copy(), Image.new("L", sign.size, 255)

    sign = ImageEnhance.Contrast(sign).enhance(rng.uniform(*aug.contrast))
    sign = ImageEnhance.Color(sign).enhance(rng.uniform(*aug.saturation))

    w, h = sign.size
    src = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float64)
    dst = _warped_corners(src, rng, aug)
    dst -= dst.min(axis=0)  # shift so the warped sign starts at (0, 0)
    out_w, out_h = (int(math.ceil(v)) for v in dst.max(axis=0))
    coeffs = _perspective_coeffs(dst, src)  # PIL maps output -> input
    rgb = sign.transform((out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    mask = Image.new("L", (w, h), 255).transform((out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    return rgb, mask


def _warped_corners(corners: np.ndarray, rng: np.random.Generator, aug: SignAugment) -> np.ndarray:
    """Rotate the four corners about their center, then jitter each independently."""
    center = corners.mean(axis=0)
    theta = math.radians(rng.uniform(-aug.max_rotation_deg, aug.max_rotation_deg))
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    rotated = (corners - center) @ rot.T + center
    jitter = rng.uniform(-aug.max_perspective, aug.max_perspective, size=(4, 2)) * corners.max()
    return rotated + jitter


def _perspective_coeffs(from_pts: np.ndarray, to_pts: np.ndarray) -> tuple[float, ...]:
    """8 coefficients (a..h) with to = ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1))
    for each of the 4 point pairs, the form PIL's Image.PERSPECTIVE expects."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(from_pts, to_pts):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    return tuple(np.linalg.solve(np.array(rows), np.array(rhs)).tolist())
