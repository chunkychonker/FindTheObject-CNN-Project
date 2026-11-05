"""Pure scene synthesis: paste one sign crop onto one background and build its targets.

No file I/O here. Callers pass in already-loaded PIL images and a seeded RNG, so the
same (inputs, rng state) always produce the same Scene.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class SynthConfig:
    """Knobs for scene generation. Raises ValueError on construction if inconsistent."""

    scene_size: int = 128
    heatmap_size: int = 32
    sign_min: int = 16  # longer side of the pasted sign, in scene pixels
    sign_max: int = 48
    brightness: tuple[float, float] = (0.6, 1.4)  # multiplier range applied to the sign
    blur_prob: float = 0.5
    blur_max_radius: float = 1.0  # Gaussian blur radius (px) on the whole scene
    noise_max_std: float = 8.0  # per-pixel Gaussian noise std (0-255 scale) on the whole scene
    heatmap_sigma: float = 1.0  # in heatmap cells

    def __post_init__(self) -> None:
        if self.scene_size % self.heatmap_size != 0:
            raise ValueError("scene_size must be a multiple of heatmap_size")
        if not 1 <= self.sign_min <= self.sign_max <= self.scene_size:
            raise ValueError("need 1 <= sign_min <= sign_max <= scene_size")
        lo, hi = self.brightness
        if not 0 < lo <= hi:
            raise ValueError("brightness must be (lo, hi) with 0 < lo <= hi")
        if not 0 <= self.blur_prob <= 1:
            raise ValueError("blur_prob must be in [0, 1]")
        if self.blur_max_radius < 0 or self.noise_max_std < 0 or self.heatmap_sigma <= 0:
            raise ValueError("blur_max_radius, noise_max_std must be >= 0; heatmap_sigma > 0")


@dataclass(frozen=True)
class Scene:
    image: np.ndarray  # uint8 [S, S, 3]
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in scene pixels; x2/y2 exclusive
    heatmap: np.ndarray  # float32 [H, H], peak at the sign center


def gaussian_heatmap(center_xy: tuple[float, float], size: int, sigma: float) -> np.ndarray:
    """Return a float32 [size, size] array with a 2D Gaussian bump at center_xy.

    center_xy is (x, y) in heatmap-cell coordinates, where cell (row r, col c) has its
    center at (x=c, y=r). It may be fractional and may lie between cells.
    Value at cell (r, c) = exp(-((c - x)^2 + (r - y)^2) / (2 * sigma^2)),
    so it is exactly 1.0 when the center lands on a cell, and always in [0, 1]
    (far cells may underflow to exactly 0 in float32).

    Raises ValueError if size < 1 or sigma <= 0.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    x, y = center_xy
    cells = np.arange(size, dtype=np.float32)
    dx2 = (cells - x) ** 2  # [size], one entry per column
    dy2 = (cells - y) ** 2  # [size], one entry per row
    # [size, 1] + [1, size] broadcasts to [size, size]: dist2[r, c] = dy2[r] + dx2[c]
    dist2 = dy2[:, None] + dx2[None, :]
    return np.exp(-dist2 / (2 * sigma**2)).astype(np.float32, copy=False)


def make_scene(sign: Image.Image, bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene:
    """Paste `sign` at a random scale/position onto a random square crop of `bg`.

    Augmentation: brightness on the sign; optional blur and Gaussian noise on the whole scene.
    Does not mutate the input images. Raises whatever PIL raises on unreadable image data.
    """
    s = cfg.scene_size
    canvas = _random_square_crop(bg.convert("RGB"), rng).resize((s, s), Image.BILINEAR)

    sign = sign.convert("RGB")
    long_side = int(rng.integers(cfg.sign_min, cfg.sign_max + 1))
    scale = long_side / max(sign.size)
    w = max(1, round(sign.width * scale))
    h = max(1, round(sign.height * scale))
    sign = sign.resize((w, h), Image.BILINEAR)
    sign = ImageEnhance.Brightness(sign).enhance(rng.uniform(*cfg.brightness))

    x1 = int(rng.integers(0, s - w + 1))
    y1 = int(rng.integers(0, s - h + 1))
    canvas.paste(sign, (x1, y1))

    if rng.random() < cfg.blur_prob:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0, cfg.blur_max_radius)))

    image = np.asarray(canvas, dtype=np.float32)
    image = image + rng.normal(0.0, rng.uniform(0, cfg.noise_max_std), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Heatmap cell c covers scene pixels [c*stride, (c+1)*stride), centered at (c + 0.5)*stride.
    stride = s / cfg.heatmap_size
    center = ((x1 + w / 2) / stride - 0.5, (y1 + h / 2) / stride - 0.5)
    heatmap = gaussian_heatmap(center, cfg.heatmap_size, cfg.heatmap_sigma)

    return Scene(image=image, box=(x1, y1, x1 + w, y1 + h), heatmap=heatmap)


def _random_square_crop(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    side = min(img.size)
    x = int(rng.integers(0, img.width - side + 1))
    y = int(rng.integers(0, img.height - side + 1))
    return img.crop((x, y, x + side, y + side))
