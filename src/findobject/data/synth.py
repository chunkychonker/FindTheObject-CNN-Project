"""Pure scene synthesis: paste one sign crop onto one background and build its targets.

No file I/O here. Callers pass in already-loaded PIL images and a seeded RNG, so the
same (inputs, rng state) always produce the same Scene.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from findobject.data.sign_augment import SignAugment, augment_sign


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
    sign_augment: SignAugment = SignAugment()  # train-only extras; the default is off (val/test unchanged)

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

    Augmentation: brightness on the sign; optional blur and Gaussian noise on the whole scene; and,
    only if cfg.sign_augment is not the identity, recolor/rotate/warp/occlusion of the sign
    (see sign_augment.py). With the identity default, output is exactly what it was before
    sign_augment existed. Does not mutate the input images. Raises whatever PIL raises on unreadable image data.
    """
    s = cfg.scene_size
    canvas = _canvas(bg, rng, cfg)

    sign = sign.convert("RGB")
    long_side = int(rng.integers(cfg.sign_min, cfg.sign_max + 1))
    scale = long_side / max(sign.size)
    w = max(1, round(sign.width * scale))
    h = max(1, round(sign.height * scale))
    sign = sign.resize((w, h), Image.BILINEAR)
    sign = ImageEnhance.Brightness(sign).enhance(rng.uniform(*cfg.brightness))

    if cfg.sign_augment.is_identity:
        # The original path, untouched: it must draw from rng in exactly the same order so that
        # regenerating v1/v1.5 data stays byte-identical.
        x1 = int(rng.integers(0, s - w + 1))
        y1 = int(rng.integers(0, s - h + 1))
        canvas.paste(sign, (x1, y1))
        box = (x1, y1, x1 + w, y1 + h)
    else:
        box = _paste_augmented(canvas, sign, rng, cfg.sign_augment)

    return Scene(image=_finish(canvas, rng, cfg), box=box, heatmap=box_heatmap(box, cfg))


def make_empty_scene(bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Scene:
    """A scene with no sign: the same crop, resize, blur and noise as make_scene, with no paste.

    Box (0, 0, 0, 0) and an all-zero heatmap. Its crop draws come first, as in make_scene, so with the
    same rng state it crops the same region of bg that make_scene would paste onto. Does not mutate bg.
    Raises whatever PIL raises on unreadable image data.
    """
    canvas = _canvas(bg, rng, cfg)
    heatmap = np.zeros((cfg.heatmap_size, cfg.heatmap_size), np.float32)
    return Scene(image=_finish(canvas, rng, cfg), box=(0, 0, 0, 0), heatmap=heatmap)


def _canvas(bg: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> Image.Image:
    """First half of every scene's background pipeline: a random square crop of bg, resized to the scene."""
    s = cfg.scene_size
    return _random_square_crop(bg.convert("RGB"), rng).resize((s, s), Image.BILINEAR)


def _finish(canvas: Image.Image, rng: np.random.Generator, cfg: SynthConfig) -> np.ndarray:
    """Second half, after any paste: optional blur, then Gaussian noise, as uint8 [S, S, 3]."""
    if rng.random() < cfg.blur_prob:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0, cfg.blur_max_radius)))
    image = np.asarray(canvas, dtype=np.float32)
    image = image + rng.normal(0.0, rng.uniform(0, cfg.noise_max_std), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def box_heatmap(box: tuple[int, int, int, int], cfg: SynthConfig) -> np.ndarray:
    """The target heatmap for a sign at `box` (scene px, x2/y2 exclusive): a Gaussian at the box center.

    Shared by synthetic scenes and real-photo windows so both use one formula. Never raises for a
    valid SynthConfig (the center may lie anywhere; far-off centers just give a near-zero map).
    """
    # Heatmap cell c covers scene pixels [c*stride, (c+1)*stride), centered at (c + 0.5)*stride.
    stride = cfg.scene_size / cfg.heatmap_size
    bx1, by1, bx2, by2 = box
    center = ((bx1 + bx2) / 2 / stride - 0.5, (by1 + by2) / 2 / stride - 0.5)
    return gaussian_heatmap(center, cfg.heatmap_size, cfg.heatmap_sigma)


def _paste_augmented(
    canvas: Image.Image, sign: Image.Image, rng: np.random.Generator, aug: SignAugment
) -> tuple[int, int, int, int]:
    """Warp `sign`, paste it through its mask at a random position, maybe occlude it with a patch
    of the same background. Mutates canvas. Returns the box: the tight bounds of every pixel the
    sign touches (mask > 0), x2/y2 exclusive. Occlusion does not shrink the box: the sign is still there.
    """
    s = canvas.width
    rgb, mask = augment_sign(sign, rng, aug)
    w, h = rgb.size
    if w > s or h > s:  # only possible with extreme settings; shrink back to fit
        fit = s / max(w, h)
        size = (max(1, int(w * fit)), max(1, int(h * fit)))
        rgb, mask = rgb.resize(size, Image.BILINEAR), mask.resize(size, Image.BILINEAR)
        w, h = size
    x1 = int(rng.integers(0, s - w + 1))
    y1 = int(rng.integers(0, s - h + 1))
    background = canvas.copy()
    canvas.paste(rgb, (x1, y1), mask)
    mx1, my1, mx2, my2 = mask.getbbox()  # bounds of mask > 0, right/bottom exclusive
    box = (x1 + mx1, y1 + my1, x1 + mx2, y1 + my2)
    if rng.random() < aug.occlusion_prob:
        _occlude(canvas, background, box, rng, aug.occlusion_area)
    return box


def _occlude(
    canvas: Image.Image, background: Image.Image, box: tuple[int, int, int, int],
    rng: np.random.Generator, area_range: tuple[float, float],
) -> None:
    """Paste a rectangle cut from elsewhere in `background` over part of `box`, like a branch or a
    car in front of the sign. Area is a fraction of the box drawn from area_range; aspect 1:2 to 2:1."""
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    area = rng.uniform(*area_range) * bw * bh
    aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
    ow = int(np.clip(round(math.sqrt(area * aspect)), 1, bw))
    oh = int(np.clip(round(math.sqrt(area / aspect)), 1, bh))
    ox = bx1 + int(rng.integers(0, bw - ow + 1))
    oy = by1 + int(rng.integers(0, bh - oh + 1))
    sx = int(rng.integers(0, canvas.width - ow + 1))
    sy = int(rng.integers(0, canvas.height - oh + 1))
    canvas.paste(background.crop((sx, sy, sx + ow, sy + oh)), (ox, oy))


def _random_square_crop(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    side = min(img.size)
    x = int(rng.integers(0, img.width - side + 1))
    y = int(rng.integers(0, img.height - side + 1))
    return img.crop((x, y, x + side, y + side))
