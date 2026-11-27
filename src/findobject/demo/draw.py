"""Render each detection stage as an image for the demo (docs/demo-spec.md §2). Pure: in-memory images only.

Every function draws in the coordinates of the photo it's given (the fitted photo's content), so points from
a Trace can be drawn directly.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from findobject.data.class_names import CLASS_NAMES
from findobject.demo.grade import AnswerGrade
from findobject.demo.trace import ScaleView
from findobject.eval.photo_truth import TruthSign
from findobject.inference.detection import PYRAMID_SCALES, Detection
from findobject.inference.suppress import SUPPRESS_CELLS
from findobject.models.baseline import STRIDE

SCALE_COLORS = {1.0: (0, 229, 255), 0.5: (255, 214, 0), 0.25: (255, 64, 255)}  # cyan, yellow, magenta
SCALE_LABELS = {1.0: "1", 0.5: "½", 0.25: "¼"}
FOUND_COLOR = (0, 230, 118)  # green: a labeled sign that was found
MISSED_COLOR = (255, 145, 0)  # orange: a labeled sign that wasn't
WRONG_COLOR = (255, 59, 48)  # red: a detection that found nothing
ANSWER_COLOR = (0, 229, 255)  # a detection when there's nothing to grade it against
HEATMAP_ALPHA = 0.75
RING_RADIUS = 10
GAP = 16
_cmap = colormaps["inferno"]


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _ring(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], width: int = 3) -> None:
    r = RING_RADIUS
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)


def _label(draw: ImageDraw.ImageDraw, image_size: tuple[int, int], x: float, y: float, text: str,
           size: int = 16) -> None:
    """White text on a dark box, nudged to stay inside the image."""
    font = _font(size)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 8, b - t + 6
    W, H = image_size
    x0 = min(max(0, x), W - w)
    y0 = min(max(0, y), H - h)
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(0, 0, 0))
    draw.text((x0 + 4 - l, y0 + 3 - t), text, fill=(255, 255, 255), font=font)


def draw_scales(photo: Image.Image) -> Image.Image:
    """The photo at every pyramid scale, side by side at true relative size, each with what the model sees."""
    sizes = [(max(1, round(photo.width * s)), max(1, round(photo.height * s))) for s in PYRAMID_SCALES]  # ≥1: thin uploads
    strip = 40
    canvas = Image.new("RGB", (sum(w for w, _ in sizes) + GAP * (len(sizes) - 1), photo.height + strip), (24, 24, 27))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for s, (w, h) in zip(PYRAMID_SCALES, sizes):
        canvas.paste(photo if s == 1 else photo.resize((w, h), Image.BILINEAR), (x, 0))
        draw.text((x + 2, photo.height + 8), f"×{SCALE_LABELS[s]}: {w}×{h} px", fill=(230, 230, 230), font=_font(20))
        x += w + GAP
    return canvas


def draw_heatmap(photo: Image.Image, view: ScaleView) -> Image.Image:
    """The heatmap at one scale stretched back over the photo, one solid block per cell. Each pixel blends
    toward the colormap by HEATMAP_ALPHA x the cell's value, so a cell at 0 leaves the photo as it was.

    The heatmap may cover more than the photo (the padding of a fitted photo); the extra is cropped off.
    Raises ValueError if it covers less, in either direction: then it wasn't made from this photo at this scale.
    """
    cell = STRIDE / view.scale  # photo px per heatmap cell
    full = (round(view.heatmap.shape[1] * cell), round(view.heatmap.shape[0] * cell))
    if full[0] < photo.width or full[1] < photo.height:
        raise ValueError(f"heatmap does not cover the photo: photo is {photo.width}x{photo.height} px, heatmap "
                         f"{view.heatmap.shape} cells at scale {view.scale} covers {full[0]}x{full[1]} px")
    rgb = (_cmap(view.heatmap)[..., :3] * 255).astype(np.uint8)
    colored = np.asarray(Image.fromarray(rgb).resize(full, Image.NEAREST), dtype=np.float32)
    value = np.asarray(Image.fromarray(view.heatmap.astype(np.float32)).resize(full, Image.NEAREST))
    colored, value = colored[:photo.height, :photo.width], value[:photo.height, :photo.width]
    w = HEATMAP_ALPHA * value[..., None]
    base = np.asarray(photo, dtype=np.float32)
    return Image.fromarray(np.clip((1 - w) * base + w * colored, 0, 255).round().astype(np.uint8))


def draw_points(photo: Image.Image, dets: Sequence[Detection], show_suppression: bool = False) -> Image.Image:
    """A ring at every detection, colored by the scale it came from. With show_suppression, also the square
    each one clears: detections strictly inside it (Chebyshev distance) were dropped as repeats."""
    out = photo.copy()
    draw = ImageDraw.Draw(out)
    for d in dets:
        color = SCALE_COLORS[d.scale]
        if show_suppression:
            r = SUPPRESS_CELLS * STRIDE / d.scale
            draw.rectangle((d.x - r, d.y - r, d.x + r, d.y + r), outline=color, width=1)
        _ring(draw, d.x, d.y, color)
    return out


def draw_answer(photo: Image.Image, dets: Sequence[Detection], truth: Sequence[TruthSign] | None = None,
                grade: AnswerGrade | None = None) -> Image.Image:
    """The final answer: a ring and "class score" label per detection. With truth and its grade, labeled
    signs are boxed green (found) or orange (missed), and detections that found nothing get a red ring."""
    out = photo.copy()
    draw = ImageDraw.Draw(out)
    for s in truth or ():
        color = MISSED_COLOR if grade is not None and s.gt_index in grade.missed else FOUND_COLOR
        draw.rectangle((s.box[0], s.box[1], s.box[2] - 1, s.box[3] - 1), outline=color, width=3)
    for i, d in enumerate(dets):
        wrong = grade is not None and grade.found[i] is None
        _ring(draw, d.x, d.y, WRONG_COLOR if wrong else ANSWER_COLOR)
        _label(draw, out.size, d.x + RING_RADIUS + 4, d.y - RING_RADIUS - 22, f"{CLASS_NAMES[d.class_id]} {d.score:.2f}")
    return out
