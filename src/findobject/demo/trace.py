"""Run detection on any photo and keep each scale's heatmap, so every stage can be shown (docs/demo-spec.md §2)."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from findobject.demo.fit import FittedPhoto, fit_photo
from findobject.inference.detect import DetectResult, detect
from findobject.inference.detection import PYRAMID_SCALES, Detection
from findobject.models.peak import PeakCNN


@dataclass(frozen=True)
class ScaleView:
    scale: float
    heatmap: np.ndarray  # sigmoid probabilities, float32 [h, w], one value per heatmap cell


@dataclass(frozen=True)
class Trace:
    fitted: FittedPhoto
    views: tuple[ScaleView, ...]  # one per pyramid scale, finest first
    result: DetectResult  # detect() on fitted.image, minus detections whose point lies on the padding


def trace(model: PeakCNN, photo: Image.Image) -> Trace:
    """Fit the photo, run detect() on it unchanged, and run the trunk and heatmap head once more per scale to
    keep the heatmaps (the second pass costs about as much as detect itself; it leaves the graded code alone).

    Detections whose point lands on the black padding are dropped from raw and kept alike: they can't be a
    sign in the photo. A photo already FIT_MAX in size has no padding, so its result is exactly detect()'s.
    Raises as detect() does (RuntimeError if model.training).
    """
    fitted = fit_photo(photo)
    result = detect(model, fitted.image)
    on_photo = lambda dets: tuple(d for d in dets if fitted.in_content(d.x, d.y))  # noqa: E731
    views = tuple(ScaleView(s, _heatmap(model, fitted.image, s)) for s in PYRAMID_SCALES)
    return Trace(fitted=fitted, views=views, result=DetectResult(raw=on_photo(result.raw), kept=on_photo(result.kept)))


def above(dets: tuple[Detection, ...], tau: float) -> tuple[Detection, ...]:
    """The detections scoring at least tau, in their given order. Pure; never calls the model."""
    return tuple(d for d in dets if d.score >= tau)


@torch.no_grad()
def _heatmap(model: PeakCNN, img: Image.Image, scale: float) -> np.ndarray:
    """Sigmoid heatmap of img resized exactly as detect() resizes it (PIL bilinear; none at scale 1)."""
    w, h = img.size
    size = (round(w * scale), round(h * scale))
    small = img if size == img.size else img.resize(size, Image.BILINEAR)
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(small, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)[None]
    logits = model.heatmap_head(model.trunk(x.to(device)))[0, 0]
    return torch.sigmoid(logits).float().cpu().numpy()
