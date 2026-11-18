"""Find every sign in one whole photo: run the model on an image pyramid, list its peaks, suppress repeats (docs/v3b-spec.md §2.1-2.2)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from findobject.inference.detection import PYRAMID_SCALES, Detection, detection_order
from findobject.inference.peaks import cell_to_photo, find_peaks
from findobject.inference.suppress import suppress
from findobject.models.baseline import STRIDE, check_images
from findobject.models.peak import PeakCNN, window_features_at


@dataclass(frozen=True)
class DetectResult:
    raw: tuple[Detection, ...]  # every peak at every scale, in detection_order (for located before suppression)
    kept: tuple[Detection, ...]  # after suppression, in detection_order: the model's answer


def _scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    """(w, h) * scale, which must come out as integers divisible by STRIDE. Raises ValueError otherwise."""
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    out = []
    for n in size:
        m = n * scale
        if abs(m - round(m)) > 1e-9 or round(m) % STRIDE or round(m) == 0:
            raise ValueError(f"{size} at scale {scale} is not a whole size divisible by {STRIDE}")
        out.append(round(m))
    return out[0], out[1]


def _to_tensor(img: Image.Image) -> Tensor:
    """RGB PIL image -> float [1, 3, H, W] in [0, 1], the conversion every scene so far went through."""
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)[None]


@torch.no_grad()
def _detections_at(model: PeakCNN, img: Image.Image, scale: float, device: torch.device) -> list[Detection]:
    x = _to_tensor(img).to(device)
    check_images(x)
    features = model.trunk(x)
    logits = model.heatmap_head(features)[0, 0]
    cells = find_peaks(logits.float().cpu())
    if len(cells) == 0:
        return []
    scores = torch.sigmoid(logits[cells[:, 0].to(device), cells[:, 1].to(device)]).float().cpu()
    class_probs = model.class_head(window_features_at(features, cells.to(device))).softmax(dim=1).cpu()
    class_prob, class_id = class_probs.max(dim=1)
    xy = cell_to_photo(cells, scale)
    return [Detection(x=float(px), y=float(py), score=float(s), class_id=int(c), class_prob=float(p), scale=scale)
            for (px, py), s, c, p in zip(xy.tolist(), scores.tolist(), class_id.tolist(), class_prob.tolist())]


def detect(model: PeakCNN, photo: Image.Image, scales: Sequence[float] = PYRAMID_SCALES) -> DetectResult:
    """All detections in one photo. Each scale resizes the photo with PIL bilinear (none at scale 1), runs the
    trunk and heatmap head once on the whole of it, and reads a class at every peak. Then suppression runs
    across all scales together.

    Raises RuntimeError if model.training (BatchNorm would use batch statistics); ValueError if a scale is
    not in (0, 1], a scaled size is not a whole number divisible by STRIDE, or as check_images does.
    """
    if model.training:
        raise RuntimeError("detect needs the model in eval mode (call model.eval())")
    device = next(model.parameters()).device
    photo = photo.convert("RGB")
    raw: list[Detection] = []
    for scale in scales:
        size = _scaled_size(photo.size, scale)
        img = photo if size == photo.size else photo.resize(size, Image.BILINEAR)
        raw.extend(_detections_at(model, img, scale, device))
    return DetectResult(raw=tuple(sorted(raw, key=detection_order)), kept=tuple(suppress(raw)))
