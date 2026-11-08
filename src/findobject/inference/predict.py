"""Converts model outputs into per-image readable predictions."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from findobject.eval.heatmap_metrics import peak_centers


class Prediction(NamedTuple):
    class_id: int
    class_prob: float  # softmax probability of class_id
    peak_xy: tuple[float, float]  # scene px of the argmax cell's center, as heatmap_metrics.peak_centers
    peak_prob: float  # sigmoid(heatmap) at the peak: how sure the model is that a sign center is there
    heatmap: Tensor  # float [h, w] in [0, 1], on CPU


@torch.no_grad()
def predict(model: nn.Module, images: Tensor) -> list[Prediction]:
    """Run `model` (expected in eval mode) on float [B, 3, H, W] images in [0, 1], one Prediction each.

    Uses the same argmax rules as evaluation (class: argmax of logits; location: first max cell in
    row-major order), so a scene scored here matches its evaluate.py outcome.

    Raises ValueError if H != W (peak locations assume square scenes), or as
    models.baseline.check_images does for a malformed batch.
    """
    if images.dim() == 4 and images.shape[2] != images.shape[3]:
        raise ValueError(f"scenes must be square, got {tuple(images.shape[2:])}")
    out = model(images)
    class_probs = out.class_logits.softmax(dim=1).cpu()
    heatmaps = out.heatmap_logits.sigmoid().squeeze(1).cpu()  # [B, h, w]
    peaks = peak_centers(out.heatmap_logits, images.shape[-1]).cpu()
    preds = []
    for probs, hm, (x, y) in zip(class_probs, heatmaps, peaks.tolist()):
        class_id = int(probs.argmax())
        preds.append(Prediction(
            class_id=class_id,
            class_prob=float(probs[class_id]),
            peak_xy=(x, y),
            peak_prob=float(hm.max()),
            heatmap=hm,
        ))
    return preds
