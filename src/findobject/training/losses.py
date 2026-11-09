"""Combined loss: sign-class cross-entropy plus weighted per-cell heatmap BCE."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from findobject.data.dataset import IGNORE_CLASS, Sample
from findobject.models.baseline import ModelOutput


class LossParts(NamedTuple):
    total: Tensor  # class_loss + heatmap_weight * heatmap_loss; the only one to backprop
    class_loss: Tensor  # cross-entropy summed over sign scenes / batch size; ln(43) = 3.76 for a uniform guess on all-sign batches
    heatmap_loss: Tensor  # BCE summed over cells, mean over batch
    heatmap_floor: Tensor  # lowest heatmap_loss possible for these soft targets (no grad)


def detection_loss(out: ModelOutput, batch: Sample, heatmap_weight: float) -> LossParts:
    """Raises ValueError if the heatmap logits and targets differ in shape.

    Scenes with class_id == IGNORE_CLASS (no sign) add nothing to the class loss, but its sum is still
    divided by the whole batch size b: each sign scene keeps weight 1/b however many empties share the
    batch, and an all-empty batch gives class loss 0, not the NaN a mean over zero rows would.
    """
    b = len(batch.class_id)
    class_loss = F.cross_entropy(out.class_logits, batch.class_id, ignore_index=IGNORE_CLASS, reduction="sum") / b
    # Sum (not mean) over cells: only ~15 of 1024 cells carry signal, and a mean would dilute them.
    heatmap_loss = F.binary_cross_entropy_with_logits(out.heatmap_logits, batch.heatmap, reduction="sum") / b
    return LossParts(
        total=class_loss + heatmap_weight * heatmap_loss,
        class_loss=class_loss,
        heatmap_loss=heatmap_loss,
        heatmap_floor=_binary_entropy(batch.heatmap).sum() / b,
    )


def _binary_entropy(t: Tensor) -> Tensor:
    """-(t log t + (1-t) log(1-t)), with 0 log 0 = 0: BCE of a target against itself."""
    t = t.detach()
    return -(torch.special.xlogy(t, t) + torch.special.xlogy(1 - t, 1 - t))
