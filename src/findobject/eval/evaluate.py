"""Run a model over a DataLoader and aggregate classification + localization metrics."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from findobject.eval.heatmap_metrics import center_error_px, peak_in_box
from findobject.training.losses import detection_loss


class EvalResult(NamedTuple):
    class_loss: float
    heatmap_loss: float
    heatmap_excess: float  # heatmap_loss minus its floor; 0 = perfect
    class_acc: float  # top-1 class correct
    pointing_acc: float  # heatmap peak inside the true box
    joint_acc: float  # both at once: found it and named it
    center_err_mean: float  # px from peak cell center to true box center
    center_err_median: float
    y_true: Tensor  # int64 [N], cpu
    y_pred: Tensor  # int64 [N], cpu

    def scalars(self) -> dict[str, float]:
        """The float fields only (drops y_true / y_pred)."""
        return {k: v for k, v in self._asdict().items() if isinstance(v, float)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, heatmap_weight: float) -> EvalResult:
    """Puts the model in eval mode (BatchNorm uses running stats) and leaves it there.

    Raises ValueError if the loader is empty.
    """
    model.eval()
    n = 0
    class_loss = heatmap_loss = floor = 0.0
    y_true, y_pred, hit, err = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.image)
        parts = detection_loss(out, batch, heatmap_weight)
        b = len(batch.class_id)
        n += b
        class_loss += parts.class_loss.item() * b
        heatmap_loss += parts.heatmap_loss.item() * b
        floor += parts.heatmap_floor.item() * b
        scene = batch.image.shape[-1]
        y_true.append(batch.class_id.cpu())
        y_pred.append(out.class_logits.argmax(dim=1).cpu())
        hit.append(peak_in_box(out.heatmap_logits, batch.box, scene).cpu())
        err.append(center_error_px(out.heatmap_logits, batch.box, scene).cpu())
    if n == 0:
        raise ValueError("empty loader")
    y_true_t, y_pred_t, hit_t, err_t = torch.cat(y_true), torch.cat(y_pred), torch.cat(hit), torch.cat(err)
    correct = y_true_t == y_pred_t
    return EvalResult(
        class_loss=class_loss / n,
        heatmap_loss=heatmap_loss / n,
        heatmap_excess=(heatmap_loss - floor) / n,
        class_acc=correct.float().mean().item(),
        pointing_acc=hit_t.float().mean().item(),
        joint_acc=(correct & hit_t).float().mean().item(),
        center_err_mean=err_t.mean().item(),
        center_err_median=err_t.median().item(),
        y_true=y_true_t,
        y_pred=y_pred_t,
    )
