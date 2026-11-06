"""One optimization pass over a DataLoader."""

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from findobject.eval.heatmap_metrics import peak_in_box
from findobject.training.losses import LossParts, detection_loss


class EpochStats(NamedTuple):
    """Averages over the epoch's samples, measured while the weights were still changing."""

    mean_total: float
    class_acc: float
    pointing_acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    heatmap_weight: float,
    on_step: Callable[[LossParts], None],
) -> EpochStats:
    """Puts the model in train mode; steps optimizer and scheduler once per batch; calls on_step
    after each step.

    Raises FloatingPointError on the first non-finite loss, before that batch's update is applied
    (earlier batches' updates are kept). Raises ValueError if the loader is empty.
    """
    model.train()
    n = correct = hits = 0
    total = 0.0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        out = model(batch.image)
        parts = detection_loss(out, batch, heatmap_weight)
        if not torch.isfinite(parts.total):
            raise FloatingPointError(f"non-finite loss {parts.total.item()} at batch {batch_idx}")
        optimizer.zero_grad(set_to_none=True)
        parts.total.backward()
        optimizer.step()
        scheduler.step()
        on_step(parts)

        b = len(batch.class_id)
        n += b
        total += parts.total.item() * b
        correct += (out.class_logits.argmax(dim=1) == batch.class_id).sum().item()
        hits += peak_in_box(out.heatmap_logits.detach(), batch.box, batch.image.shape[-1]).sum().item()
    if n == 0:
        raise ValueError("empty loader")
    return EpochStats(mean_total=total / n, class_acc=correct / n, pointing_acc=hits / n)
