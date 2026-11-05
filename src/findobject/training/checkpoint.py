"""Atomic save of training state, and model reload from a saved checkpoint."""

import os
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from findobject.models.baseline import BaselineCNN


class CheckpointInfo(NamedTuple):
    epoch: int
    config: dict  # the full YAML config the run was trained with
    metrics: dict[str, float]  # val metrics recorded when this checkpoint was saved


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    config: dict,
    metrics: dict[str, float],
) -> None:
    """Write to <path>.tmp, then rename over path, so path is always either the previous
    checkpoint or the new one, never partial. Creates parent dirs. Raises OSError on write
    failure (a stray .tmp may remain; path is untouched)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "torch_version": str(torch.__version__),  # plain str: TorchVersion is rejected by weights_only loads
        },
        tmp,
    )
    os.replace(tmp, path)


def load_model(path: Path, device: torch.device) -> BaselineCNN:
    """Rebuild a BaselineCNN from a checkpoint, in eval mode on `device`.

    Raises FileNotFoundError if missing; RuntimeError if the weights don't fit the architecture.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BaselineCNN()
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_checkpoint_info(path: Path) -> CheckpointInfo:
    """Read what a checkpoint recorded about its training run, without building a model.

    Raises FileNotFoundError if missing; KeyError if the file lacks the fields save_checkpoint writes.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return CheckpointInfo(epoch=ckpt["epoch"], config=ckpt["config"], metrics=ckpt["metrics"])
