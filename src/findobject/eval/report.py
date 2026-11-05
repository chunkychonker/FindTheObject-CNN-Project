"""Per-class accuracy and confusion counts from true/predicted class ids (pure, no I/O)."""

from typing import NamedTuple

import torch


class ClassReport(NamedTuple):
    counts: list[int]  # [C] scenes per true class
    accuracy: list[float | None]  # [C] fraction correct; None for classes absent from the split
    confusion: list[list[int]]  # [C][C] rows = true class, cols = predicted class


def class_report(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> ClassReport:
    """Count, per true class, how many scenes there are and how they were predicted.

    Plain lists in the result, so it drops straight into json.dump.
    Raises ValueError if y_true / y_pred are not 1-D of equal length, or hold ids outside [0, num_classes).
    """
    if y_true.dim() != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"need equal-length 1-D tensors, got {tuple(y_true.shape)} and {tuple(y_pred.shape)}")
    for name, ids in (("y_true", y_true), ("y_pred", y_pred)):
        if len(ids) and (ids.min() < 0 or ids.max() >= num_classes):
            raise ValueError(f"{name} has class ids outside [0, {num_classes})")

    # Each (true, pred) pair maps to one cell index true*C + pred; bincount tallies all cells at once.
    cells = torch.bincount(y_true.long() * num_classes + y_pred.long(), minlength=num_classes * num_classes)
    confusion = cells.reshape(num_classes, num_classes)
    counts = confusion.sum(dim=1)
    correct = confusion.diagonal()
    accuracy = [c / n if n else None for c, n in zip(correct.tolist(), counts.tolist())]
    return ClassReport(counts=counts.tolist(), accuracy=accuracy, confusion=confusion.tolist())
