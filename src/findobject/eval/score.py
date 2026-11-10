"""Run a model over stored scenes and keep each scene's presence, predicted class and peak-in-box outcome."""

from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from findobject.eval.heatmap_metrics import peak_in_box
from findobject.eval.presence import presence_scores


class SceneScores(NamedTuple):
    """Row i = scene i of the loader's dataset (the loader must not shuffle)."""

    presence: np.ndarray  # float32 [N]
    y_true: np.ndarray  # int64 [N]; IGNORE_CLASS for a scene with no sign
    y_pred: np.ndarray  # int64 [N]: argmax class
    hit: np.ndarray  # bool [N]: heatmap peak inside the true box (meaningless for a scene with no sign)


@torch.no_grad()
def score_scenes(model: nn.Module, loader: DataLoader, device: torch.device) -> SceneScores:
    """Puts the model in eval mode and leaves it there. Raises ValueError if the loader is empty."""
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in SceneScores._fields}
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.image)
        parts["presence"].append(presence_scores(out.heatmap_logits).cpu())
        parts["y_true"].append(batch.class_id.cpu())
        parts["y_pred"].append(out.class_logits.argmax(dim=1).cpu())
        parts["hit"].append(peak_in_box(out.heatmap_logits, batch.box, batch.image.shape[-1]).cpu())
    if not parts["presence"]:
        raise ValueError("empty loader")
    return SceneScores(**{k: torch.cat(v).numpy() for k, v in parts.items()})
