"""Peak CNN: BaselineCNN's trunk and heatmap head, with the class read where the heatmap peaks."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from findobject.data.gtsrb import NUM_CLASSES
from findobject.models.baseline import FEATURE_CHANNELS, ModelOutput, build_heatmap_head, build_trunk, check_images

PEAK_WINDOW = 3  # class features = mean over this many cells square, centered on the peak


class PeakCNN(nn.Module):
    """Same layers, parameter names and shapes as BaselineCNN (so the same seed gives the same
    initial weights); only the class head's input differs. Instead of averaging all cells, it
    averages the PEAK_WINDOW x PEAK_WINDOW cells around the heatmap's own argmax.

    The peak is chosen the same way in training and eval, and choosing it passes no gradient:
    the class loss trains the trunk features under the window, not the heatmap head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.trunk = build_trunk()
        self.heatmap_head = build_heatmap_head()
        self.class_head = nn.Linear(FEATURE_CHANNELS, num_classes)

    def forward(self, images: Tensor) -> ModelOutput:
        """images: float [B, 3, H, W] in [0, 1]. Raises ValueError as check_images does."""
        check_images(images)
        features = self.trunk(images)  # [B, 128, H/4, W/4]
        heatmap_logits = self.heatmap_head(features)
        return ModelOutput(
            heatmap_logits=heatmap_logits,
            class_logits=self.class_head(window_features(features, peak_cells(heatmap_logits))),
        )


def peak_cells(heatmap_logits: Tensor) -> Tensor:
    """[B, 1, h, w] -> int64 [B, 2] (row, col) of each argmax cell.

    Ties break like heatmap_metrics.peak_centers (first max in row-major order), so the cell
    classified here is the cell the pointing metric scores. Raises ValueError on other shapes.
    """
    if heatmap_logits.dim() != 4 or heatmap_logits.shape[1] != 1:
        raise ValueError(f"expected [B, 1, h, w], got {tuple(heatmap_logits.shape)}")
    b, _, _, w = heatmap_logits.shape
    idx = heatmap_logits.detach().reshape(b, -1).argmax(dim=1)
    return torch.stack([idx // w, idx % w], dim=1)


def window_features(features: Tensor, cells: Tensor) -> Tensor:
    """[B, C, h, w] features, int64 [B, 2] (row, col) cells -> [B, C]: the mean feature over the
    PEAK_WINDOW x PEAK_WINDOW cells centered on each sample's cell. Near a border the window is
    clipped and only in-bounds cells are averaged (a corner averages 2 x 2 = 4 cells).
    """
    windows = F.avg_pool2d(features, PEAK_WINDOW, stride=1, padding=PEAK_WINDOW // 2, count_include_pad=False)
    # Two index tensors separated by a slice: their shared batch dim goes first -> [B, C].
    return windows[torch.arange(len(cells), device=cells.device), :, cells[:, 0], cells[:, 1]]
