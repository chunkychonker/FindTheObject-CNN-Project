"""Find the local maxima of a heatmap, and map heatmap cells back to native photo pixels (docs/v3b-spec.md §2.1-2.2). Pure."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from findobject.models.baseline import STRIDE

MIN_PEAK_SCORE = 0.05  # peaks below this probability are not listed; every reported threshold is above it


def find_peaks(logits: Tensor, min_score: float = MIN_PEAK_SCORE) -> Tensor:
    """[h, w] logits -> int64 [N, 2] (row, col) of every cell whose logit is >= logit(min_score) and equal to
    the max logit of its 3x3 neighborhood (clipped at borders), in row-major order. N may be 0.

    Works on logits, not probabilities: float32 sigmoid maps nearby large logits to the same value, which
    would turn one peak into a tie. Every cell of a flat top is a peak; suppression merges them.
    Raises ValueError if logits is not 2-D, has a non-finite value, or min_score is not in (0, 1).
    """
    if logits.dim() != 2:
        raise ValueError(f"need [h, w] logits, got {tuple(logits.shape)}")
    if not 0.0 < min_score < 1.0:
        raise ValueError(f"min_score must be in (0, 1), got {min_score}")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must all be finite")
    floor = math.log(min_score / (1.0 - min_score))
    # max_pool2d pads with -inf, so a border cell is compared only with its in-bounds neighbors.
    neighborhood_max = F.max_pool2d(logits[None, None], 3, stride=1, padding=1)[0, 0]
    is_peak = (logits == neighborhood_max) & (logits >= floor)
    return torch.nonzero(is_peak)  # row-major


def cell_to_photo(cells: Tensor, scale: float) -> Tensor:
    """int64 [N, 2] (row, col) cells of the heatmap at pyramid `scale` -> float64 [N, 2] (x, y) in native
    photo px: the center of the cell, ((col + 0.5) * STRIDE / scale, (row + 0.5) * STRIDE / scale).
    Raises ValueError if scale <= 0 or cells is not [N, 2]."""
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if cells.dim() != 2 or cells.shape[1] != 2:
        raise ValueError(f"need [N, 2] cells, got {tuple(cells.shape)}")
    centers = (cells.double() + 0.5) * (STRIDE / scale)
    return centers.flip(1)  # (row, col) -> (x, y)
