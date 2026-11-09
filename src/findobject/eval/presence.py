"""Presence: how strongly a heatmap says "a sign is here", and the threshold τ on it (docs/v3a-spec.md §2.1, §2.5).

Pure, no I/O.
"""

from fractions import Fraction
import math

import numpy as np
import torch
from torch import Tensor


def presence_scores(heatmap_logits: Tensor) -> Tensor:
    """[B, 1, h, w] (or [B, h, w]) logits -> [B] float32: the largest cell probability, max sigmoid(logit).

    Computed as sigmoid(max logit), which is the same number: sigmoid never decreases, so the cell with
    the largest logit has the largest probability. Raises ValueError for any other shape or B = 0.
    """
    if heatmap_logits.dim() not in (3, 4) or (heatmap_logits.dim() == 4 and heatmap_logits.shape[1] != 1):
        raise ValueError(f"need [B, 1, h, w] or [B, h, w] logits, got {tuple(heatmap_logits.shape)}")
    if len(heatmap_logits) == 0:
        raise ValueError("no scenes")
    return torch.sigmoid(heatmap_logits.float().flatten(1).amax(dim=1))


def threshold_at_false_alarm(neg: np.ndarray, rate: float) -> np.float32:
    """τ such that at most k = floor(rate * n) of the n negative scores are >= τ, the smallest float32 that
    guarantees it: sort descending s_1 >= ... >= s_n, then τ = nextafter(s_{k+1}, +inf) in float32.

    Scores tied with s_{k+1} all fall below τ, so ties can only push the realised rate below `rate`.
    k uses rate as the decimal it was written as (0.29 * 100 is 28.999... in floats; k is still 29).
    Raises ValueError if neg is empty, not 1-D, or not all finite, or rate is not in [0, 1).
    """
    neg = np.asarray(neg, dtype=np.float32)
    if neg.ndim != 1 or len(neg) == 0 or not np.isfinite(neg).all():
        raise ValueError(f"need a non-empty 1-D array of finite scores, got shape {neg.shape}")
    if not 0 <= rate < 1:
        raise ValueError(f"rate must be in [0, 1), got {rate}")
    k = math.floor(Fraction(rate).limit_denominator(10**9) * len(neg))
    s_k1 = np.sort(neg)[::-1][k]  # s_{k+1}, 1-based
    return np.nextafter(s_k1, np.float32(np.inf), dtype=np.float32)
