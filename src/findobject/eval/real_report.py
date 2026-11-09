"""Group per-scene outcomes on real/hybrid windows into rates by size and subset (pure, no I/O)."""

from enum import Enum
from typing import Hashable, Sequence

import numpy as np

from findobject.eval.stats import Rate, wilson

SMALL_BELOW = 24  # px, sign's native longer side
LARGE_FROM = 48


class SizeBucket(Enum):
    SMALL = "small"  # L < 24
    MEDIUM = "medium"  # 24 <= L < 48
    LARGE = "large"  # L >= 48


def size_bucket(native_long_side: int) -> SizeBucket:
    """Bucket a sign by its longer side in the original photo. Raises ValueError if it is < 1."""
    if native_long_side < 1:
        raise ValueError(f"native_long_side must be >= 1, got {native_long_side}")
    if native_long_side < SMALL_BELOW:
        return SizeBucket.SMALL
    return SizeBucket.MEDIUM if native_long_side < LARGE_FROM else SizeBucket.LARGE


def rates_by(ok: np.ndarray, keys: Sequence[Hashable], min_n: int = 1) -> dict[Hashable, Rate]:
    """Success rate (with Wilson interval) of the bool array `ok` within each distinct key, keys in first-seen
    order. Groups with fewer than min_n scenes are left out.

    Raises ValueError if ok is not 1-D bool or len(keys) != len(ok).
    """
    ok = np.asarray(ok)
    if ok.ndim != 1 or ok.dtype != bool or len(keys) != len(ok):
        raise ValueError(f"need a 1-D bool array and as many keys, got {ok.dtype} {ok.shape}, {len(keys)} keys")
    groups: dict[Hashable, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    return {k: wilson(int(ok[idx].sum()), len(idx)) for k, idx in groups.items() if len(idx) >= min_n}


def discordant(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int]:
    """(only a right, only b right) for paired bool outcomes; the inputs to mcnemar_exact.

    Raises ValueError if the arrays are not 1-D bool of equal length.
    """
    a_ok, b_ok = np.asarray(a_ok), np.asarray(b_ok)
    if a_ok.ndim != 1 or a_ok.shape != b_ok.shape or a_ok.dtype != bool or b_ok.dtype != bool:
        raise ValueError(f"need equal-length 1-D bool arrays, got {a_ok.dtype} {a_ok.shape}, {b_ok.dtype} {b_ok.shape}")
    return int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
