"""How much more a model believes a pasted sign than the same real sign, split by relative size. Pure.

Row i of every array is one real_dev sign and its hybrid twin (the same sign's pixels pasted onto COCO).
The twin also differs in size and finishing, not only in its paste edge, so the gap is reported per
relative-size group (docs/v3a1-spec.md §2.3). Intervals resample whole photos, with the same seed and
resample count as presence_report, so every interval of one report uses the same resamples.
"""

from enum import Enum

import numpy as np

from findobject.eval.presence_report import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, seam_check
from findobject.eval.stats import cluster_bootstrap

SIZE_MATCH_PX = 4  # long sides within this many pixels of each other count as the same size


class RelativeSize(Enum):
    HYBRID_LARGER = "hybrid_larger"  # hybrid long side > real + SIZE_MATCH_PX
    SIMILAR = "similar"
    HYBRID_SMALLER = "hybrid_smaller"  # hybrid long side < real − SIZE_MATCH_PX


def box_long_side(boxes: np.ndarray) -> np.ndarray:
    """Longer side of each (x1, y1, x2, y2) box, x2/y2 exclusive, as int64. Raises ValueError unless boxes is [n, 4]."""
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"need boxes of shape [n, 4], got {boxes.shape}")
    b = boxes.astype(np.int64)
    return np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])


def relative_size(real_long: np.ndarray, hybrid_long: np.ndarray) -> np.ndarray:
    """Each pair's RelativeSize, as an object array of enum members."""
    d = np.asarray(hybrid_long, np.int64) - np.asarray(real_long, np.int64)
    out = np.full(len(d), RelativeSize.SIMILAR, dtype=object)
    out[d > SIZE_MATCH_PX] = RelativeSize.HYBRID_LARGER
    out[d < -SIZE_MATCH_PX] = RelativeSize.HYBRID_SMALLER
    return out


def seam_report(real: np.ndarray, hybrid: np.ndarray, real_long: np.ndarray, hybrid_long: np.ndarray,
                photo: np.ndarray, tau: np.float32) -> dict:
    """The seam report of docs/v3a1-spec.md §2.3:
    1. median paired gap (hybrid − real) with a photo-clustered 95% interval, for all pairs and per RelativeSize;
    2. share of pairs where the hybrid scores strictly higher, with interval;
    3. median real and median hybrid presence, each with interval;
    4. the said-yes check at τ (presence_report.seam_check).
    Raises ValueError if the five arrays differ in length, are empty, or a size group is empty (in the data
    or in a bootstrap resample, whose median would be undefined).
    """
    n = len(real)
    if not n or any(len(a) != n for a in (hybrid, real_long, hybrid_long, photo)):
        raise ValueError(f"need five equal non-empty arrays, got {[len(a) for a in (real, hybrid, real_long, hybrid_long, photo)]}")
    real, hybrid = np.asarray(real, np.float64), np.asarray(hybrid, np.float64)
    gap = hybrid - real
    groups = relative_size(real_long, hybrid_long)

    def interval(values: np.ndarray, within: np.ndarray, stat) -> dict:
        if not within.any():
            raise ValueError("a size group holds no pair")
        def on_rows(rows: np.ndarray) -> float:
            kept = rows[within[rows]]
            if not len(kept):
                raise ValueError("a bootstrap resample holds no pair of a size group")
            return float(stat(values[kept]))
        lo, hi = cluster_bootstrap(on_rows, np.asarray(photo), BOOTSTRAP_RESAMPLES, np.random.default_rng(BOOTSTRAP_SEED))
        return {"n": int(within.sum()), "value": float(stat(values[within])), "lo": lo, "hi": hi}

    everyone = np.ones(n, bool)
    return {
        "size_match_px": SIZE_MATCH_PX,
        "median_gap": {"all": interval(gap, everyone, np.median)}
                      | {g.value: interval(gap, groups == g, np.median) for g in RelativeSize},
        "share_hybrid_higher": interval((hybrid > real).astype(np.float64), everyone, np.mean),
        "median_real": interval(real, everyone, np.median),
        "median_hybrid": interval(hybrid, everyone, np.median),
        "said_yes": seam_check(real.astype(np.float32), hybrid.astype(np.float32), tau),
    }
