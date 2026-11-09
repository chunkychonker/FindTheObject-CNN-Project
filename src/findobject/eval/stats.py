"""Statistical summaries of scored outcomes: rate intervals, a paired test, AUROC, a clustered bootstrap. Pure."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054  # standard normal 97.5th percentile


@dataclass(frozen=True)
class Rate:
    """k successes out of n, with a Wilson 95% interval [lo, hi]."""

    k: int
    n: int
    lo: float
    hi: float

    @property
    def value(self) -> float:
        return self.k / self.n


def wilson(k: int, n: int) -> Rate:
    """Wilson score 95% interval for k / n. Unlike the normal approximation it stays inside [0, 1] and is
    sensible at k = 0 or k = n, which matters for small per-bucket counts.

    Raises ValueError if n < 1 or k is not in [0, n].
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got k={k}, n={n}")
    p, z2 = k / n, Z95 * Z95
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return Rate(k, n, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes.

    only_a = pairs that A got right and B wrong; only_b = the reverse. Pairs both got right (or both wrong)
    carry no information about which is better and are not passed. Under "A and B are equally good" each
    discordant pair is a fair coin, so p = P(a split at least this lopsided) under Binomial(only_a + only_b,
    1/2), doubled and capped at 1. Returns 1.0 when there are no discordant pairs.

    Raises ValueError if either count is negative.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError(f"counts must be >= 0, got {only_a}, {only_b}")
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1))  # exact integers; divide once at the end
    return min(1.0, 2 * tail / 2**n)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC curve: the chance a random positive scores above a random negative, ties
    counted half (the Mann-Whitney U statistic divided by n_pos * n_neg).

    Raises ValueError if either array is empty, not 1-D, or holds a non-finite score.
    """
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    for name, a in (("pos", pos), ("neg", neg)):
        if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a non-empty 1-D array of finite scores, got shape {a.shape}")
    sorted_neg = np.sort(neg)
    below = np.searchsorted(sorted_neg, pos, side="left")  # negatives strictly below each positive
    tied = np.searchsorted(sorted_neg, pos, side="right") - below
    return (int(below.sum()) + 0.5 * int(tied.sum())) / (len(pos) * len(neg))


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_resamples: int,
                      rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% interval of `stat` over resamples that draw whole clusters with replacement.

    clusters[i] is row i's cluster (e.g. its photo). Each resample draws as many clusters as there are
    distinct ones and passes stat the row indices of every drawn cluster (a cluster drawn twice brings its
    rows twice). Rows of one cluster are not independent, so resampling rows one by one would give too
    narrow an interval. With one row per cluster this is the ordinary bootstrap.
    Raises ValueError if clusters is empty or not 1-D or n_resamples < 1; whatever stat raises (e.g. a
    resample with no positives), uncaught.
    """
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or len(clusters) == 0 or n_resamples < 1:
        raise ValueError(f"need non-empty 1-D clusters and n_resamples >= 1, got {clusters.shape}, {n_resamples}")
    order = np.argsort(clusters, kind="stable")  # rows grouped by cluster
    _, starts, sizes = np.unique(clusters[order], return_index=True, return_counts=True)
    values = np.empty(n_resamples)
    for r in range(n_resamples):
        drawn = rng.integers(len(starts), size=len(starts))
        n_rows = sizes[drawn]
        # Row positions start..start+size-1 for every drawn cluster, without a Python loop over clusters.
        offset = np.arange(n_rows.sum()) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
        values[r] = stat(order[np.repeat(starts[drawn], n_rows) + offset])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)
