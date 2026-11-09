"""Confidence intervals for rates and an exact paired test (pure, no I/O)."""

import math
from dataclasses import dataclass

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
