"""Turn per-scene presence outcomes into v3a's reported numbers, as JSON-ready dicts (docs/v3a-spec.md §2.6). Pure.

Every number comes with a 95% interval from an Intervals object, which fixes how scenes are resampled:
PhotoIntervals for real splits (whole photos, since windows of one photo are not independent),
SceneIntervals for synthetic splits (Wilson for rates; each scene its own cluster for AUROC). Every
bootstrap of one report draws the same resamples (fixed seed), so its intervals are coherent.
"""

from collections.abc import Callable
from enum import Enum

import numpy as np

from findobject.eval.presence import threshold_at_false_alarm
from findobject.eval.real_report import discordant
from findobject.eval.stats import auroc, cluster_bootstrap, mcnemar_exact, wilson

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
MATCHED_RATES = (0.01, 0.05)  # descriptive ROC points on a split's own negatives; never used for grading


class Side(Enum):
    POS = "pos"
    NEG = "neg"


class PhotoIntervals:
    """Resample whole clusters (photos): a drawn photo brings every positive and negative window it has."""

    def __init__(self, pos_clusters: np.ndarray, neg_clusters: np.ndarray) -> None:
        """pos_clusters[i] is positive row i's photo, neg_clusters[j] negative row j's; the same id in both
        means the same photo. Raises ValueError if either array is empty or not 1-D."""
        pos_clusters, neg_clusters = np.asarray(pos_clusters), np.asarray(neg_clusters)
        if pos_clusters.ndim != 1 or neg_clusters.ndim != 1 or not len(pos_clusters) or not len(neg_clusters):
            raise ValueError("need non-empty 1-D cluster arrays for positives and negatives")
        self.n_pos, self.n_neg = len(pos_clusters), len(neg_clusters)
        self._clusters = np.concatenate([pos_clusters, neg_clusters])

    def bootstrap(self, stat: Callable[[np.ndarray, np.ndarray], float]) -> tuple[float, float]:
        """Percentile 95% interval of stat(positive rows, negative rows) over BOOTSTRAP_RESAMPLES resamples.
        Raises whatever stat raises (e.g. a resample that holds no positive)."""
        p = self.n_pos
        return cluster_bootstrap(lambda idx: stat(idx[idx < p], idx[idx >= p] - p), self._clusters,
                                 BOOTSTRAP_RESAMPLES, np.random.default_rng(BOOTSTRAP_SEED))

    def auroc(self, pos: np.ndarray, neg: np.ndarray, pos_within: np.ndarray | None = None,
              neg_within: np.ndarray | None = None) -> dict:
        """AUROC of pos vs neg, each optionally restricted to the rows where its mask is True, with interval.
        Raises ValueError if a restriction leaves no rows, in the data or in a resample."""
        pk, nk = _all(len(pos), pos_within), _all(len(neg), neg_within)
        lo, hi = self.bootstrap(lambda p, n: auroc(pos[p[pk[p]]], neg[n[nk[n]]]))
        return {"value": auroc(pos[pk], neg[nk]), "lo": lo, "hi": hi}

    def rate(self, ok: np.ndarray, side: Side, within: np.ndarray | None = None) -> dict:
        """Share of True in ok (one entry per row of `side`, restricted to `within`), with its interval."""
        keep = _all(len(ok), within)
        pick = (lambda p, n: p) if side is Side.POS else (lambda p, n: n)
        lo, hi = self.bootstrap(lambda p, n: float(ok[(rows := pick(p, n))[keep[rows]]].mean()))
        return _rate_dict(int(ok[keep].sum()), int(keep.sum()), lo, hi)


class SceneIntervals(PhotoIntervals):
    """Synthetic splits: scenes are independent. Rates get Wilson intervals; AUROC the ordinary two-sample
    bootstrap, which redraws n_pos positives and n_neg negatives separately (so no resample lacks either)."""

    def __init__(self, n_pos: int, n_neg: int) -> None:
        super().__init__(np.arange(n_pos), np.arange(n_pos, n_pos + n_neg))

    def bootstrap(self, stat: Callable[[np.ndarray, np.ndarray], float]) -> tuple[float, float]:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        values = [stat(rng.integers(self.n_pos, size=self.n_pos), rng.integers(self.n_neg, size=self.n_neg))
                  for _ in range(BOOTSTRAP_RESAMPLES)]
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    def rate(self, ok: np.ndarray, side: Side, within: np.ndarray | None = None) -> dict:
        return wilson_rate(ok[_all(len(ok), within)])


def wilson_rate(ok: np.ndarray) -> dict:
    """Share of True in ok with its Wilson 95% interval. Raises ValueError if ok is empty."""
    r = wilson(int(ok.sum()), len(ok))
    return _rate_dict(r.k, r.n, r.lo, r.hi)


def family_metrics(presence: np.ndarray, correct: np.ndarray, hit: np.ndarray, neg_presence: np.ndarray,
                   tau: np.float32, intervals: PhotoIntervals, pos_within: np.ndarray | None = None) -> dict:
    """AUROC, false alarms at τ, found at τ (presence >= τ, peak in box and class right), plain joint
    (ignoring τ). presence/correct/hit are per positive, neg_presence per negative; pos_within restricts the
    positives to a subset (v2's unseen). Raises ValueError (from auroc) for empty or non-finite scores."""
    joint = correct & hit
    return {
        "auroc": intervals.auroc(presence, neg_presence, pos_within=pos_within),
        "false_alarm": intervals.rate(neg_presence >= tau, Side.NEG),
        "found": intervals.rate((presence >= tau) & joint, Side.POS, pos_within),
        "joint": intervals.rate(joint, Side.POS, pos_within),
    }


def matched_found(presence: np.ndarray, joint: np.ndarray, neg_presence: np.ndarray) -> dict:
    """For each rate in MATCHED_RATES: the threshold giving that false-alarm rate on these negatives, and
    the share of positives found at it. Descriptive only."""
    out = {}
    for rate in MATCHED_RATES:
        t = threshold_at_false_alarm(neg_presence, rate)
        out[str(rate)] = {"threshold": float(t), "false_alarms": int((neg_presence >= t).sum()),
                          "found": float(((presence >= t) & joint).mean())}
    return out


def auroc_difference(a_pos: np.ndarray, a_neg: np.ndarray, b_pos: np.ndarray, b_neg: np.ndarray,
                     intervals: PhotoIntervals) -> dict:
    """AUROC_a − AUROC_b on the same rows, with a paired interval: both models are scored on each resample.
    Raises ValueError if a and b differ in shape (they must be the same scenes)."""
    if a_pos.shape != b_pos.shape or a_neg.shape != b_neg.shape:
        raise ValueError(f"a and b must score the same scenes: {a_pos.shape}/{a_neg.shape} vs {b_pos.shape}/{b_neg.shape}")
    lo, hi = intervals.bootstrap(lambda p, n: auroc(a_pos[p], a_neg[n]) - auroc(b_pos[p], b_neg[n]))
    return {"value": auroc(a_pos, a_neg) - auroc(b_pos, b_neg), "lo": lo, "hi": hi}


def seam_check(real_presence: np.ndarray, hybrid_presence: np.ndarray, tau: np.float32) -> dict:
    """Said-yes (presence >= τ) on real windows vs their hybrid twins, row i = the same sign in both: rates,
    exact McNemar, and the median paired presence difference, hybrid minus real. A model relying on the
    paste seam says yes less often on real. Raises ValueError if the arrays differ in length or are empty."""
    if len(real_presence) != len(hybrid_presence) or not len(real_presence):
        raise ValueError(f"need equal non-empty arrays, got {len(real_presence)} and {len(hybrid_presence)}")
    real_yes, hybrid_yes = real_presence >= tau, hybrid_presence >= tau
    only_real, only_hybrid = discordant(real_yes, hybrid_yes)
    return {
        # No interval: windows of one photo are not independent, so Wilson would be too narrow.
        "real_yes": _rate_dict(int(real_yes.sum()), len(real_yes), None, None),
        "hybrid_yes": _rate_dict(int(hybrid_yes.sum()), len(hybrid_yes), None, None),
        "gap_points": float(100 * (hybrid_yes.mean() - real_yes.mean())),  # P8: hybrid minus real
        "only_real": only_real, "only_hybrid": only_hybrid, "mcnemar_p": mcnemar_exact(only_real, only_hybrid),
        "median_presence_diff_hybrid_minus_real":
            float(np.median(hybrid_presence.astype(np.float64) - real_presence.astype(np.float64))),
    }


def _all(n: int, mask: np.ndarray | None) -> np.ndarray:
    return np.ones(n, bool) if mask is None else np.asarray(mask, bool)


def _rate_dict(k: int, n: int, lo: float | None, hi: float | None) -> dict:
    return {"k": k, "n": n, "value": k / n, "lo": lo, "hi": hi}
