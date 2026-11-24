"""Whole-photo operating points: the threshold for a false-positives-per-photo budget, the FROC curve and point AP (docs/v3b-spec.md §2.4). Pure."""

import math
from fractions import Fraction

import numpy as np

from findobject.inference.peaks import MIN_PEAK_SCORE


def threshold_at_fppi(fp_scores: np.ndarray, n_photos: int, fppi: float) -> np.float32:
    """Smallest float32 τ with at most k = floor(fppi * n_photos) of fp_scores >= τ: sort descending
    s_1 >= ... >= s_n, then τ = nextafter(s_{k+1}, +inf). Scores tied with s_{k+1} fall below τ.
    If there are at most k scores in all, every detection is allowed and τ = MIN_PEAK_SCORE.

    Uses only false-positive scores, so no sign's score enters the choice. k uses fppi as the decimal it was
    written as (as threshold_at_false_alarm does). Raises ValueError if fp_scores is not 1-D or has a
    non-finite value, n_photos < 1, or fppi < 0.
    """
    fp = np.asarray(fp_scores, dtype=np.float32)
    if fp.ndim != 1 or not np.isfinite(fp).all():
        raise ValueError(f"need a 1-D array of finite scores, got shape {fp.shape}")
    if n_photos < 1 or fppi < 0:
        raise ValueError(f"need n_photos >= 1 and fppi >= 0, got {n_photos}, {fppi}")
    k = math.floor(Fraction(fppi).limit_denominator(10**9) * n_photos)
    if len(fp) <= k:
        return np.float32(MIN_PEAK_SCORE)
    s_k1 = np.sort(fp)[::-1][k]
    return np.nextafter(s_k1, np.float32(np.inf), dtype=np.float32)


def _check(scores: np.ndarray, is_tp: np.ndarray, n_signs: int) -> tuple[np.ndarray, np.ndarray]:
    scores, is_tp = np.asarray(scores, dtype=np.float64), np.asarray(is_tp, dtype=bool)
    if scores.ndim != 1 or scores.shape != is_tp.shape or not np.isfinite(scores).all():
        raise ValueError(f"need equal-length 1-D finite scores and flags, got {scores.shape}, {is_tp.shape}")
    if n_signs < 1:
        raise ValueError(f"need n_signs >= 1, got {n_signs}")
    if is_tp.sum() > n_signs:
        raise ValueError(f"{int(is_tp.sum())} true positives but only {n_signs} signs")
    return scores, is_tp


def _cumulative(scores: np.ndarray, is_tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, TP count, FP count) at each distinct score, descending: counts of detections >= it."""
    order = np.argsort(-scores, kind="stable")
    s, tp = scores[order], is_tp[order]
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(~tp)
    last_of_each = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1] if len(s) else np.array([], dtype=int)
    return s[last_of_each], tp_cum[last_of_each], fp_cum[last_of_each]


def froc_curve(scores: np.ndarray, is_tp: np.ndarray, n_signs: int, n_photos: int) -> dict[str, np.ndarray]:
    """Pooled kept detections of all photos -> thresholds (every distinct score, descending), found recall
    and FPPI of the detections >= each. Raises ValueError for unequal lengths, a non-finite score,
    n_signs < 1, n_photos < 1, or more true positives than signs."""
    scores, is_tp = _check(scores, is_tp, n_signs)
    if n_photos < 1:
        raise ValueError(f"need n_photos >= 1, got {n_photos}")
    t, tp, fp = _cumulative(scores, is_tp)
    return {"thresholds": t, "recall": tp / n_signs, "fppi": fp / n_photos}


def ap_point(scores: np.ndarray, is_tp: np.ndarray, n_signs: int) -> float:
    """Area under found precision vs found recall, all-points interpolation: at each recall, precision is the
    best precision at that recall or higher. 0.0 with no detections. Not comparable to IoU-based AP.
    Raises as froc_curve."""
    scores, is_tp = _check(scores, is_tp, n_signs)
    _, tp, fp = _cumulative(scores, is_tp)
    if len(tp) == 0:
        return 0.0
    recall, precision = tp / n_signs, tp / (tp + fp)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.r_[0.0, recall])
    return float((steps * envelope).sum())
