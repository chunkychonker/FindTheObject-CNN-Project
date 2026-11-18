"""Summarize matched whole-photo detections into recall, FPPI and their breakdowns at named thresholds (docs/v3b-spec.md §2.4). Pure."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from findobject.eval.froc import ap_point, froc_curve
from findobject.eval.photo_match import PhotoMatch
from findobject.eval.photo_truth import TruthSign
from findobject.eval.real_report import SizeBucket, size_bucket
from findobject.eval.stats import cluster_bootstrap, mcnemar_exact
from findobject.inference.detection import Detection

LARGE_FOR_PYRAMID = 64  # px, native long side: W3's "large enough to need the pyramid" subset
TAU_SYN = "tau_syn"  # the run's stored window τ; W1 compares with windows only here
TAU_W = "tau_w"  # dev FPPI <= 0.5


@dataclass(frozen=True)
class PhotoRecord:
    """One photo's signs, its kept detections (in detection_order), their matches, and each sign's best raw score."""

    index: int
    truth: tuple[TruthSign, ...]
    kept: tuple[Detection, ...]
    match: PhotoMatch
    best_raw: dict[int, float]


@dataclass(frozen=True)
class _Signs:
    """One row per sign, over all photos. A score is -inf where the sign was never found / located."""

    photo: np.ndarray  # position of the sign's photo in the photo list
    key: list[tuple[int, int]]  # (photo index, gt_index)
    found_score: np.ndarray
    located_score: np.ndarray
    raw_score: np.ndarray
    long_side: np.ndarray
    stacked: np.ndarray
    windowed: np.ndarray  # in the windowed split
    medium: np.ndarray


def _sign_table(photos: Sequence[PhotoRecord], windowed: dict[tuple[int, int], bool]) -> _Signs:
    rows = []
    for pos, p in enumerate(photos):
        found_by = {g: d.score for d, g in zip(p.kept, p.match.found) if g is not None}
        located_by = {g: d.score for d, g in zip(p.kept, p.match.located) if g is not None}
        for s in p.truth:
            rows.append((pos, (p.index, s.gt_index), found_by.get(s.gt_index, -np.inf),
                         located_by.get(s.gt_index, -np.inf), p.best_raw.get(s.gt_index, -np.inf),
                         s.long_side, s.stacked, (p.index, s.gt_index) in windowed))
    cols = list(zip(*rows)) if rows else [()] * 8
    long_side = np.array(cols[5], dtype=int)
    return _Signs(photo=np.array(cols[0], dtype=int), key=list(cols[1]),
                  found_score=np.array(cols[2], dtype=float), located_score=np.array(cols[3], dtype=float),
                  raw_score=np.array(cols[4], dtype=float), long_side=long_side,
                  stacked=np.array(cols[6], dtype=bool), windowed=np.array(cols[7], dtype=bool),
                  medium=np.array([size_bucket(int(n)) is SizeBucket.MEDIUM for n in long_side], dtype=bool))


def _groups(signs: _Signs, with_windowed: bool) -> dict[str, np.ndarray]:
    buckets = np.array([size_bucket(int(n)).value for n in signs.long_side])
    groups = {
        "all": np.ones(len(buckets), dtype=bool),
        **{b.value: buckets == b.value for b in SizeBucket},
        f"ge{LARGE_FOR_PYRAMID}": signs.long_side >= LARGE_FOR_PYRAMID,
        "stacked": signs.stacked,
        "isolated": ~signs.stacked,
        "medium_stacked": signs.medium & signs.stacked,
        "medium_isolated": signs.medium & ~signs.stacked,
    }
    if with_windowed:
        groups["windowed"] = signs.windowed
        groups["not_windowed"] = ~signs.windowed
    return groups


def _per_photo(values: np.ndarray, photo: np.ndarray, n_photos: int) -> np.ndarray:
    """Sum a per-row quantity within each photo -> [n_photos]."""
    return np.bincount(photo, weights=values.astype(float), minlength=n_photos)


def _interval(stat: Callable[[np.ndarray], float], n_photos: int, rng_seed: int,
              n_resamples: int) -> tuple[float, float] | None:
    """Bootstrap over photos (one row per photo). A fresh rng with the same seed per statistic means every
    statistic, and so every difference, uses the same resamples. None if some resample leaves the statistic
    undefined (it drew no sign of a group, so 0 / 0): the interval doesn't exist, rather than being NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lo, hi = cluster_bootstrap(stat, np.arange(n_photos), n_resamples, np.random.default_rng(rng_seed))
    return None if np.isnan(lo) or np.isnan(hi) else (lo, hi)


def _ratio(num: np.ndarray, den: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda idx: num[idx].sum() / den[idx].sum()


def detect_report(photos: Sequence[PhotoRecord], thresholds: dict[str, float],
                  windowed: dict[tuple[int, int], bool], rng_seed: int, n_resamples: int) -> dict:
    """§2.4 at each named threshold: for every group, found (with a photo-clustered 95% interval), located,
    located before suppression, and class accuracy given located (found among located); the isolated -
    stacked found differences, overall and in the medium bucket, on the same resamples; FPPI with its
    interval; false positives by kind and by scale. At TAU_SYN, when `windowed` is given, whole-photo vs
    windowed found on the windowed signs with an exact McNemar test. Once for all thresholds: the FROC
    curve and AP_point.

    `windowed` maps (photo index, gt_index) to the windowed found outcome; pass {} where there is none.
    An interval is None when some resample draws no sign of its group (possible only for groups in very
    few photos); graders must refuse a None they need. Empty groups are left out.
    Raises ValueError for no photos, no signs, or a windowed key that names no sign.
    """
    if not photos:
        raise ValueError("no photos")
    signs = _sign_table(photos, windowed)
    if len(signs.key) == 0:
        raise ValueError("no signs")
    unknown = set(windowed) - set(signs.key)
    if unknown:
        raise ValueError(f"windowed keys name no sign: {sorted(unknown)[:5]}")
    n_photos = len(photos)
    groups = _groups(signs, with_windowed=bool(windowed))

    det_photo = np.array([pos for pos, p in enumerate(photos) for _ in p.kept], dtype=int)
    det_score = np.array([d.score for p in photos for d in p.kept], dtype=float)
    det_scale = np.array([d.scale for p in photos for d in p.kept], dtype=float)
    det_tp = np.array([g is not None for p in photos for g in p.match.found], dtype=bool)
    det_kind = [k.value if k is not None else None for p in photos for k in p.match.fp_kind]

    def rate(mask: np.ndarray, ok: np.ndarray) -> dict:
        k, n = int((ok & mask).sum()), int(mask.sum())
        return {"k": k, "n": n, "value": k / n if n else None}

    at = {}
    for name, t in thresholds.items():
        t = float(t)
        found, located, raw = signs.found_score >= t, signs.located_score >= t, signs.raw_score >= t
        out_groups = {}
        for g, mask in groups.items():
            if not mask.any():
                continue
            entry = {"found": rate(mask, found), "located": rate(mask, located),
                     "located_before_suppression": rate(mask, raw),
                     "class_acc_given_located": rate(mask & located, found)}
            entry["found"]["ci"] = _interval(_ratio(_per_photo(found & mask, signs.photo, n_photos),
                                                    _per_photo(mask, signs.photo, n_photos)),
                                             n_photos, rng_seed, n_resamples)
            out_groups[g] = entry

        def diff(a: str, b: str) -> dict:
            fa, na = _per_photo(found & groups[a], signs.photo, n_photos), _per_photo(groups[a], signs.photo, n_photos)
            fb, nb = _per_photo(found & groups[b], signs.photo, n_photos), _per_photo(groups[b], signs.photo, n_photos)
            stat = lambda idx: fa[idx].sum() / na[idx].sum() - fb[idx].sum() / nb[idx].sum()
            return {"value": float(stat(np.arange(n_photos))), "ci": _interval(stat, n_photos, rng_seed, n_resamples)}

        fp = (~det_tp) & (det_score >= t)
        fp_pp = _per_photo(fp, det_photo, n_photos)
        entry = {
            "threshold": t,
            "groups": out_groups,
            "differences": {"isolated_minus_stacked": diff("isolated", "stacked"),
                            "medium_isolated_minus_stacked": diff("medium_isolated", "medium_stacked")},
            "fppi": {"n_fp": int(fp.sum()), "value": float(fp.sum() / n_photos),
                     "ci": _interval(lambda idx: fp_pp[idx].sum() / len(idx), n_photos, rng_seed, n_resamples)},
            "fp_by_kind": dict(Counter(k for k, f in zip(det_kind, fp) if f)),
            "fp_by_scale": {str(s): int((fp & (det_scale == s)).sum()) for s in sorted(set(det_scale.tolist()))},
        }
        if name == TAU_SYN and windowed:
            w = groups["windowed"]
            win_found = np.array([windowed.get(k, False) for k in signs.key], dtype=bool)
            only_whole, only_window = int((w & found & ~win_found).sum()), int((w & ~found & win_found).sum())
            entry["windowed_vs_whole"] = {"n": int(w.sum()), "whole_found": int((w & found).sum()),
                                          "window_found": int((w & win_found).sum()), "only_whole": only_whole,
                                          "only_window": only_window,
                                          "mcnemar_p": mcnemar_exact(only_whole, only_window)}
        at[name] = entry

    curve = froc_curve(det_score, det_tp, len(signs.key), n_photos)
    return {"n_photos": n_photos, "n_signs": len(signs.key), "photo_indices": [p.index for p in photos],
            "thresholds": {k: float(v) for k, v in thresholds.items()}, "at": at,
            "froc": {k: v.tolist() for k, v in curve.items()}, "ap_point": ap_point(det_score, det_tp, len(signs.key))}
