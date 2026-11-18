"""Whole-photo ground truth and matching (docs/v3b-spec.md A6, A7)."""

import random
from pathlib import Path

import numpy as np
import pytest

from findobject.data.gtsdb import RealPhoto, RealSign, RealSplit, photos_in, read_gtsdb
from findobject.eval.photo_match import FalsePositive, best_raw_score, match_photo
from findobject.eval.photo_truth import TruthSign, photo_truth
from findobject.eval.real_report import SizeBucket, size_bucket
from findobject.inference.detection import Detection, detection_order

REAL_ROOT = Path("data/raw/gtsdb/FullIJCNN2013")
REAL_DEV = Path("data/processed/real_dev.npz")
needs_gtsdb = pytest.mark.skipif(not (REAL_ROOT / "gt.txt").exists(), reason="GTSDB not downloaded")


def _sign(i, box, class_id, stacked=False):
    return TruthSign(gt_index=i, box=box, class_id=class_id, long_side=max(box[2] - box[0], box[3] - box[1]),
                     stacked=stacked)


def _det(x, y, score, class_id):
    return Detection(x=x, y=y, score=score, class_id=class_id, class_prob=0.9, scale=1.0)


def _photo(*signs):
    return RealPhoto(path=Path("x.ppm"), index=0, signs=tuple(RealSign(box=b, class_id=c) for b, c in signs))


# --- A7: ground truth ---

def test_exact_repeats_are_dropped_and_indices_keep_their_gt_rows():
    truth = photo_truth(_photo(((0, 0, 10, 10), 1), ((0, 0, 10, 10), 1), ((0, 0, 10, 10), 2), ((50, 50, 60, 60), 1)))
    assert [(s.gt_index, s.class_id) for s in truth] == [(0, 1), (2, 2), (3, 1)]


def test_stacked_rule_is_euclidean_strict_and_uses_its_own_long_side():
    # Long side 20 -> stacked if another center is < 30 px away. Centers (10, 10) and (10 + dx, 10).
    def flags(dx, other_side=20):
        return [s.stacked for s in photo_truth(_photo(((0, 0, 20, 20), 0),
                                                      ((dx, 0, dx + other_side, other_side), 0)))]
    assert flags(29) == [True, True]
    assert flags(30) == [False, False]  # strict
    # Diagonal: Chebyshev 25 < 30 but Euclidean 35.4 >= 30 -> not stacked.
    assert [s.stacked for s in photo_truth(_photo(((0, 0, 20, 20), 0), ((25, 25, 45, 45), 0)))] == [False, False]
    # A big sign next to a small one: each uses its own long side.
    big_small = photo_truth(_photo(((0, 0, 40, 40), 0), ((40, 14, 52, 26), 0)))  # centers 26 px apart
    assert [s.stacked for s in big_small] == [True, False]  # 26 < 60 for the big one, 26 >= 18 for the small one


@needs_gtsdb
def test_real_label_counts_match_the_spec():
    photos = read_gtsdb(REAL_ROOT)
    dev = {p.index: photo_truth(p) for p in photos_in(photos, RealSplit.DEV)}
    test = [s for p in photos_in(photos, RealSplit.TEST) for s in photo_truth(p)]
    raw_dev = {p.index: p for p in photos_in(photos, RealSplit.DEV)}
    dropped = [(i, r) for i, p in raw_dev.items() for r in range(len(p.signs))
               if r not in {s.gt_index for s in dev[i]}]
    assert dropped == [(340, 2)]
    signs = [s for t in dev.values() for s in t]
    assert len(signs) == 851 and len(test) == 361
    assert sum(s.stacked for s in signs) == 234 and sum(s.stacked for s in test) == 85
    buckets = [size_bucket(s.long_side) for s in signs]
    assert [buckets.count(b) for b in SizeBucket] == [85, 482, 284]
    assert sum(s.long_side >= 64 for s in signs) == 139
    medium = [s for s, b in zip(signs, buckets) if b is SizeBucket.MEDIUM]
    assert (sum(s.stacked for s in medium), sum(not s.stacked for s in medium)) == (146, 336)
    if REAL_DEV.exists():
        z = np.load(REAL_DEV)
        windowed = set(zip(z["photo_index"].tolist(), z["sign_index"].tolist()))
        keys = {(i, s.gt_index) for i, t in dev.items() for s in t}
        assert windowed <= keys and len(windowed) == 645 and len(keys - windowed) == 206
        assert all(s.stacked for i, t in dev.items() for s in t if (i, s.gt_index) not in windowed)


# --- A6: matching ---

def test_box_edges_are_half_open():
    truth = [_sign(0, (10, 10, 20, 20), 5)]
    assert match_photo([_det(10, 10, 0.9, 5)], truth).found == (0,)
    assert match_photo([_det(20, 15, 0.9, 5)], truth).found == (None,)
    assert match_photo([_det(15, 20, 0.9, 5)], truth).fp_kind == (FalsePositive.BACKGROUND,)


def test_a_point_in_two_boxes_goes_to_the_nearer_center():
    truth = [_sign(0, (0, 0, 40, 40), 5), _sign(1, (30, 30, 50, 50), 5)]
    assert match_photo([_det(35, 35, 0.9, 5)], truth).found == (1,)  # 7 px from (40, 40), 21 from (20, 20)
    assert match_photo([_det(30, 30, 0.9, 5)], truth).found == (0,)  # 14.1 px from both: lower gt_index


def test_found_and_located_differ_when_the_class_is_wrong():
    m = match_photo([_det(15, 15, 0.9, 7)], [_sign(0, (10, 10, 20, 20), 5)])
    assert m.found == (None,) and m.located == (0,) and m.fp_kind == (FalsePositive.WRONG_CLASS,)


def test_a_second_same_class_detection_on_a_matched_sign_is_a_duplicate():
    m = match_photo([_det(15, 15, 0.9, 5), _det(12, 12, 0.6, 5)], [_sign(0, (10, 10, 20, 20), 5)])
    assert m.found == (0, None) and m.fp_kind == (None, FalsePositive.DUPLICATE)


def test_reviewer_case_wrong_class_first_then_found():
    # D1 (class 2) scores above D2 (class 1); both point only at sign A (class 1).
    m = match_photo([_det(15, 15, 0.9, 2), _det(16, 16, 0.8, 1)], [_sign(0, (10, 10, 20, 20), 1)])
    assert m.fp_kind == (FalsePositive.WRONG_CLASS, None) and m.found == (None, 0)
    assert m.located == (0, None)


def test_match_photo_refuses_detections_out_of_order():
    with pytest.raises(ValueError):
        match_photo([_det(15, 15, 0.5, 5), _det(15, 15, 0.9, 5)], [])


def _random_case(rng):
    truth = []
    for i in range(rng.randint(0, 5)):
        x, y, side = rng.randint(0, 80), rng.randint(0, 80), rng.randint(5, 30)
        truth.append(_sign(i, (x, y, x + side, y + side), rng.randint(0, 3)))
    dets = [Detection(x=rng.uniform(0, 110), y=rng.uniform(0, 110), score=round(rng.uniform(0.05, 1), 1),
                      class_id=rng.randint(0, 3), class_prob=0.5, scale=rng.choice([1.0, 0.5, 0.25]))
            for _ in range(rng.randint(0, 25))]
    return sorted(dets, key=detection_order), truth


def test_prefix_property_every_threshold_gives_the_full_pass_restricted():
    rng = random.Random(0)
    for _ in range(300):
        dets, truth = _random_case(rng)
        full = match_photo(dets, truth)
        for t in sorted({d.score for d in dets}):
            n = sum(d.score >= t for d in dets)  # detections >= t are a prefix of detection order
            part = match_photo(dets[:n], truth)
            assert part.found == full.found[:n] and part.located == full.located[:n]
            assert part.fp_kind == full.fp_kind[:n]


def test_kinds_are_set_exactly_where_found_is_none():
    rng = random.Random(1)
    for _ in range(200):
        m = match_photo(*_random_case(rng))
        assert all((f is None) == (k is not None) for f, k in zip(m.found, m.fp_kind))
        found = [f for f in m.found if f is not None]
        assert len(found) == len(set(found))  # each sign matched at most once


def test_best_raw_score_takes_the_top_raw_detection_per_sign():
    truth = [_sign(0, (10, 10, 20, 20), 5), _sign(1, (100, 100, 120, 120), 5)]
    raw = [_det(15, 15, 0.4, 9), _det(12, 12, 0.7, 5), _det(50, 50, 0.99, 5)]
    assert best_raw_score(raw, truth) == {0: 0.7}
