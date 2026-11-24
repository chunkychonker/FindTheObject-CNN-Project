"""Whole-photo thresholds, FROC and point AP (docs/v3b-spec.md A8, A9)."""

import numpy as np
import pytest

from findobject.eval.froc import ap_point, froc_curve, threshold_at_fppi
from findobject.inference.peaks import MIN_PEAK_SCORE


# --- A8 ---

def test_at_most_floor_fppi_times_photos_false_positives_are_at_or_above_tau():
    rng = np.random.default_rng(0)
    fp = rng.uniform(0.05, 1, 1000).astype(np.float32)
    for n_photos, fppi in [(600, 0.5), (600, 0.1), (300, 1.0), (7, 0.3)]:
        tau = threshold_at_fppi(fp, n_photos, fppi)
        k = int(np.floor(fppi * n_photos + 1e-9))
        assert (fp >= tau).sum() <= k
        assert (fp >= np.nextafter(tau, np.float32(0))).sum() > k  # smallest such τ


def test_ties_fall_below_tau():
    fp = np.array([0.9, 0.7, 0.7, 0.7, 0.2], dtype=np.float32)
    tau = threshold_at_fppi(fp, n_photos=4, fppi=0.5)  # k = 2: s_3 = 0.7 is tied with s_2 and s_4
    assert (fp >= tau).sum() == 1 and tau > np.float32(0.7)


def test_a_budget_larger_than_all_false_positives_allows_everything():
    assert threshold_at_fppi(np.array([0.9, 0.8], dtype=np.float32), 4, 0.5) == np.float32(MIN_PEAK_SCORE)
    assert threshold_at_fppi(np.array([], dtype=np.float32), 600, 0.5) == np.float32(MIN_PEAK_SCORE)


def test_fppi_uses_the_decimal_as_written():
    fp = np.linspace(0.1, 0.9, 200).astype(np.float32)
    assert (fp >= threshold_at_fppi(fp, 100, 0.29)).sum() == 29  # 0.29 * 100 is 28.999... in floats


@pytest.mark.parametrize("args", [(np.array([0.5]), 0, 0.5), (np.array([0.5]), 10, -0.1),
                                  (np.array([np.nan]), 10, 0.5), (np.zeros((2, 2)), 10, 0.5)])
def test_threshold_at_fppi_rejects_bad_input(args):
    with pytest.raises(ValueError):
        threshold_at_fppi(*args)


# --- A9 ---

def test_froc_and_ap_on_the_hand_case():
    scores, is_tp = np.array([0.7, 0.9, 0.8]), np.array([True, True, False])  # any input order
    curve = froc_curve(scores, is_tp, n_signs=2, n_photos=1)
    assert curve["thresholds"].tolist() == [0.9, 0.8, 0.7]
    assert curve["recall"].tolist() == [0.5, 0.5, 1.0]
    assert curve["fppi"].tolist() == [0.0, 1.0, 1.0]
    assert ap_point(scores, is_tp, n_signs=2) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_tied_scores_form_one_threshold():
    curve = froc_curve(np.array([0.9, 0.5, 0.5]), np.array([False, True, False]), n_signs=1, n_photos=2)
    assert curve["thresholds"].tolist() == [0.9, 0.5]
    assert curve["recall"].tolist() == [0.0, 1.0] and curve["fppi"].tolist() == [0.5, 1.0]


def test_ap_is_one_for_a_perfect_ranking_and_zero_without_detections():
    assert ap_point(np.array([0.9, 0.8, 0.1]), np.array([True, True, False]), n_signs=2) == 1.0
    assert ap_point(np.array([]), np.array([], dtype=bool), n_signs=3) == 0.0


def test_froc_rejects_bad_input():
    with pytest.raises(ValueError):
        froc_curve(np.array([0.5]), np.array([True, False]), 1, 1)
    with pytest.raises(ValueError):
        froc_curve(np.array([0.5]), np.array([True]), 0, 1)
    with pytest.raises(ValueError):
        froc_curve(np.array([0.5]), np.array([True]), 1, 0)
    with pytest.raises(ValueError):
        ap_point(np.array([0.5, 0.4]), np.array([True, True]), 1)
