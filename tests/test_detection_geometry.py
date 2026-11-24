"""v3b's peak finding, cell-to-photo mapping, Detection's checks and suppression (docs/v3b-spec.md A3, A4, A5, A11)."""

import math
import random

import pytest
import torch

from findobject.inference.detection import Detection, detection_order
from findobject.inference.peaks import MIN_PEAK_SCORE, cell_to_photo, find_peaks
from findobject.inference.suppress import suppress


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _bump(h: int, w: int, centers: list[tuple[int, int]], peak: float = 4.0) -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    out = torch.full((h, w), -10.0)
    for r, c in centers:
        out = torch.maximum(out, peak - ((rows - r) ** 2 + (cols - c) ** 2) / 2.0)
    return out


def _det(x, y, score, scale=1.0, class_id=0):
    return Detection(x=x, y=y, score=score, class_id=class_id, class_prob=0.9, scale=scale)


# --- A3: find_peaks ---

def test_one_bump_gives_one_peak_at_its_center():
    assert find_peaks(_bump(20, 30, [(7, 11)])).tolist() == [[7, 11]]


def test_two_bumps_four_cells_apart_give_two_peaks():
    assert find_peaks(_bump(20, 30, [(10, 5), (10, 9)])).tolist() == [[10, 5], [10, 9]]


def test_flat_two_by_two_top_gives_four_peaks_that_suppression_merges():
    logits = torch.full((10, 10), -10.0)
    logits[4:6, 4:6] = 2.0
    cells = find_peaks(logits)
    assert cells.tolist() == [[4, 4], [4, 5], [5, 4], [5, 5]]
    xy = cell_to_photo(cells, 1.0)
    dets = [_det(float(x), float(y), 0.88) for x, y in xy.tolist()]
    assert len(suppress(dets)) == 1


def test_a_peak_below_min_score_is_dropped_and_one_at_it_is_kept():
    logits = torch.full((9, 9), -10.0)
    logits[2, 2] = _logit(0.049)
    logits[6, 6] = _logit(MIN_PEAK_SCORE)
    assert find_peaks(logits).tolist() == [[6, 6]]


def test_logits_that_tie_after_float32_sigmoid_still_have_one_maximum():
    logits = torch.full((5, 5), -10.0)
    logits[2, 2], logits[2, 3] = 20.0, 19.0
    assert torch.sigmoid(logits[2, 2]) == torch.sigmoid(logits[2, 3])  # the tie the rule avoids
    assert find_peaks(logits).tolist() == [[2, 2]]


def test_find_peaks_rejects_bad_input():
    with pytest.raises(ValueError):
        find_peaks(torch.zeros(1, 5, 5))
    with pytest.raises(ValueError):
        find_peaks(torch.zeros(5, 5), min_score=0.0)
    with pytest.raises(ValueError):
        find_peaks(torch.tensor([[float("nan")]]))


# --- A5: cell -> photo ---

@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
def test_single_peak_maps_to_the_cell_center_in_native_pixels(scale):
    logits = torch.full((50, 85), -10.0)
    logits[13, 41] = 3.0
    (x, y), = cell_to_photo(find_peaks(logits), scale).tolist()
    assert (x, y) == ((41 + 0.5) * 4 / scale, (13 + 0.5) * 4 / scale)


def test_cell_to_photo_rejects_bad_scale():
    with pytest.raises(ValueError):
        cell_to_photo(torch.zeros(1, 2, dtype=torch.long), 0.0)


# --- A11: Detection's illegal states ---

@pytest.mark.parametrize("field, value", [("score", 1.1), ("score", -0.1), ("class_prob", 2.0),
                                          ("class_id", 43), ("class_id", -1), ("scale", 0.0), ("scale", 1.5),
                                          ("x", float("nan"))])
def test_detection_rejects_illegal_values(field, value):
    kwargs = dict(x=1.0, y=2.0, score=0.5, class_id=3, class_prob=0.5, scale=1.0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        Detection(**kwargs)


def test_order_is_score_then_finer_scale_then_y_then_x():
    a, b = _det(10, 10, 0.7, scale=1.0), _det(10, 10, 0.7, scale=0.25)
    c, d = _det(5, 20, 0.7), _det(9, 20, 0.7)
    assert sorted([b, d, c, a, _det(0, 0, 0.9)], key=detection_order) == [_det(0, 0, 0.9), a, c, d, b]


# --- A4: suppress ---

def test_scale1_detections_16px_apart_both_survive_8px_apart_only_the_higher():
    assert len(suppress([_det(100, 100, 0.9), _det(116, 100, 0.8)])) == 2
    assert suppress([_det(100, 100, 0.9), _det(108, 100, 0.8)]) == [_det(100, 100, 0.9)]


def test_the_radius_is_strict():
    assert len(suppress([_det(100, 100, 0.9), _det(112, 100, 0.8)])) == 2  # exactly 12 px: kept
    assert len(suppress([_det(100, 100, 0.9), _det(111.9, 111.9, 0.8)])) == 1  # Chebyshev, not Euclidean


def test_a_coarse_detection_suppresses_with_its_own_larger_radius():
    coarse = _det(200, 200, 0.9, scale=0.25)  # radius 48 px
    assert suppress([coarse, _det(220, 200, 0.8)]) == [coarse]
    assert len(suppress([coarse, _det(260, 200, 0.8)])) == 2


def test_on_equal_scores_the_finer_scale_is_kept():
    fine, coarse = _det(100, 100, 0.8, scale=1.0), _det(104, 104, 0.8, scale=0.25)
    assert suppress([coarse, fine]) == [fine]


def test_result_does_not_depend_on_input_order():
    rng = random.Random(0)
    dets = [_det(rng.uniform(0, 300), rng.uniform(0, 200), round(rng.uniform(0.05, 1), 2),
                 scale=rng.choice([1.0, 0.5, 0.25])) for _ in range(200)]
    expected = suppress(dets)
    for _ in range(5):
        rng.shuffle(dets)
        assert suppress(dets) == expected
    assert expected == sorted(expected, key=detection_order)


def test_suppress_of_nothing_is_nothing():
    assert suppress([]) == []
