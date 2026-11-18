"""detect_report on hand-made photos (docs/v3b-spec.md A13)."""

from pathlib import Path

import pytest

from findobject.data.gtsdb import RealPhoto, RealSign
from findobject.eval.detect_report import TAU_SYN, PhotoRecord, detect_report
from findobject.eval.photo_match import best_raw_score, match_photo
from findobject.eval.photo_truth import photo_truth
from findobject.inference.detection import Detection, detection_order

N = 200  # resamples; enough for the properties tested here


def _det(x, y, score, class_id):
    return Detection(x=x, y=y, score=score, class_id=class_id, class_prob=0.9, scale=1.0)


def _record(index, signs, kept, extra_raw=()):
    truth = photo_truth(RealPhoto(path=Path("x"), index=index,
                                  signs=tuple(RealSign(box=b, class_id=c) for b, c in signs)))
    kept = tuple(sorted(kept, key=detection_order))
    return PhotoRecord(index=index, truth=truth, kept=kept, match=match_photo(kept, truth),
                       best_raw=best_raw_score(list(kept) + list(extra_raw), truth))


def _hand_set():
    return [
        # A and B are stacked (centers 35 px apart, < 1.5 x 30); A found at 0.9, B located with the wrong class.
        _record(0, [((100, 100, 130, 130), 1), ((100, 135, 130, 165), 2)],
                [_det(115, 115, 0.9, 1), _det(115, 150, 0.6, 5)]),
        # C isolated, medium: found at 0.8, then a duplicate at 0.5.
        _record(1, [((300, 300, 340, 340), 3)], [_det(320, 320, 0.8, 3), _det(322, 322, 0.5, 3)]),
        # No signs: one background false positive.
        _record(2, [], [_det(50, 50, 0.7, 0)]),
        # D isolated, large (70 px): nothing kept, but a raw peak at 0.3 points at it.
        _record(3, [((500, 500, 570, 570), 4)], [], extra_raw=[_det(535, 535, 0.3, 4)]),
    ]


@pytest.fixture(scope="module")
def report():
    return detect_report(_hand_set(), {TAU_SYN: 0.55, "low": 0.25}, {(0, 0): True, (1, 0): False},
                         rng_seed=0, n_resamples=N)


def _value(entry, metric):
    return entry[metric]["k"], entry[metric]["n"]


def test_group_counts_at_tau_syn(report):
    g = report["at"][TAU_SYN]["groups"]
    assert _value(g["all"], "found") == (2, 4)
    assert _value(g["stacked"], "found") == (1, 2) and _value(g["isolated"], "found") == (1, 2)
    assert _value(g["medium"], "found") == (2, 3) and _value(g["large"], "found") == (0, 1)
    assert _value(g["ge64"], "found") == (0, 1)
    assert _value(g["medium_stacked"], "found") == (1, 2) and _value(g["medium_isolated"], "found") == (1, 1)
    assert _value(g["all"], "located") == (3, 4)
    assert _value(g["all"], "class_acc_given_located") == (2, 3)
    assert _value(g["all"], "located_before_suppression") == (3, 4)
    assert _value(g["windowed"], "found") == (2, 2) and _value(g["not_windowed"], "found") == (0, 2)
    assert "small" not in g  # empty groups are left out


def test_differences_and_false_positives(report):
    at = report["at"][TAU_SYN]
    assert at["differences"]["isolated_minus_stacked"]["value"] == pytest.approx(0.0)
    assert at["differences"]["medium_isolated_minus_stacked"]["value"] == pytest.approx(0.5)
    assert at["fppi"]["n_fp"] == 2 and at["fppi"]["value"] == 0.5
    assert at["fp_by_kind"] == {"wrong_class": 1, "background": 1}
    low = report["at"]["low"]
    assert low["fp_by_kind"] == {"wrong_class": 1, "background": 1, "duplicate": 1}
    assert low["fp_by_scale"] == {"1.0": 3}
    assert _value(low["groups"]["all"], "located_before_suppression") == (4, 4)


def test_windowed_comparison_only_at_tau_syn(report):
    w = report["at"][TAU_SYN]["windowed_vs_whole"]
    assert (w["n"], w["whole_found"], w["window_found"], w["only_whole"], w["only_window"]) == (2, 2, 1, 1, 0)
    assert w["mcnemar_p"] == 1.0
    assert "windowed_vs_whole" not in report["at"]["low"]


def _forty():
    """The hand set ten times over, renumbered: every group is in at least 10 photos."""
    return [PhotoRecord(index=i, truth=r.truth, kept=r.kept, match=r.match, best_raw=r.best_raw)
            for i, r in enumerate(r for _ in range(10) for r in _hand_set())]


def test_every_interval_contains_its_estimate():
    report = detect_report(_forty(), {TAU_SYN: 0.55, "low": 0.25}, {}, rng_seed=0, n_resamples=N)
    for at in report["at"].values():
        for entry in at["groups"].values():
            lo, hi = entry["found"]["ci"]
            assert lo <= entry["found"]["value"] <= hi
        for d in at["differences"].values():
            assert d["ci"][0] <= d["value"] <= d["ci"][1]
        lo, hi = at["fppi"]["ci"]
        assert lo <= at["fppi"]["value"] <= hi


def test_an_interval_a_resample_cannot_define_is_none_not_nan(report):
    # 4 photos: some resample draws only photo 2, which has no signs, so found's 0 / 0 is undefined.
    assert report["at"][TAU_SYN]["groups"]["ge64"]["found"]["ci"] is None
    assert report["at"][TAU_SYN]["fppi"]["ci"] is not None  # FPPI's denominator is never 0


def test_froc_and_ap_are_reported(report):
    assert report["froc"]["thresholds"] == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert report["froc"]["recall"][-1] == 0.5 and report["froc"]["fppi"][-1] == 0.75
    assert 0 < report["ap_point"] < 1
    assert (report["n_photos"], report["n_signs"], report["photo_indices"]) == (4, 4, [0, 1, 2, 3])


def test_differences_use_the_same_resamples_as_their_groups():
    # Each photo holds a stacked pair and an isolated sign with the same outcome, so stacked and isolated
    # found are equal in every resample, and the difference is 0 in all of them, only if both sides are
    # computed on the same draws.
    photos = []
    for i in range(30):
        hit = i % 3 != 0
        signs = [((100, 100, 130, 130), 1), ((100, 135, 130, 165), 1), ((600, 100, 630, 130), 2)]
        kept = [_det(115, 115, 0.9, 1), _det(115, 150, 0.8, 1), _det(615, 115, 0.7, 2)] if hit else []
        photos.append(_record(i, signs, kept))
    diff = detect_report(photos, {TAU_SYN: 0.5}, {}, rng_seed=3, n_resamples=N)["at"][TAU_SYN]["differences"]
    assert diff["isolated_minus_stacked"]["ci"] == pytest.approx((0.0, 0.0))


def test_same_seed_same_intervals_other_seed_other_intervals():
    a = detect_report(_forty(), {TAU_SYN: 0.55}, {}, rng_seed=0, n_resamples=N)
    b = detect_report(_forty(), {TAU_SYN: 0.55}, {}, rng_seed=0, n_resamples=N)
    c = detect_report(_forty(), {TAU_SYN: 0.55}, {}, rng_seed=1, n_resamples=N)
    assert a == b
    assert a["at"][TAU_SYN]["fppi"]["ci"] != c["at"][TAU_SYN]["fppi"]["ci"]


def test_detect_report_rejects_bad_input():
    with pytest.raises(ValueError, match="no photos"):
        detect_report([], {TAU_SYN: 0.5}, {}, 0, N)
    with pytest.raises(ValueError, match="windowed keys"):
        detect_report(_hand_set(), {TAU_SYN: 0.5}, {(9, 0): True}, 0, N)
