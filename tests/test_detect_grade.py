"""Grading W1-W6 from report dicts: the frozen bars, met or missed at their edges."""

import copy

import pytest

from findobject.eval.detect_grade import grade_dev, grade_test
from findobject.eval.detect_report import TAU_SYN, TAU_W


def _at(found=0.9, ge64=0.9, fppi=0.8, gap=0.1, gap_ci=(0.02, 0.2), windowed=None):
    groups = {"all": {"found": {"value": found}}, "ge64": {"found": {"value": ge64, "n": 139}},
              "medium_isolated": {"found": {"value": 0.94}}, "medium_stacked": {"found": {"value": 0.94 - gap}}}
    at = {"groups": groups, "fppi": {"value": fppi, "ci": (0.5, 1.1)},
          "differences": {"medium_isolated_minus_stacked": {"value": gap, "ci": gap_ci}}}
    if windowed:
        at["windowed_vs_whole"] = windowed
    return at


def _report(tau_w=0.7, photos=(0, 1), **at_syn):
    windowed = {"n": 645, "whole_found": 600, "window_found": 609, "mcnemar_p": 0.2}
    return {"photo_indices": list(photos), "thresholds": {TAU_SYN: 0.5, TAU_W: tau_w},
            "at": {TAU_SYN: _at(windowed=windowed, **at_syn), TAU_W: _at(found=at_syn.get("found", 0.85))}}


def _dev(**seed0):
    return [_report(**seed0), _report(), _report()]


def test_everything_passing():
    g = grade_dev(_dev(), _report(ge64=0.3))
    assert all(g[w]["pass"] for w in ("W1", "W2", "W3", "W4", "W5")), g


def test_w1_edges():
    dev = _dev()
    w = dev[0]["at"][TAU_SYN]["windowed_vs_whole"]
    w["whole_found"] = 609 - 19  # 2.95 points below
    assert grade_dev(dev, _report(ge64=0.3))["W1"]["pass"]
    w["whole_found"] = 609 - 20  # 3.10 points below
    assert not grade_dev(dev, _report(ge64=0.3))["W1"]["pass"]


@pytest.mark.parametrize("seed0, others, ok", [(dict(gap=0.05), 0.1, True), (dict(gap=0.049), 0.1, False),
                                               (dict(gap=0.1, gap_ci=(-0.01, 0.2)), 0.1, False),
                                               (dict(gap=0.1), -0.001, False)])
def test_w2_needs_the_gap_its_interval_and_all_three_seeds(seed0, others, ok):
    dev = [_report(**seed0), _report(gap=others), _report()]
    assert grade_dev(dev, _report(ge64=0.3))["W2"]["pass"] is ok


def test_w2_refuses_an_undefined_interval():
    with pytest.raises(ValueError, match="W2"):
        grade_dev(_dev(gap_ci=None), _report(ge64=0.3))


@pytest.mark.parametrize("pyramid, scale1, ok", [(0.85, 0.60, True), (0.849, 0.3, False), (0.9, 0.601, False)])
def test_w3_edges(pyramid, scale1, ok):
    assert grade_dev(_dev(ge64=pyramid), _report(ge64=scale1))["W3"]["pass"] is ok


@pytest.mark.parametrize("fppi, ok", [(0.4, True), (1.6, True), (0.399, False), (1.601, False)])
def test_w4_is_a_two_sided_band(fppi, ok):
    assert grade_dev(_dev(fppi=fppi), _report(ge64=0.3))["W4"]["pass"] is ok


def test_w5_uses_the_three_seed_mean():
    dev = _dev()
    for r, f in zip(dev, (0.70, 0.86, 0.85)):  # seed 0 alone misses; the mean is 0.803
        r["at"][TAU_W]["groups"]["all"]["found"]["value"] = f
    assert grade_dev(dev, _report(ge64=0.3))["W5"]["pass"]
    dev[0]["at"][TAU_W]["groups"]["all"]["found"]["value"] = 0.68  # mean 0.797
    assert not grade_dev(dev, _report(ge64=0.3))["W5"]["pass"]


def test_grade_dev_refuses_mismatched_runs():
    with pytest.raises(ValueError):
        grade_dev(_dev()[:2], _report())
    with pytest.raises(ValueError, match="different photos"):
        grade_dev([_report(), _report(photos=(0, 2)), _report()], _report())
    with pytest.raises(ValueError, match="scale-1"):
        grade_dev(_dev(), _report(photos=(5,)))


def _with(reports, fppi, found):
    out = copy.deepcopy(reports)
    for r in out:
        r["at"][TAU_W]["fppi"]["value"] = fppi
        r["at"][TAU_W]["groups"]["all"]["found"]["value"] = found
    return out


def test_w6_is_one_sided():
    dev = _with(_dev(), 0.5, 0.85)
    assert grade_test(dev, _with(dev, 0.75, 0.80))["W6"]["pass"]
    assert grade_test(dev, _with(dev, 0.2, 0.99))["W6"]["pass"]  # better than dev is fine
    assert not grade_test(dev, _with(dev, 0.76, 0.85))["W6"]["pass"]
    assert not grade_test(dev, _with(dev, 0.5, 0.799))["W6"]["pass"]


def test_w6_refuses_a_test_tau_w_that_is_not_dev_tau_w():
    with pytest.raises(ValueError, match="τ_W"):
        grade_test(_dev(), [_report(tau_w=0.71), _report(), _report()])
