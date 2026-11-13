"""The seam report (docs/v3a1-spec.md §2.3, A9)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from findobject.eval.seam_report import RelativeSize, box_long_side, relative_size, seam_report


def test_box_long_side():
    assert box_long_side(np.array([[0, 0, 10, 4], [5, 5, 8, 20]])).tolist() == [10, 15]
    with pytest.raises(ValueError):
        box_long_side(np.zeros((3, 3)))


def test_relative_size_boundaries():
    # differences −5, −4, 0, 4, 5: only beyond ±4 px leaves "similar"
    g = relative_size(np.array([30] * 5), np.array([25, 26, 30, 34, 35]))
    assert g.tolist() == [RelativeSize.HYBRID_SMALLER, RelativeSize.SIMILAR, RelativeSize.SIMILAR,
                          RelativeSize.SIMILAR, RelativeSize.HYBRID_LARGER]


def _pairs():
    # 6 pairs, two of each relative size, on 2 photos that each hold one pair of every size (so no
    # resample can leave a size group empty).
    real = np.float32([0.5, 0.6, 0.2, 0.9, 0.4, 0.7])
    hybrid = np.float32([0.9, 0.7, 0.5, 0.8, 0.4, 0.95])
    real_long = np.array([20, 20, 30, 30, 40, 40])
    hybrid_long = np.array([10, 12, 30, 33, 48, 46])  # smaller, smaller, similar, similar, larger, larger
    photo = np.array([0, 1, 0, 1, 0, 1])
    return real, hybrid, real_long, hybrid_long, photo


def test_seam_report_by_hand():
    out = seam_report(*_pairs(), np.float32(0.5))
    g = out["median_gap"]
    # gaps: 0.4, 0.1, 0.3, −0.1, 0, 0.25 -> sorted −0.1, 0, 0.1, 0.25, 0.3, 0.4 -> median 0.175
    assert g["all"]["value"] == pytest.approx(0.175, abs=1e-6) and g["all"]["n"] == 6
    assert g["hybrid_smaller"]["value"] == pytest.approx(0.25, abs=1e-6)  # (0.4, 0.1)
    assert g["similar"]["value"] == pytest.approx(0.1, abs=1e-6)  # (0.3, −0.1)
    assert g["hybrid_larger"]["value"] == pytest.approx(0.125, abs=1e-6)  # (0, 0.25)
    assert out["share_hybrid_higher"]["value"] == pytest.approx(4 / 6)  # ties (0.4 = 0.4) do not count
    assert out["median_real"]["value"] == pytest.approx(0.55, abs=1e-6)
    assert out["median_hybrid"]["value"] == pytest.approx(0.75, abs=1e-6)
    for d in [*g.values(), out["share_hybrid_higher"], out["median_real"], out["median_hybrid"]]:
        assert d["lo"] <= d["value"] <= d["hi"]
    assert out["said_yes"]["real_yes"]["k"] == 4  # 0.5, 0.6, 0.9, 0.7


def test_seam_report_photo_clustered():
    # Every pair on one photo: every resample draws that photo, so every interval collapses to the value.
    real, hybrid, real_long, hybrid_long, _ = _pairs()
    out = seam_report(real, hybrid, real_long, hybrid_long, np.zeros(6, int), np.float32(0.5))
    a = out["median_gap"]["all"]
    assert a["lo"] == a["hi"] == pytest.approx(a["value"])


def test_seam_report_rejects_bad_input():
    real, hybrid, real_long, hybrid_long, photo = _pairs()
    with pytest.raises(ValueError):
        seam_report(real, hybrid[:5], real_long, hybrid_long, photo, np.float32(0.5))
    with pytest.raises(ValueError):  # no "hybrid larger" pair
        seam_report(real, hybrid, real_long, real_long - 10, photo, np.float32(0.5))
    with pytest.raises(ValueError):  # one photo per size group: some resample misses a group entirely
        seam_report(real, hybrid, real_long, hybrid_long, np.array([0, 0, 1, 1, 2, 2]), np.float32(0.5))


# --- A9: reproduce the §2.3 table from the stored v3a and v1.5b seed 0 results ---

V3A_S0, V15B_S0 = "peak_empty-20261109-140021", "peak_fresh_aug-20261108-024220"
# run -> (all, hybrid_larger, similar, hybrid_smaller), as in docs/v3a1-spec.md §2.3
EXPECTED = {V3A_S0: (0.119, 0.057, 0.088, 0.158), V15B_S0: (0.111, 0.040, 0.088, 0.155)}


@pytest.mark.parametrize("run", EXPECTED)
def test_a9_reproduces_the_spec_table(run):
    result_path = Path("outputs/eval") / run / "presence_real_dev.json"
    if not result_path.exists():
        pytest.skip("stored v3a/v1.5b results are local-only (outputs/ is not in git)")
    spec = importlib.util.spec_from_file_location("seam_report_cli", Path(__file__).parents[1] / "scripts/seam_report.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    _boxes = cli._boxes

    r = json.loads(result_path.read_text())
    ps = r["per_scene"]
    out = seam_report(np.float32(ps["positives"]["presence"]), np.float32(ps["hybrid"]["presence"]),
                      box_long_side(_boxes(r["data"]["real_dev"])), box_long_side(_boxes(r["data"]["hybrid_dev"])),
                      np.asarray(ps["positives"]["photo_index"]), np.float32(r["tau"]))
    g = out["median_gap"]
    got = tuple(g[k]["value"] for k in ("all", "hybrid_larger", "similar", "hybrid_smaller"))
    assert got == pytest.approx(EXPECTED[run], abs=5e-4)
    assert (g["hybrid_larger"]["n"], g["similar"]["n"], g["hybrid_smaller"]["n"]) == (148, 139, 358)
    assert g["all"]["value"] == pytest.approx(r["report"]["seam"]["median_presence_diff_hybrid_minus_real"])
