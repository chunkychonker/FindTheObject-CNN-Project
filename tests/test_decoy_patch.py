"""Choosing a training decoy's patch (docs/v3a1-spec.md §2.1, A5, A6)."""

from collections import Counter

import numpy as np
import pytest

from findobject.data.decoy_patch import draw_patch, eligible_patch_sources

NEED = (192, 183)
# Background 1 is too small (as 8 real train backgrounds are); 4 is too short.
SIZES = [(300, 300), (200, 145), (640, 480), (192, 183), (500, 182)]


def test_eligible_keeps_only_backgrounds_that_fit_every_patch():
    src = eligible_patch_sources(SIZES, NEED)
    assert src.index == (0, 2, 3) and src.sizes == ((300, 300), (640, 480), (192, 183)) and src.need == NEED


def test_eligible_needs_two():
    with pytest.raises(ValueError):
        eligible_patch_sources([(300, 300), (100, 100)], NEED)


@pytest.mark.parametrize("scene_bg", [0, 1, 2])  # eligible, ineligible, eligible
def test_draw_never_picks_own_or_small_background_and_stays_inside(scene_bg):
    src = eligible_patch_sources(SIZES, NEED)
    picks = Counter()
    for k in range(3000):
        size = (int(1 + k % NEED[0]), int(1 + k % NEED[1]))
        chosen, (x, y) = draw_patch(src, scene_bg, size, np.random.default_rng(k))
        w, h = SIZES[chosen]
        assert chosen in src.index and chosen != scene_bg
        assert 0 <= x <= w - size[0] and 0 <= y <= h - size[1]
        picks[chosen] += 1
    allowed = set(src.index) - {scene_bg}
    assert set(picks) == allowed
    for c in allowed:  # uniform: each near 3000 / |allowed|
        assert abs(picks[c] - 3000 / len(allowed)) < 0.1 * 3000


def test_largest_patch_from_an_exactly_fitting_source_has_one_corner():
    src = eligible_patch_sources([(192, 183), (192, 183)], NEED)
    assert draw_patch(src, 0, NEED, np.random.default_rng(0)) == (1, (0, 0))


@pytest.mark.parametrize("size", [(193, 10), (10, 184), (0, 5)])
def test_draw_rejects_sizes_outside_need(size):
    with pytest.raises(ValueError):
        draw_patch(eligible_patch_sources(SIZES, NEED), 0, size, np.random.default_rng(0))
