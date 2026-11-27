"""paste_sign and friends: the sign lands at the asked size and place, and nothing else changes
(docs/demo-spec.md A4)."""

import numpy as np
import pytest
from PIL import Image

from findobject.demo.paste import center_from_fractions, paste_sign, pasted_size

PHOTO = (400, 300)


def _photo():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 256, (PHOTO[1], PHOTO[0], 3), dtype=np.uint8))


@pytest.mark.parametrize("sign_size,long_side,center", [
    ((50, 50), 32, (200.0, 150.0)), ((60, 45), 17, (30.3, 290.0)), ((45, 60), 120, (100.0, 150.5)),
    ((31, 30), 1, (0.5, 0.5)),
])
def test_a4_size_center_and_untouched_pixels(sign_size, long_side, center):
    photo = _photo()
    before = np.asarray(photo).copy()
    p = paste_sign(photo, Image.new("RGB", sign_size, (255, 0, 0)), center, long_side)
    x1, y1, x2, y2 = p.box
    assert abs(max(x2 - x1, y2 - y1) - long_side) <= 1
    assert abs((x1 + x2) / 2 - center[0]) <= 1 and abs((y1 + y2) / 2 - center[1]) <= 1
    after = np.asarray(p.image)
    outside = np.ones(after.shape[:2], bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(after[outside], before[outside])
    assert (after[y1:y2, x1:x2] == (255, 0, 0)).all()
    assert np.array_equal(np.asarray(photo), before)  # input untouched


def test_shape_is_kept():
    assert pasted_size((60, 30), 40) == (40, 20)
    assert pasted_size((30, 60), 40) == (20, 40)
    assert pasted_size((100, 1), 10) == (10, 1)  # never a zero side


@pytest.mark.parametrize("center,long_side", [((5, 150), 32), ((200, 299), 32), ((200, 150), 301)])
def test_sign_off_the_photo_raises(center, long_side):
    with pytest.raises(ValueError):
        paste_sign(_photo(), Image.new("RGB", (40, 40)), center, long_side)


def test_long_side_below_one_raises():
    with pytest.raises(ValueError):
        pasted_size((40, 40), 0)


def test_fractions_reach_both_edges_and_keep_sign_inside():
    w, h = 40, 20
    assert center_from_fractions(PHOTO, (w, h), 0, 0) == (w / 2, h / 2)
    assert center_from_fractions(PHOTO, (w, h), 1, 1) == (PHOTO[0] - w / 2, PHOTO[1] - h / 2)
    for fx in np.linspace(0, 1, 11):
        c = center_from_fractions(PHOTO, (w, h), fx, 1 - fx)
        paste_sign(_photo(), Image.new("RGB", (w, h)), c, w)  # never raises


@pytest.mark.parametrize("size,fx,fy", [((401, 10), 0.5, 0.5), ((10, 10), -0.1, 0.5), ((10, 10), 0.5, 1.01)])
def test_bad_fractions_or_oversized_sign_raise(size, fx, fy):
    with pytest.raises(ValueError):
        center_from_fractions(PHOTO, size, fx, fy)
