"""fit_photo and to_original: any upload becomes a size detect() accepts, and points map back inside the
original (docs/demo-spec.md A2, A3)."""

import numpy as np
import pytest
from PIL import Image

from findobject.demo.fit import FIT_MAX, FIT_MULTIPLE, fit_photo, to_original

SIZES = [(16, 16), (1, 1), (6000, 6000), (6000, 10), (10, 6000), (1000, 777), (1361, 800), (1360, 801)]


@pytest.mark.parametrize("size", SIZES)
def test_a3_fitted_size_is_valid_and_never_enlarged(size):
    f = fit_photo(Image.new("RGB", size, (90, 120, 30)))
    pw, ph = f.image.size
    cw, ch = f.content_size
    assert pw % FIT_MULTIPLE == 0 and ph % FIT_MULTIPLE == 0
    assert cw <= FIT_MAX[0] and ch <= FIT_MAX[1]  # content fits the graded photo size
    assert cw <= size[0] and ch <= size[1]  # never enlarged
    assert cw <= pw < cw + FIT_MULTIPLE and ch <= ph < ch + FIT_MULTIPLE  # padding is less than one multiple
    assert f.original_size == size
    # Shape kept: each side is within rounding of the same factor.
    factor = min(1.0, FIT_MAX[0] / size[0], FIT_MAX[1] / size[1])
    assert abs(cw - size[0] * factor) <= 1 and abs(ch - size[1] * factor) <= 1


def test_a3_graded_size_photo_comes_back_unchanged():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (FIT_MAX[1], FIT_MAX[0], 3), dtype=np.uint8))
    f = fit_photo(img)
    assert f.image.size == FIT_MAX and f.content_size == FIT_MAX
    assert np.array_equal(np.asarray(f.image), np.asarray(img))


def test_padding_is_black_and_right_bottom():
    f = fit_photo(Image.new("RGB", (1000, 777), (200, 200, 200)))
    a = np.asarray(f.image)
    cw, ch = f.content_size
    assert (a[:ch, :cw] == 200).all()
    assert (a[ch:] == 0).all() and (a[:, cw:] == 0).all()


def test_non_rgb_input_is_converted():
    assert fit_photo(Image.new("L", (40, 40), 7)).image.mode == "RGB"
    assert fit_photo(Image.new("RGBA", (40, 40))).image.mode == "RGB"


@pytest.mark.parametrize("size", SIZES)
def test_a2_points_on_content_map_inside_original(size):
    f = fit_photo(Image.new("RGB", size))
    cw, ch = f.content_size
    for x, y in [(0, 0), (cw - 1e-6, ch - 1e-6), (cw / 2, ch / 2)]:
        assert f.in_content(x, y)
        ox, oy = to_original(x, y, f)
        assert 0 <= ox < size[0] and 0 <= oy < size[1]
    assert not f.in_content(cw, 0) and not f.in_content(0, ch)


def test_to_original_undoes_the_shrink():
    f = fit_photo(Image.new("RGB", (2720, 1600)))  # exactly half
    assert f.content_size == FIT_MAX
    assert to_original(100, 50, f) == (200, 100)
