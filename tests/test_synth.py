import numpy as np
import pytest
from PIL import Image

from findobject.data.synth import SynthConfig, gaussian_heatmap, make_scene


# --- gaussian_heatmap (your TODO) ---

def test_heatmap_shape_dtype_and_range():
    hm = gaussian_heatmap((10.0, 20.0), size=32, sigma=1.0)
    assert hm.shape == (32, 32) and hm.dtype == np.float32
    assert hm.min() >= 0 and hm.max() <= 1.0


def test_heatmap_peak_is_at_center_row_y_col_x():
    hm = gaussian_heatmap((10.0, 20.0), size=32, sigma=1.0)  # x=10 -> column, y=20 -> row
    assert np.unravel_index(hm.argmax(), hm.shape) == (20, 10)
    assert hm[20, 10] == pytest.approx(1.0)


def test_heatmap_one_sigma_away():
    hm = gaussian_heatmap((5.0, 5.0), size=16, sigma=2.0)
    assert hm[5, 7] == pytest.approx(np.exp(-0.5), rel=1e-5)


def test_heatmap_fractional_center_splits_evenly():
    hm = gaussian_heatmap((4.5, 4.0), size=16, sigma=1.0)
    assert hm[4, 4] == pytest.approx(hm[4, 5])


def test_heatmap_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gaussian_heatmap((1.0, 1.0), size=8, sigma=0.0)


# --- make_scene (needs gaussian_heatmap) ---

def _inputs():
    rs = np.random.default_rng(123)
    bg = Image.fromarray(rs.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    sign = Image.fromarray(np.full((40, 38, 3), 200, np.uint8))
    return sign, bg


def test_scene_shapes_box_and_heatmap_peak():
    cfg = SynthConfig()
    sign, bg = _inputs()
    for seed in range(20):
        scene = make_scene(sign, bg, np.random.default_rng(seed), cfg)
        assert scene.image.shape == (128, 128, 3) and scene.image.dtype == np.uint8
        x1, y1, x2, y2 = scene.box
        assert 0 <= x1 < x2 <= 128 and 0 <= y1 < y2 <= 128
        assert cfg.sign_min <= max(x2 - x1, y2 - y1) <= cfg.sign_max
        r, c = np.unravel_index(scene.heatmap.argmax(), scene.heatmap.shape)
        assert x1 <= (c + 0.5) * 4 <= x2 and y1 <= (r + 0.5) * 4 <= y2  # peak cell lies on the sign


def test_scene_is_deterministic_for_a_seed():
    sign, bg = _inputs()
    a = make_scene(sign, bg, np.random.default_rng(7), SynthConfig())
    b = make_scene(sign, bg, np.random.default_rng(7), SynthConfig())
    assert a.box == b.box and np.array_equal(a.image, b.image)


def test_config_rejects_sign_bigger_than_scene():
    with pytest.raises(ValueError):
        SynthConfig(sign_max=200)
