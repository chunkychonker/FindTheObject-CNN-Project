"""The stage drawings keep the photo's size and coordinates (docs/demo-spec.md §2)."""

import numpy as np
import pytest
from PIL import Image

from findobject.demo.draw import GAP, draw_answer, draw_heatmap, draw_points, draw_scales
from findobject.demo.trace import ScaleView
from findobject.inference.detection import Detection

PHOTO = Image.new("RGB", (320, 160), (40, 80, 120))
DET = Detection(x=100.0, y=50.0, score=0.9, class_id=14, class_prob=0.99, scale=0.5)


def test_scales_sit_side_by_side_at_true_size():
    out = draw_scales(PHOTO)
    assert out.width == 320 + 160 + 80 + 2 * GAP and out.height > 160


def test_zero_heatmap_leaves_the_photo_unchanged_and_hot_cells_change_it():
    cold = ScaleView(0.5, np.zeros((20, 40), np.float32))  # 320x160 photo at ½ -> 160x80 px -> 40x20 cells (stride 4)
    assert np.array_equal(np.asarray(draw_heatmap(PHOTO, cold)), np.asarray(PHOTO))
    hot = np.zeros((20, 40), np.float32)
    hot[2, 3] = 1.0  # covers photo px x 24-31, y 16-23 at 8 px per cell
    out = np.asarray(draw_heatmap(PHOTO, ScaleView(0.5, hot))).astype(int)
    diff = np.abs(out - np.asarray(PHOTO)).sum(axis=2) > 0
    ys, xs = np.nonzero(diff)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (24, 31, 16, 23)


def test_heatmap_that_does_not_cover_the_photo_names_both_shapes():
    short = ScaleView(0.5, np.zeros((10, 20), np.float32))  # the ¼-scale shape: covers only 160x80 of 320x160
    with pytest.raises(ValueError, match=r"320x160.*\(10, 20\).*160x80"):
        draw_heatmap(PHOTO, short)


def test_heatmap_larger_than_the_photo_is_cropped_to_it():
    padded = ScaleView(0.5, np.zeros((21, 42), np.float32))  # as for fitted.content inside a padded photo
    assert draw_heatmap(PHOTO, padded).size == PHOTO.size


def test_point_and_answer_drawings_keep_the_size():
    for img in (draw_points(PHOTO, [DET]), draw_points(PHOTO, [DET], show_suppression=True), draw_answer(PHOTO, [DET])):
        assert img.size == PHOTO.size and img.mode == "RGB"
        assert not np.array_equal(np.asarray(img), np.asarray(PHOTO))
    assert np.array_equal(np.asarray(draw_points(PHOTO, [])), np.asarray(PHOTO))
