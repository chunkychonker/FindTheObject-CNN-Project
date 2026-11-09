import numpy as np
import pytest
from PIL import Image

from findobject.data.real_windows import Skip, WindowPlan, cut_window, plan_window, scaled_size
from findobject.data.synth import SynthConfig

CFG = SynthConfig()  # scene 128, heatmap 32, sign_max 48


def _random_layout(rs: np.random.Generator) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """A photo size and 1-4 boxes of 8-150 px, possibly overlapping, all inside the photo."""
    w, h = int(rs.integers(400, 1400)), int(rs.integers(400, 900))
    boxes = []
    for _ in range(int(rs.integers(1, 5))):
        bw, bh = int(rs.integers(8, 150)), int(rs.integers(8, 150))
        x1, y1 = int(rs.integers(0, w - bw)), int(rs.integers(0, h - bh))
        boxes.append((x1, y1, x1 + bw, y1 + bh))
    return (w, h), boxes


def _scaled(box, s):
    import math
    return math.floor(box[0] * s), math.floor(box[1] * s), math.ceil(box[2] * s), math.ceil(box[3] * s)


def test_plans_respect_every_geometric_rule():
    rs = np.random.default_rng(0)
    planned = skipped = 0
    for trial in range(400):
        size, boxes = _random_layout(rs)
        for t, (x1, y1, x2, y2) in enumerate(boxes):
            plan = plan_window(size, boxes, t, np.random.default_rng(trial), CFG)
            if plan is Skip.OTHER_SIGN_IN_VIEW:
                skipped += 1
                continue
            planned += 1
            long_side = max(x2 - x1, y2 - y1)
            # A7: never enlarge; native size exactly when it already fits; scaled sign <= sign_max (+1 rounding)
            assert plan.scale <= 1 and (plan.scale == 1) == (long_side <= CFG.sign_max)
            bx1, by1, bx2, by2 = plan.box
            assert max(bx2 - bx1, by2 - by1) <= CFG.sign_max + 1
            # A5: target inside window, window inside scaled photo
            assert 0 <= bx1 < bx2 <= CFG.scene_size and 0 <= by1 < by2 <= CFG.scene_size
            w, h = scaled_size(size, plan.scale)
            assert 0 <= plan.x0 <= w - CFG.scene_size and 0 <= plan.y0 <= h - CFG.scene_size
            # A6: no other box touches the window
            win = (plan.x0, plan.y0, plan.x0 + CFG.scene_size, plan.y0 + CFG.scene_size)
            for j, other in enumerate(boxes):
                if j != t:
                    o = _scaled(other, plan.scale)
                    assert not (win[0] < o[2] and o[0] < win[2] and win[1] < o[3] and o[1] < win[3])
    assert planned > 100 and skipped > 0  # both paths actually exercised


def test_same_seed_same_plan():  # A9
    size, boxes = (1360, 800), [(500, 300, 540, 340), (900, 100, 980, 180)]
    a = plan_window(size, boxes, 0, np.random.default_rng(5), CFG)
    assert a == plan_window(size, boxes, 0, np.random.default_rng(5), CFG)
    assert a != plan_window(size, boxes, 0, np.random.default_rng(6), CFG)


def test_signs_close_on_both_sides_force_a_skip():  # A6
    boxes = [(100, 100, 130, 130), (60, 100, 99, 130), (131, 100, 160, 130)]
    assert plan_window((1360, 800), boxes, 0, np.random.default_rng(0), CFG) is Skip.OTHER_SIGN_IN_VIEW


def test_bad_arguments_raise():
    with pytest.raises(IndexError):
        plan_window((1360, 800), [(0, 0, 10, 10)], 1, np.random.default_rng(0), CFG)
    with pytest.raises(ValueError):
        plan_window((1360, 800), [(5, 5, 5, 10)], 0, np.random.default_rng(0), CFG)
    with pytest.raises(ValueError):
        plan_window((100, 100), [(0, 0, 10, 10)], 0, np.random.default_rng(0), CFG)
    with pytest.raises(ValueError):
        cut_window(Image.new("RGB", (300, 300)), WindowPlan(1.0, 250, 0, (0, 0, 10, 10)), CFG)


def _marked_photo(box) -> Image.Image:
    """Black photo with the box filled white: every non-black window pixel is sign."""
    a = np.zeros((800, 1360, 3), np.uint8)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 255
    return Image.fromarray(a)


def test_native_window_is_an_exact_crop_with_the_sign_at_plan_box():
    box = (700, 400, 740, 436)  # 40 px: fits, so scale 1
    photo = _marked_photo(box)
    plan = plan_window(photo.size, [box], 0, np.random.default_rng(1), CFG)
    scene = cut_window(photo, plan, CFG)
    assert plan.scale == 1 and scene.image.shape == (128, 128, 3) and scene.image.dtype == np.uint8
    assert np.array_equal(scene.image, np.asarray(photo)[plan.y0:plan.y0 + 128, plan.x0:plan.x0 + 128])
    ys, xs = np.nonzero(scene.image.max(axis=2))
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == plan.box


def test_downscaled_sign_lands_on_plan_box_within_a_pixel():
    box = (600, 300, 720, 400)  # 120 px: scaled to 48
    plan = plan_window((1360, 800), [box], 0, np.random.default_rng(2), CFG)
    scene = cut_window(_marked_photo(box), plan, CFG)
    ys, xs = np.nonzero(scene.image.max(axis=2) > 127)
    got = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    assert plan.scale == pytest.approx(48 / 120)
    assert np.abs(got - np.array(plan.box)).max() <= 1


def test_heatmap_peaks_at_box_center():  # A8
    rs = np.random.default_rng(3)
    photo = Image.fromarray(rs.integers(0, 256, (800, 1360, 3), dtype=np.uint8))
    stride = CFG.scene_size // CFG.heatmap_size
    for seed in range(30):
        _, boxes = _random_layout(np.random.default_rng(100 + seed))
        plan = plan_window(photo.size, boxes[:1], 0, np.random.default_rng(seed), CFG)
        scene = cut_window(photo, plan, CFG)
        x1, y1, x2, y2 = scene.box
        r, c = np.unravel_index(scene.heatmap.argmax(), scene.heatmap.shape)
        assert abs((c + 0.5) * stride - (x1 + x2) / 2) <= stride / 2 + 1e-6
        assert abs((r + 0.5) * stride - (y1 + y2) / 2) <= stride / 2 + 1e-6


def test_downscaled_sign_at_the_photo_edge_still_plans():
    # 1360 * 48/101 = 646.3 -> width 646, but ceil(1360 * 48/101) = 647: the box must be clipped to the photo
    box = (1259, 699, 1360, 800)
    plan = plan_window((1360, 800), [box], 0, np.random.default_rng(0), CFG)
    w, h = scaled_size((1360, 800), plan.scale)
    assert plan.x0 + plan.box[2] <= w and plan.y0 + plan.box[3] <= h
