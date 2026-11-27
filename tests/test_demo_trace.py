"""trace() and the views built on it: the demo's answer is exactly the graded detector's, any upload runs,
and re-rendering at a new threshold never touches the model (docs/demo-spec.md A1, A2, A5)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from findobject.data.gtsdb import load_photo, read_gtsdb
from findobject.demo.trace import above, trace
from findobject.demo.views import PASTED_GT_INDEX, fool_view, place_sign, watch_view
from findobject.eval.photo_truth import photo_truth
from findobject.inference.detect import detect
from findobject.inference.detection import PYRAMID_SCALES
from findobject.models.baseline import STRIDE
from findobject.models.peak import PeakCNN
from findobject.training.checkpoint import load_model

SEED0 = Path("outputs/checkpoints/peak_decoy-20261112-011953/best.pt")
GTSDB = Path("data/raw/gtsdb/FullIJCNN2013")
TAU_W = 0.42658016085624695  # seed 0's τ_W (outputs/eval/peak_decoy-20261112-011953/detect_dev.json)
needs_real = pytest.mark.skipif(not (SEED0.is_file() and GTSDB.is_dir()), reason="needs seed 0 checkpoint + GTSDB")


@pytest.fixture(scope="module")
def seed0():
    return load_model(SEED0, torch.device("cpu"))


@pytest.fixture(scope="module")
def dev_photos():
    return {p.index: p for p in read_gtsdb(GTSDB)}


def _set_bias(model, value):
    """Push every heatmap logit up so a random PeakCNN has many peaks (an untrained one scores ~0.01 and has none)."""
    head = model.heatmap_head
    conv = head if isinstance(head, torch.nn.Conv2d) else [m for m in head.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    with torch.no_grad():
        conv.bias.fill_(value)
    return model


def _busy_model():
    torch.manual_seed(0)
    return _set_bias(PeakCNN().eval(), 3.0)


@needs_real
@pytest.mark.parametrize("index", [206, 5, 552, 54, 235])
def test_a1_answer_equals_detect_on_dev_photos(seed0, dev_photos, index):
    photo = load_photo(dev_photos[index])
    want = above(detect(seed0, photo).kept, TAU_W)
    tr = trace(seed0, photo)
    assert above(tr.result.kept, TAU_W) == want
    v = watch_view(tr, photo_truth(dev_photos[index]), TAU_W)
    assert len(v.rows) == len(want)
    assert [(r[4], r[5]) for r in v.rows] == [(round(d.x), round(d.y)) for d in want]  # no shrink -> same px
    if index == 206:
        assert len(want) > 0  # guards against a trivially empty comparison


@needs_real
def test_heatmaps_are_the_ones_detect_read(seed0, dev_photos):
    """Every raw detection's score is the shown heatmap's value at its cell."""
    tr = trace(seed0, load_photo(dev_photos[206]))
    by_scale = {v.scale: v.heatmap for v in tr.views}
    assert len(tr.result.raw) > 10
    for d in tr.result.raw:
        col, row = (round(c * d.scale / STRIDE - 0.5) for c in (d.x, d.y))
        assert by_scale[d.scale][row, col] == pytest.approx(d.score, abs=1e-5)


@pytest.mark.parametrize("size", [(16, 16), (6000, 6000), (6000, 10), (10, 6000), (1000, 777)])
def test_a2_any_size_runs_and_points_land_on_the_photo(size):
    model = _busy_model()
    rng = np.random.default_rng(0)
    w, h = size
    img = Image.fromarray(rng.integers(0, 256, (min(h, 400), min(w, 400), 3), dtype=np.uint8)).resize(size)
    tr = trace(model, img)
    assert [v.scale for v in tr.views] == list(PYRAMID_SCALES)
    for d in tr.result.raw:
        assert tr.fitted.in_content(d.x, d.y)
    v = watch_view(tr, None, 0.5)
    for _, _, _, _, x, y in v.rows:
        assert 0 <= x <= w and 0 <= y <= h


def test_padding_peaks_are_dropped():
    """A photo with lots of padding: detect() alone finds peaks on the padding, trace() keeps none of them."""
    model = _busy_model()
    img = Image.new("RGB", (1000, 401), (120, 120, 120))  # pads to 1008 x 416
    tr = trace(model, img)
    raw_all = detect(model, tr.fitted.image).raw
    on_pad = [d for d in raw_all if not tr.fitted.in_content(d.x, d.y)]
    assert on_pad, "test needs peaks on the padding to mean anything"
    assert len(tr.result.raw) == len(raw_all) - len(on_pad)


def test_a5_rerender_never_calls_the_model():
    model = _busy_model()
    tr = trace(model, Image.new("RGB", (320, 160), (90, 90, 90)))

    def boom(*a, **k):
        raise AssertionError("the model was called")
    for m in model.modules():
        m.forward = boom
    counts = [len(watch_view(tr, None, tau).rows) for tau in (0.05, 0.5, 0.95)]
    assert counts == sorted(counts, reverse=True)  # a higher threshold never adds detections


def test_truth_on_a_shrunk_photo_is_refused():
    model = _busy_model()
    tr = trace(model, Image.new("RGB", (2720, 1600)))
    with pytest.raises(ValueError):
        watch_view(tr, (), 0.5)


def test_place_sign_clamps_and_adds_truth():
    bg = Image.new("RGB", (320, 160))
    placed = place_sign(bg, (), Image.new("RGB", (50, 50), (255, 0, 0)), 14, 500, 1.0, 0.0)
    assert placed.long_side == 160 and placed.asked_long_side == 500
    (sign,) = placed.truth
    assert sign.gt_index == PASTED_GT_INDEX and sign.class_id == 14 and sign.box == (160, 0, 320, 160)


@needs_real
def test_fool_view_finds_a_mid_size_stop_sign(seed0, dev_photos):
    """A 32 px stop sign on a sign-free dev photo is found: the tab-2 path grades the pasted sign."""
    from findobject.data.gtsrb import load_sign, read_train_index
    from findobject.demo.examples import sign_per_class
    stop = load_sign(sign_per_class(read_train_index(Path("data/raw/gtsrb")))[14])
    placed = place_sign(load_photo(dev_photos[108]), (), stop, 14, 32, 0.5, 0.5)
    v = fool_view(trace(seed0, placed.image), placed, TAU_W)
    assert v.text.startswith("✅")
