"""A8 (docs/v3a-spec.md §2.10): real negative windows."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from findobject.data.gtsdb import PHOTO_SIZE, RealPhoto, RealSign
from findobject.data.real_negatives import (BAND_JITTER_PX, MARGIN_PX, Stratum, build_real_negatives,
                                            plan_negative)
from findobject.data.real_windows import Skip, scaled_size
from findobject.data.synth import SynthConfig

CFG = SynthConfig()  # 128 px windows
S = CFG.scene_size


def _touches_widened_box(x0, y0, scale, box, photo_size) -> bool:
    """Recomputed here, independently: widen in photo px, clip, scale outward, test pixel overlap."""
    w, h = scaled_size(photo_size, scale)
    x1, y1, x2, y2 = box
    m = MARGIN_PX
    x1, y1 = math.floor(max(0, x1 - m) * scale), math.floor(max(0, y1 - m) * scale)
    x2, y2 = min(w, math.ceil(min(photo_size[0], x2 + m) * scale)), min(h, math.ceil(min(photo_size[1], y2 + m) * scale))
    return x0 < x2 and x1 < x0 + S and y0 < y2 and y1 < y0 + S


def _layout(rs):
    w, h = int(rs.integers(300, 1400)), int(rs.integers(300, 900))
    boxes = []
    for _ in range(int(rs.integers(0, 6))):
        bw, bh = int(rs.integers(8, 200)), int(rs.integers(8, 200))
        x1, y1 = int(rs.integers(0, w - bw)), int(rs.integers(0, h - bh))
        boxes.append((x1, y1, x1 + bw, y1 + bh))
    return (w, h), boxes


def test_plans_lie_inside_the_photo_and_touch_no_widened_box():
    rs = np.random.default_rng(0)
    placed = 0
    for trial in range(600):
        size, boxes = _layout(rs)
        scale = float(rs.choice([1.0, 0.8, 0.5, 48 / 130]))
        w, h = scaled_size(size, scale)
        if w < S or h < S:
            continue
        stratum = Stratum.BAND if trial % 2 else Stratum.UNIFORM
        row = float(rs.uniform(0, h)) if stratum is Stratum.BAND else None
        plan = plan_negative(size, boxes, scale, stratum, row, np.random.default_rng(trial), CFG)
        if plan is Skip.NO_CLEAR_SPOT:
            continue
        placed += 1
        assert plan.scale == scale and plan.box == (0, 0, 0, 0)
        assert 0 <= plan.x0 <= w - S and 0 <= plan.y0 <= h - S
        assert not any(_touches_widened_box(plan.x0, plan.y0, scale, b, size) for b in boxes)
        if stratum is Stratum.BAND and 0 < plan.y0 < h - S:  # not clamped
            assert abs(plan.y0 + S / 2 - row) <= BAND_JITTER_PX + 0.5  # + rounding to a whole pixel
    assert placed > 400  # the test exercised the rules, not just skips


def test_margin_is_respected_to_the_pixel():
    # One box at x in [300, 340); the photo is otherwise empty, at scale 1. A window ending at x = 296
    # (exclusive) is clear; ending at 297 touches the widened box [296, 344).
    rng = np.random.default_rng(0)
    for _ in range(300):
        plan = plan_negative((1360, 800), [(300, 0, 340, 800)], 1.0, Stratum.UNIFORM, None, rng, CFG)
        assert plan.x0 + S <= 296 or plan.x0 >= 344


def test_band_windows_follow_the_row():
    rng = np.random.default_rng(1)
    centers = [plan_negative(PHOTO_SIZE, [], 1.0, Stratum.BAND, 400.0, rng, CFG).y0 + S / 2 for _ in range(500)]
    assert min(centers) >= 400 - 16.5 and max(centers) <= 400 + 16.5
    assert max(centers) - min(centers) > 25  # the jitter is really there
    # Near the top the window is clamped to the photo instead.
    assert plan_negative(PHOTO_SIZE, [], 1.0, Stratum.BAND, 5.0, rng, CFG).y0 == 0


def test_photo_covered_by_a_sign_is_skipped():
    plan = plan_negative(PHOTO_SIZE, [(0, 0, *PHOTO_SIZE)], 1.0, Stratum.UNIFORM, None, np.random.default_rng(0), CFG)
    assert plan is Skip.NO_CLEAR_SPOT


@pytest.mark.parametrize("scale,stratum,row", [(0.0, Stratum.UNIFORM, None), (1.5, Stratum.UNIFORM, None),
                                               (1.0, Stratum.UNIFORM, 100.0), (1.0, Stratum.BAND, None)])
def test_rejects_bad_arguments(scale, stratum, row):
    with pytest.raises(ValueError):
        plan_negative(PHOTO_SIZE, [], scale, stratum, row, np.random.default_rng(0), CFG)


def test_rejects_photo_smaller_than_the_window():
    with pytest.raises(ValueError):
        plan_negative(PHOTO_SIZE, [], 0.1, Stratum.UNIFORM, None, np.random.default_rng(0), CFG)


# --- build_real_negatives ---

@pytest.fixture(scope="module")
def photos(tmp_path_factory) -> list[RealPhoto]:
    root = tmp_path_factory.mktemp("gtsdb")
    rs = np.random.default_rng(0)
    out = []
    signs_for = {0: (), 1: (RealSign((600, 300, 650, 350), 1), RealSign((100, 500, 300, 700), 2)),
                 2: (RealSign((0, 0, *PHOTO_SIZE), 3),)}  # photo 2: a "sign" covers everything
    for index, signs in signs_for.items():
        path = root / f"{index:05d}.png"
        Image.fromarray(rs.integers(0, 256, (PHOTO_SIZE[1], PHOTO_SIZE[0], 3), dtype=np.uint8)).save(path)
        out.append(RealPhoto(path, index + 600, signs))
    return out


SCALES = np.minimum(1.0, 48 / np.array([30, 40, 60, 96, 150]))
ROWS = np.array([325.0, 600.0, 410.0])


def test_two_windows_per_clear_photo_and_skips_counted(photos):
    out = build_real_negatives(photos, SCALES, ROWS, seed=0, cfg=CFG)
    a = out.arrays
    assert a["photo_index"].tolist() == [600, 600, 601, 601]
    assert a["stratum"].tolist() == [0, 1, 0, 1]
    assert out.skipped == {Stratum.UNIFORM: 1, Stratum.BAND: 1}
    assert a["images"].shape == (4, S, S, 3) and a["images"].dtype == np.uint8 and int(a["heatmap_size"]) == 32
    assert set(a["scale"].tolist()) <= set(SCALES.tolist())


def test_stored_windows_match_their_origin_and_scale(photos):
    a = build_real_negatives(photos, SCALES, ROWS, seed=0, cfg=CFG).arrays
    by_index = {p.index: p for p in photos}
    for img, p, scale, (x0, y0) in zip(a["images"], a["photo_index"], a["scale"], a["origin"]):
        photo = by_index[int(p)]
        with Image.open(photo.path) as im:
            big = im.convert("RGB")
        if scale != 1:
            big = big.resize(scaled_size(big.size, scale), Image.BILINEAR)
        assert np.array_equal(img, np.asarray(big.crop((x0, y0, x0 + S, y0 + S))))
        assert not any(_touches_widened_box(int(x0), int(y0), float(scale), s.box, PHOTO_SIZE) for s in photo.signs)


def test_rebuild_is_identical_and_independent_of_photo_order(photos):
    a = build_real_negatives(photos, SCALES, ROWS, seed=0, cfg=CFG).arrays
    b = build_real_negatives(photos[::-1], SCALES, ROWS, seed=0, cfg=CFG).arrays
    order = np.lexsort((b["stratum"], b["photo_index"]))
    for k in a:
        assert np.array_equal(a[k], b[k] if b[k].ndim == 0 else b[k][order]), k
    c = build_real_negatives(photos, SCALES, ROWS, seed=1, cfg=CFG).arrays
    assert not np.array_equal(a["origin"], c["origin"])


def test_rejects_empty_scale_or_row_sets(photos):
    with pytest.raises(ValueError):
        build_real_negatives(photos, np.array([]), ROWS, 0, CFG)
    with pytest.raises(ValueError):
        build_real_negatives(photos, SCALES, np.array([]), 0, CFG)


# --- A8 on the built files (skipped until scripts/make_empty_dataset.py has run) ---

PROCESSED = Path("data/processed")
GTSDB = Path("data/raw/gtsdb/FullIJCNN2013")


@pytest.mark.parametrize("split", ["dev", "test"])
def test_built_file_obeys_a8(split):
    path = PROCESSED / f"real_empty_{split}.npz"
    if not path.exists() or not GTSDB.exists():
        pytest.skip(f"{path} or {GTSDB} not built")
    from findobject.data.gtsdb import read_gtsdb
    photos = {p.index: p for p in read_gtsdb(GTSDB)}
    with np.load(path) as d:
        a = {k: d[k] for k in d}
    with np.load(PROCESSED / f"real_{split}.npz") as d:
        positive_scales = set(np.minimum(1.0, CFG.sign_max / d["native_long_side"].astype(np.float64)).tolist())
    assert set(a["scale"].tolist()) <= positive_scales
    for p, scale, (x0, y0) in zip(a["photo_index"], a["scale"], a["origin"]):
        w, h = scaled_size(PHOTO_SIZE, float(scale))
        assert 0 <= x0 <= w - S and 0 <= y0 <= h - S
        assert not any(_touches_widened_box(int(x0), int(y0), float(scale), s.box, PHOTO_SIZE)
                       for s in photos[int(p)].signs)
