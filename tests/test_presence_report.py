"""The pure v3a report pieces: presence_report, p6_sheet, p6_tally, score_scenes (docs/v3a-spec.md §2.6, §2.9)."""

import numpy as np
import pytest
import torch

from findobject.eval.p6_sheet import LABEL_PX, N_CONTROL, N_TOP, TILE_ZOOM, Tile, choose_tiles, render_sheet
from findobject.eval.p6_tally import P6_NEEDS, Label, parse_labels, tally
from findobject.eval.presence_report import (PhotoIntervals, SceneIntervals, Side, auroc_difference, family_metrics,
                                             matched_found, seam_check)
from findobject.eval.stats import wilson


# --- presence_report ---

def test_family_metrics_by_hand():
    # 4 positives, 4 negatives, τ = 0.5. Found needs presence >= τ AND class right AND peak in box.
    pos = np.float32([0.9, 0.8, 0.6, 0.4])
    correct = np.array([True, False, True, True])
    hit = np.array([True, True, True, True])
    neg = np.float32([0.7, 0.3, 0.2, 0.1])
    m = family_metrics(pos, correct, hit, neg, np.float32(0.5), SceneIntervals(4, 4))
    assert m["false_alarm"]["k"] == 1 and m["false_alarm"]["n"] == 4  # only 0.7
    assert m["found"]["k"] == 2  # 0.9 and 0.6; 0.8 has the wrong class, 0.4 is below τ
    assert m["joint"]["k"] == 3  # τ ignored
    # pairs (pos > neg): 0.9:4, 0.8:4, 0.6:3, 0.4:3 = 14 of 16
    assert m["auroc"]["value"] == 14 / 16
    assert m["auroc"]["lo"] <= 14 / 16 <= m["auroc"]["hi"]


def test_scene_intervals_rates_are_wilson():
    ok = np.array([True] * 3 + [False] * 7)
    r = SceneIntervals(10, 5).rate(ok, Side.POS)
    w = wilson(3, 10)
    assert (r["k"], r["n"], r["lo"], r["hi"]) == (3, 10, w.lo, w.hi)


def test_subset_restricts_positives_only():
    # 20 high positives outside the subset, 20 low ones inside (40 rows, so no resample misses the subset).
    pos = np.float32([0.9] * 20 + [0.1] * 20)
    neg = np.float32([0.5] * 5)
    within = np.arange(40) >= 20
    ones = np.ones(40, bool)
    m = family_metrics(pos, ones, ones, neg, np.float32(0.5), SceneIntervals(40, 5), within)
    assert m["auroc"]["value"] == 0.0  # only the 0.1 positives are compared
    assert m["found"]["n"] == 20 and m["found"]["k"] == 0 and m["false_alarm"]["n"] == 5


def test_photo_intervals_resample_whole_photos():
    # Photo 7 has 2 positives and 1 negative; photo 8 has 1 of each. A resample must bring a photo's
    # positive and negative rows together, as many times as the photo was drawn.
    pos_photo, neg_photo = np.array([7, 7, 8]), np.array([7, 8])
    seen = []
    PhotoIntervals(pos_photo, neg_photo).bootstrap(lambda p, n: seen.append((p.copy(), n.copy())) or 0.0)
    for p, n in seen:
        draws7, draws8 = (n == 0).sum(), (n == 1).sum()
        assert draws7 + draws8 == 2  # as many photos drawn as there are
        assert (p == 0).sum() == (p == 1).sum() == draws7 and (p == 2).sum() == draws8


def test_photo_intervals_are_reproducible_and_constant_is_degenerate():
    iv = PhotoIntervals(np.arange(20) // 2, np.arange(20) // 2)
    rng = np.random.default_rng(1)
    pos, neg = rng.random(20), rng.random(20)
    assert iv.auroc(pos, neg) == iv.auroc(pos, neg)
    assert iv.bootstrap(lambda p, n: 0.25) == (0.25, 0.25)


def test_matched_found_respects_the_rate():
    neg = np.linspace(0, 1, 200, dtype=np.float32)
    pos = np.float32([0.999, 0.98, 0.5])
    out = matched_found(pos, np.ones(3, bool), neg)
    assert out["0.01"]["false_alarms"] <= 2 and out["0.05"]["false_alarms"] <= 10
    assert out["0.05"]["found"] >= out["0.01"]["found"]


def test_auroc_difference_needs_the_same_scenes():
    with pytest.raises(ValueError):
        auroc_difference(np.zeros(3), np.zeros(2), np.zeros(4), np.zeros(2), SceneIntervals(3, 2))


def test_seam_check_by_hand():
    tau = np.float32(0.5)
    real = np.float32([0.9, 0.4, 0.3, 0.6])
    hybrid = np.float32([0.9, 0.7, 0.8, 0.2])
    s = seam_check(real, hybrid, tau)
    assert s["real_yes"]["k"] == 2 and s["hybrid_yes"]["k"] == 3
    assert s["gap_points"] == pytest.approx(25.0)
    assert (s["only_real"], s["only_hybrid"]) == (1, 2)
    # hybrid − real: 0, 0.3, 0.5, −0.4 -> median of (−0.4, 0, 0.3, 0.5) = 0.15
    assert s["median_presence_diff_hybrid_minus_real"] == pytest.approx(0.15, abs=1e-6)
    with pytest.raises(ValueError):
        seam_check(real, hybrid[:3], tau)


# --- p6_sheet ---

def test_choose_tiles_top_and_controls():
    presence = np.random.default_rng(0).random(300).astype(np.float32)
    tiles = choose_tiles(presence, np.random.default_rng(6))
    assert sorted(t.number for t in tiles) == list(range(1, N_TOP + N_CONTROL + 1))
    top = {t.row for t in tiles if t.top}
    assert top == set(np.argsort(-presence)[:N_TOP].tolist())
    rows = [t.row for t in tiles]
    assert len(set(rows)) == len(rows)  # controls never repeat a top row or each other
    assert choose_tiles(presence, np.random.default_rng(6)) == tiles
    # Shuffled: the top tiles are not simply numbers 1-20.
    assert {t.number for t in tiles if t.top} != set(range(1, N_TOP + 1))


def test_choose_tiles_needs_enough_rows():
    with pytest.raises(ValueError):
        choose_tiles(np.zeros(N_TOP + N_CONTROL - 1), np.random.default_rng(0))


def test_render_sheet_draws_each_tile_untouched_under_its_number():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (50, 8, 8, 3), dtype=np.uint8)
    tiles = [Tile(2, 10, True), Tile(1, 3, False)]
    sheet = np.asarray(render_sheet(images, tiles))
    s = 8 * TILE_ZOOM
    first = sheet[LABEL_PX:LABEL_PX + s, :s]  # tile number 1 comes first
    assert np.array_equal(first, images[3].repeat(TILE_ZOOM, 0).repeat(TILE_ZOOM, 1))
    second = sheet[LABEL_PX:LABEL_PX + s, s:2 * s]
    assert np.array_equal(second, images[10].repeat(TILE_ZOOM, 0).repeat(TILE_ZOOM, 1))


# --- p6_tally ---

def _tiles():
    return [Tile(i + 1, i, i < N_TOP) for i in range(N_TOP + N_CONTROL)]


def test_tally_counts_and_grades():
    labels = {i: Label.D for i in range(1, 41)}
    for i in range(1, P6_NEEDS + 1):  # tiles 1..10 are top
        labels[i] = Label.A if i % 2 else Label.B
    labels[N_TOP + 1] = Label.A  # one control sign
    labels[N_TOP + 2] = Label.C
    out = tally(labels, _tiles())
    assert out["top"]["sign"] == P6_NEEDS and out["p6_holds"]
    assert out["control"]["sign"] == 1 and out["control"]["counts"]["C"] == 1
    labels[1] = Label.C
    assert not tally(labels, _tiles())["p6_holds"]


@pytest.mark.parametrize("raw", [{str(i): "D" for i in range(1, 40)},  # one missing
                                 {**{str(i): "D" for i in range(1, 41)}, "41": "D"},  # one extra
                                 {**{str(i): "D" for i in range(1, 41)}, "7": "E"}])  # not a label
def test_parse_labels_rejects_incomplete_or_bad(raw):
    with pytest.raises(ValueError):
        parse_labels(raw, 40)


# --- score_scenes ---

def test_score_scenes_keeps_dataset_order():
    from torch.utils.data import DataLoader

    from findobject.data.dataset import Sample
    from findobject.eval.score import score_scenes
    from findobject.models.baseline import ModelOutput

    class Stub(torch.nn.Module):
        def forward(self, image):  # presence logit = image mean; class 1; peak at cell (0, 0)
            b = len(image)
            heat = torch.full((b, 1, 4, 4), -10.0)
            heat[:, 0, 0, 0] = image.mean(dim=(1, 2, 3))
            return ModelOutput(heatmap_logits=heat, class_logits=torch.tensor([[0.0, 1.0]]).repeat(b, 1))

    samples = [Sample(torch.full((3, 16, 16), v), torch.zeros(1, 4, 4), torch.tensor(c), torch.tensor([0., 0, 4, 4]))
               for v, c in ((0.2, 1), (0.9, 0), (0.5, 1))]
    s = score_scenes(Stub(), DataLoader(samples, batch_size=2), torch.device("cpu"))
    assert np.allclose(s.presence, torch.sigmoid(torch.tensor([0.2, 0.9, 0.5])).numpy())
    assert s.presence.dtype == np.float32
    assert s.y_true.tolist() == [1, 0, 1] and s.y_pred.tolist() == [1, 1, 1]
    assert s.hit.tolist() == [True, True, True]  # peak cell (0,0) centre is at (2, 2), inside (0,0)-(4,4)
