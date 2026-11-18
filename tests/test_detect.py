"""detect() on whole inputs: agrees with windows where padding can't reach, and with predict() on one scene
(docs/v3b-spec.md A1, A10, A11)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from findobject.inference.detect import detect
from findobject.inference.peaks import MIN_PEAK_SCORE
from findobject.inference.predict import predict
from findobject.models.peak import PeakCNN, window_features_at

SEED0 = Path("outputs/checkpoints/peak_decoy-20261112-011953/best.pt")
VAL = Path("data/processed/val.npz")
MARGIN = 8  # cells: the receptive field reaches 32 px = 8 cells, so from 8 cells in, padding can't be seen


def _logits_and_features(model, x):
    f = model.trunk(x)
    return model.heatmap_head(f)[0, 0], f


@torch.no_grad()
def test_a1_crop_equals_whole_input_away_from_the_crop_border():
    torch.manual_seed(0)
    model = PeakCNN().eval()
    x = torch.rand(1, 3, 800, 1360)
    whole, whole_f = _logits_and_features(model, x)
    for y0, x0, h, w in [(200, 400, 256, 256), (0, 0, 128, 192), (672, 1232, 128, 128)]:
        crop, crop_f = _logits_and_features(model, x[:, :, y0:y0 + h, x0:x0 + w])
        ref = whole[y0 // 4:(y0 + h) // 4, x0 // 4:(x0 + w) // 4]
        inner = (slice(MARGIN, h // 4 - MARGIN), slice(MARGIN, w // 4 - MARGIN))
        assert (crop - ref)[inner].abs().max() <= 1e-6
        # The class read pools 3x3 cells, so it needs one more cell of margin.
        rows, cols = torch.meshgrid(torch.arange(MARGIN + 1, h // 4 - MARGIN - 1),
                                    torch.arange(MARGIN + 1, w // 4 - MARGIN - 1), indexing="ij")
        cells = torch.stack([rows.flatten(), cols.flatten()], 1)
        shifted = cells + torch.tensor([y0 // 4, x0 // 4])
        diff = window_features_at(crop_f, cells) - window_features_at(whole_f, shifted)
        assert diff.abs().max() <= 1e-6


@torch.no_grad()
def test_a1_negative_control_the_border_does_differ():
    torch.manual_seed(0)
    model = PeakCNN().eval()
    x = (torch.rand(1, 3, 800, 1360) > 0.5).float()  # high contrast
    whole, _ = _logits_and_features(model, x)
    crop, _ = _logits_and_features(model, x[:, :, 200:456, 400:656])
    ref = whole[50:114, 100:164]
    near = (crop - ref).abs()
    ring = torch.cat([near[:5].flatten(), near[-5:].flatten(), near[:, :5].flatten(), near[:, -5:].flatten()])
    assert ring.max() > 1e-6


def test_a11_detect_refuses_a_model_in_train_mode():
    with pytest.raises(RuntimeError, match="eval"):
        detect(PeakCNN().train(), Image.new("RGB", (64, 64)), scales=(1.0,))


def test_detect_refuses_a_scale_that_does_not_divide_the_photo():
    with pytest.raises(ValueError):
        detect(PeakCNN().eval(), Image.new("RGB", (1360, 800)), scales=(0.33,))  # 448.8 x 264: not whole
    with pytest.raises(ValueError):
        detect(PeakCNN().eval(), Image.new("RGB", (1360, 800)), scales=(2.0,))


def test_detect_returns_raw_and_suppressed_lists_in_order():
    torch.manual_seed(1)
    img = Image.fromarray((np.random.default_rng(0).random((800, 1360, 3)) * 255).astype(np.uint8))
    res = detect(PeakCNN().eval(), img)
    assert len(res.kept) <= len(res.raw) and set(res.kept) <= set(res.raw)
    assert {d.scale for d in res.raw} <= {1.0, 0.5, 0.25}


@pytest.mark.skipif(not (SEED0.exists() and VAL.exists()), reason="needs the v3a.1 seed-0 run and val.npz")
def test_a10_on_one_scene_the_top_detection_is_predicts_peak():
    from findobject.training.checkpoint import load_model
    model = load_model(SEED0, torch.device("cpu"))
    images = np.load(VAL)["images"]
    checked = 0
    for img in images[:200]:
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None]
        p = predict(model, x)[0]
        if p.peak_prob < MIN_PEAK_SCORE:
            continue
        top = detect(model, Image.fromarray(img), scales=(1.0,)).kept[0]
        assert abs(top.score - p.peak_prob) <= 1e-6
        assert (top.x, top.y) == p.peak_xy and top.class_id == p.class_id
        checked += 1
        if checked == 64:
            break
    assert checked == 64
