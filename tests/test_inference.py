import numpy as np
import pytest
import torch
from PIL import Image

from findobject.data.class_names import CLASS_NAMES
from findobject.data.gtsrb import NUM_CLASSES
from findobject.eval.heatmap_metrics import point_in_box
from findobject.inference.predict import predict
from findobject.inference.preprocess import to_scene
from findobject.models.peak import PeakCNN
from findobject.viz.overlay import COLORMAP, heatmap_overlay


def test_one_class_name_per_class_id():
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert CLASS_NAMES[14] == "stop" and CLASS_NAMES[27] == "pedestrians"


# --- to_scene: arbitrary photo -> model-sized square ---

def test_to_scene_non_square_gives_square_scene_in_unit_range():
    scene = to_scene(Image.new("RGB", (200, 100), (255, 128, 0)), 128)
    assert scene.shape == (3, 128, 128) and scene.dtype == torch.float32
    assert 0 <= scene.min() and scene.max() <= 1


def test_to_scene_keeps_the_middle_of_a_wide_image():
    # Thirds: red | blue | green. A centered square crop of a 300x100 image is exactly the blue third.
    arr = np.zeros((100, 300, 3), dtype=np.uint8)
    arr[:, :100, 0] = arr[:, 100:200, 2] = arr[:, 200:, 1] = 255
    scene = to_scene(Image.fromarray(arr), 128)
    assert torch.equal(scene[2], torch.ones(128, 128))  # all blue
    assert scene[0].max() == 0 and scene[1].max() == 0  # no red or green leaked in


def test_to_scene_passes_a_scene_sized_image_through_exactly():
    arr = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    scene = to_scene(Image.fromarray(arr), 128)
    assert torch.equal((scene * 255).round().to(torch.uint8), torch.from_numpy(arr).permute(2, 0, 1))


def test_to_scene_converts_grayscale_to_rgb():
    assert to_scene(Image.new("L", (64, 64), 77), 64).shape == (3, 64, 64)


@pytest.mark.parametrize("size", [130, 0, -4])
def test_to_scene_rejects_sizes_that_are_not_positive_multiples_of_stride(size):
    with pytest.raises(ValueError):
        to_scene(Image.new("RGB", (100, 100)), size)


# --- predict: model outputs -> readable predictions ---

def test_predict_reads_class_and_peak_like_evaluation():
    torch.manual_seed(0)
    model = PeakCNN().eval()
    torch.nn.init.zeros_(model.heatmap_head.weight)  # constant heatmap of logit 0:
    torch.nn.init.zeros_(model.heatmap_head.bias)  # argmax is the first cell, sigmoid(0) = 0.5
    images = torch.rand(2, 3, 128, 128)
    preds = predict(model, images)
    class_logits = model(images).class_logits.detach()

    assert len(preds) == 2
    for p, logits in zip(preds, class_logits):
        assert p.class_id == int(logits.argmax())
        assert p.class_prob == pytest.approx(float(logits.softmax(0).max()), abs=1e-6)
        assert p.peak_xy == (2.0, 2.0)  # center of cell (0, 0) at stride 4
        assert p.peak_prob == pytest.approx(0.5)
        assert p.heatmap.shape == (32, 32) and p.heatmap.device.type == "cpu"


def test_predict_rejects_non_square_scenes():
    with pytest.raises(ValueError, match="square"):
        predict(PeakCNN().eval(), torch.rand(1, 3, 128, 64))


# --- overlay ---

def test_zero_heatmap_leaves_the_scene_untouched():
    arr = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
    out = heatmap_overlay(image, torch.zeros(16, 16))
    assert out.dtype == np.uint8 and out.shape == (64, 64, 3)
    assert np.array_equal(out, arr)


def test_full_heatmap_at_alpha_1_is_the_colormap_color():
    import matplotlib.pyplot as plt

    out = heatmap_overlay(torch.zeros(3, 8, 8), torch.ones(2, 2), alpha=1.0)
    expected = np.round(np.array(plt.get_cmap(COLORMAP)(1.0)[:3]) * 255).astype(np.uint8)
    assert (out == expected).all()


def test_overlay_rejects_bad_alpha_and_shape():
    with pytest.raises(ValueError):
        heatmap_overlay(torch.zeros(3, 8, 8), torch.zeros(2, 2), alpha=1.5)
    with pytest.raises(ValueError):
        heatmap_overlay(torch.zeros(8, 8), torch.zeros(2, 2))


# --- point_in_box: the joint-accuracy rule, x2/y2 exclusive ---

def test_point_in_box_edges():
    box = torch.tensor([[10, 20, 30, 40]] * 4)
    points = torch.tensor([[10.0, 20.0], [29.9, 39.9], [30.0, 25.0], [15.0, 40.0]])
    assert point_in_box(points, box).tolist() == [True, True, False, False]
