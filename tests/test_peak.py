import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from findobject.eval.heatmap_metrics import peak_centers
from findobject.models.baseline import BaselineCNN
from findobject.models.peak import PeakCNN, peak_cells, window_features
from findobject.training.checkpoint import load_model, save_checkpoint


def _peak_at_top_left(model):
    """Zero the heatmap head: a constant map's argmax is its first cell, (0, 0)."""
    torch.nn.init.zeros_(model.heatmap_head.weight)
    torch.nn.init.zeros_(model.heatmap_head.bias)
    return model.eval()


# --- same model as v1 except the class head's input ---

def test_state_dict_matches_baseline_and_heatmap_is_identical():
    baseline, peak = BaselineCNN().eval(), PeakCNN().eval()
    assert {k: v.shape for k, v in baseline.state_dict().items()} == {k: v.shape for k, v in peak.state_dict().items()}
    peak.load_state_dict(baseline.state_dict())
    x = torch.rand(2, 3, 128, 128)
    assert torch.equal(baseline(x).heatmap_logits, peak(x).heatmap_logits)


def test_same_seed_gives_same_initial_weights_as_baseline():
    torch.manual_seed(0)
    baseline = BaselineCNN().state_dict()
    torch.manual_seed(0)
    peak = PeakCNN().state_dict()
    assert all(torch.equal(baseline[k], peak[k]) for k in baseline)


def test_forward_shapes_and_bad_input():
    model = PeakCNN().eval()
    out = model(torch.rand(2, 3, 128, 128))
    assert out.heatmap_logits.shape == (2, 1, 32, 32) and out.class_logits.shape == (2, 43)
    assert model(torch.rand(1, 3, 96, 128)).class_logits.shape == (1, 43)
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 130, 130))


# --- the class comes from the peak only ---

def test_class_ignores_pixels_far_from_the_peak_but_baseline_does_not():
    peak = _peak_at_top_left(PeakCNN())
    baseline = BaselineCNN().eval()
    baseline.load_state_dict(peak.state_dict())
    x = torch.rand(1, 3, 128, 128)
    far = x.clone()
    # Window cells (0..1, 0..1) see scene px up to ~8 + 32 (receptive-field half-width) = 40 < 64.
    far[..., 64:, :] = torch.rand_like(far[..., 64:, :])
    far[..., :, 64:] = torch.rand_like(far[..., :, 64:])
    assert torch.equal(peak(x).class_logits, peak(far).class_logits)
    assert not torch.equal(baseline(x).class_logits, baseline(far).class_logits)


def test_forward_classifies_the_mean_of_the_window_around_the_peak():
    model = _peak_at_top_left(PeakCNN())
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        corner = model.trunk(x)[:, :, :2, :2].mean(dim=(2, 3))  # corner peak: 2x2 in-bounds cells
        assert torch.allclose(model(x).class_logits, model.class_head(corner), atol=1e-5)


def test_window_features_average_in_bounds_cells():
    features = torch.arange(16.0).reshape(1, 1, 4, 4).repeat(3, 1, 1, 1)  # cell (r, c) holds 4r + c
    cells = torch.tensor([[0, 0], [0, 2], [2, 1]])
    expected = [(0 + 1 + 4 + 5) / 4,  # corner: 4 cells
                (1 + 2 + 3 + 5 + 6 + 7) / 6,  # top edge: 6 cells
                (4 + 5 + 6 + 8 + 9 + 10 + 12 + 13 + 14) / 9]  # interior: 9 cells
    assert window_features(features, cells).squeeze(1).tolist() == pytest.approx(expected)


def test_peak_cells_is_the_cell_pointing_accuracy_scores():
    torch.manual_seed(0)
    ties = torch.zeros(2, 1, 8, 8)
    ties[1, 0, 1, 5] = ties[1, 0, 3, 2] = 1.0  # two equal maxima: row-major order picks (1, 5)
    for maps in (torch.randn(16, 1, 8, 8), ties):
        cells = peak_cells(maps).float()
        centers = torch.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5], dim=1) * (32 / 8)
        assert torch.equal(centers, peak_centers(maps, scene_size=32))
    assert peak_cells(ties).tolist() == [[0, 0], [1, 5]]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple MPS device")
def test_forward_backward_on_mps():
    model = PeakCNN().to("mps")
    out = model(torch.rand(4, 3, 128, 128, device="mps"))
    (out.heatmap_logits.mean() + out.class_logits.mean()).backward()
    assert all(p.grad is not None for p in model.parameters())


# --- checkpoints rebuild the right architecture ---

def _save(path, model, config):
    optimizer = AdamW(model.parameters(), lr=0.001)
    save_checkpoint(path, model, optimizer, CosineAnnealingLR(optimizer, T_max=1), 1, config, {})


def test_peak_checkpoint_round_trip(tmp_path):
    model = PeakCNN().eval()
    _save(tmp_path / "p.pt", model, {"model": {"arch": "peak"}})
    reloaded = load_model(tmp_path / "p.pt", torch.device("cpu"))
    x = torch.rand(1, 3, 128, 128)
    assert isinstance(reloaded, PeakCNN) and torch.equal(model(x).class_logits, reloaded(x).class_logits)


def test_checkpoint_without_model_section_loads_as_baseline(tmp_path):
    _save(tmp_path / "v1.pt", BaselineCNN(), {"train": {}})  # v1 checkpoints predate the section
    assert type(load_model(tmp_path / "v1.pt", torch.device("cpu"))) is BaselineCNN


def test_checkpoint_with_unknown_arch_raises(tmp_path):
    _save(tmp_path / "x.pt", BaselineCNN(), {"model": {"arch": "resnet"}})
    with pytest.raises(ValueError):
        load_model(tmp_path / "x.pt", torch.device("cpu"))
