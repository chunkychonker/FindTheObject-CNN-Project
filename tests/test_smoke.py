import pytest
import torch

from findobject.models.baseline import BaselineCNN, ModelOutput


def test_forward_shapes_square_and_non_square():
    model = BaselineCNN().eval()
    out = model(torch.rand(2, 3, 128, 128))
    assert isinstance(out, ModelOutput)
    assert out.heatmap_logits.shape == (2, 1, 32, 32) and out.class_logits.shape == (2, 43)
    assert model(torch.rand(1, 3, 96, 128)).heatmap_logits.shape == (1, 1, 24, 32)


@pytest.mark.parametrize("shape", [(1, 2, 128, 128), (1, 3, 130, 130), (3, 128, 128)])
def test_rejects_bad_input(shape):
    with pytest.raises(ValueError):
        BaselineCNN()(torch.rand(*shape))


def test_parameter_budget():
    assert sum(p.numel() for p in BaselineCNN().parameters()) < 2_000_000


def test_heatmap_cell_receptive_field_covers_largest_sign():
    # Which input pixels can change one central heatmap cell? Their extent is the receptive field.
    torch.manual_seed(0)
    model = BaselineCNN().eval()
    x = torch.rand(1, 3, 128, 128, requires_grad=True)
    model(x).heatmap_logits[0, 0, 16, 16].backward()
    touched = x.grad.abs().sum(dim=(0, 1)) > 0
    rows, cols = touched.any(dim=1).nonzero(), touched.any(dim=0).nonzero()
    assert rows.max() - rows.min() + 1 >= 48 and cols.max() - cols.min() + 1 >= 48  # sign_max = 48 px


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple MPS device")
def test_forward_backward_on_mps():
    model = BaselineCNN().to("mps")
    out = model(torch.rand(4, 3, 128, 128, device="mps"))
    (out.heatmap_logits.mean() + out.class_logits.mean()).backward()
    assert all(p.grad is not None for p in model.parameters())
