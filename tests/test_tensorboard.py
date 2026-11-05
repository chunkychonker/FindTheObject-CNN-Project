import torch

from findobject.eval.heatmap_metrics import center_error_px, peak_centers, peak_in_box
from findobject.viz.tensorboard import TBLogger


def _hm_with_peak(row: int, col: int, size: int = 32) -> torch.Tensor:
    hm = torch.zeros(1, size, size)
    hm[0, row, col] = 1.0
    return hm


def test_peak_center_uses_cell_center_convention():
    # scene 128, heatmap 32 -> stride 4; cell (row 2, col 5) centered at x=22, y=10
    assert peak_centers(_hm_with_peak(2, 5), 128).tolist() == [[22.0, 10.0]]


def test_center_error_and_hit():
    hm = _hm_with_peak(2, 5)  # peak at (22, 10)
    inside = torch.tensor([[16, 4, 28, 16]])  # center (22, 10)
    outside = torch.tensor([[60, 60, 80, 80]])
    assert center_error_px(hm, inside, 128).item() == 0.0
    assert peak_in_box(hm, inside, 128).item()
    assert not peak_in_box(hm, outside, 128).item()


def test_logger_writes_event_file(tmp_path):
    n = 4
    images = torch.rand(n, 3, 128, 128)
    hm = torch.rand(n, 1, 32, 32)
    boxes = torch.tensor([[10, 10, 40, 40]] * n)
    labels = torch.randint(0, 5, (n,))
    with TBLogger(tmp_path) as tb:
        tb.scalars({"loss/x": 1.0}, 0)
        tb.heatmap_panels("hm", images, hm, 0, gt_heatmaps=hm, boxes=boxes)
        tb.confusion_matrix("cm", labels, labels, 5, 0)
        tb.histograms(torch.nn.Linear(2, 2), 0)
        tb.config({"a": {"b": 1}})
    assert list(tmp_path.glob("events.out.tfevents.*"))
