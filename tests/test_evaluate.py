import pytest
import torch
from torch import nn

from findobject.data.dataset import Sample
from findobject.eval.evaluate import evaluate
from findobject.models.baseline import ModelOutput

# 4 scenes, 128 px, 32x32 heatmap (stride 4). All true boxes are [40, 40, 60, 60], center (50, 50).
BOXES = torch.tensor([[40, 40, 60, 60]] * 4)
CLASSES = torch.tensor([1, 2, 3, 4])
PRED_CLASSES = torch.tensor([1, 2, 0, 4])  # 3 of 4 right (scene 2 wrong)
PEAK_CELLS = [(12, 12), (12, 12), (12, 12), (0, 0)]  # cell (12,12) is centered at (50, 50); (0,0) at (2, 2)


class FixedOutputs(nn.Module):
    """Ignores its input; returns pre-set outputs for the single batch it is given."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        hm = torch.full((4, 1, 32, 32), -10.0)
        for i, (r, c) in enumerate(PEAK_CELLS):
            hm[i, 0, r, c] = 10.0
        return ModelOutput(hm, nn.functional.one_hot(PRED_CLASSES, 43).float() * 10)


def _loader():
    return [Sample(torch.zeros(4, 3, 128, 128), torch.zeros(4, 1, 32, 32), CLASSES, BOXES)]


def test_metrics_match_hand_counts():
    r = evaluate(FixedOutputs(), _loader(), torch.device("cpu"), heatmap_weight=0.1)
    assert r.class_acc == pytest.approx(0.75)
    assert r.pointing_acc == pytest.approx(0.75)
    assert r.joint_acc == pytest.approx(0.5)  # scenes 0 and 1 only
    far = ((50 - 2) ** 2 * 2) ** 0.5  # scene 3: peak at (2, 2), true center (50, 50)
    assert r.center_err_mean == pytest.approx(far / 4)
    assert r.center_err_median == pytest.approx(0.0)
    assert r.y_pred.tolist() == PRED_CLASSES.tolist() and r.y_true.tolist() == CLASSES.tolist()
    assert set(r.scalars()) == {"class_loss", "heatmap_loss", "heatmap_excess", "class_acc", "pointing_acc",
                                "joint_acc", "center_err_mean", "center_err_median"}


def test_empty_loader_raises():
    with pytest.raises(ValueError):
        evaluate(FixedOutputs(), [], torch.device("cpu"), heatmap_weight=0.1)
