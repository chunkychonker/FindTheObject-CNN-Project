import math

import pytest
import torch

from findobject.data.dataset import Sample
from findobject.data.synth import gaussian_heatmap
from findobject.models.baseline import HEATMAP_PRIOR, ModelOutput
from findobject.training.losses import detection_loss

B, H, C = 2, 32, 43


def _batch() -> Sample:
    hm = torch.stack([torch.from_numpy(gaussian_heatmap(c, H, 1.0)) for c in [(5.0, 9.0), (20.5, 11.5)]])
    return Sample(
        image=torch.zeros(B, 3, 128, 128),
        heatmap=hm.unsqueeze(1),
        class_id=torch.tensor([3, 40]),
        box=torch.tensor([[0, 0, 10, 10]] * B),
    )


def _out(heatmap_logits: torch.Tensor, class_logits: torch.Tensor | None = None) -> ModelOutput:
    return ModelOutput(heatmap_logits, torch.zeros(B, C) if class_logits is None else class_logits)


def test_uniform_class_guess_costs_ln_43():
    parts = detection_loss(_out(torch.zeros(B, 1, H, H)), _batch(), heatmap_weight=0.1)
    assert parts.class_loss.item() == pytest.approx(math.log(43), rel=1e-6)


def test_predicting_the_target_exactly_reaches_the_floor():
    batch = _batch()
    t = batch.heatmap.clamp(1e-7, 1 - 1e-7)
    parts = detection_loss(_out(torch.log(t / (1 - t))), batch, heatmap_weight=0.1)
    assert parts.heatmap_loss.item() == pytest.approx(parts.heatmap_floor.item(), abs=1e-2)
    assert parts.heatmap_floor.item() > 1  # soft targets: the floor is well above 0


def test_prior_bias_loss_matches_hand_formula_and_total_combines_parts():
    batch = _batch()
    logit = math.log(HEATMAP_PRIOR / (1 - HEATMAP_PRIOR))
    parts = detection_loss(_out(torch.full((B, 1, H, H), logit)), batch, heatmap_weight=0.1)
    t = batch.heatmap.double()
    expected = -(t * math.log(HEATMAP_PRIOR) + (1 - t) * math.log(1 - HEATMAP_PRIOR)).sum() / B
    assert parts.heatmap_loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert parts.total.item() == pytest.approx(parts.class_loss.item() + 0.1 * parts.heatmap_loss.item(), rel=1e-6)


def test_heatmap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        detection_loss(_out(torch.zeros(B, 1, 16, 16)), _batch(), heatmap_weight=0.1)
