import math

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from findobject.data.dataset import Sample
from findobject.models.baseline import HEATMAP_PRIOR, BaselineCNN
from findobject.training.checkpoint import load_checkpoint_info, load_model, save_checkpoint
from findobject.training.config import TrainConfig
from findobject.training.epoch import train_one_epoch

VALID = dict(run_name="t", epochs=2, batch_size=4, lr=0.001, weight_decay=0.0001, heatmap_loss_weight=0.1,
             seed=0, device="cpu", log_every_steps=1, histogram_every_epochs=1)


def _batch(image: torch.Tensor) -> Sample:
    return Sample(image, torch.zeros(2, 1, 8, 8), torch.tensor([0, 1]), torch.tensor([[0, 0, 8, 8]] * 2))


def _opt(model):
    optimizer = AdamW(model.parameters(), lr=0.001)
    return optimizer, CosineAnnealingLR(optimizer, T_max=10)


# --- config ---

def test_config_accepts_valid_values():
    assert TrainConfig(**VALID).epochs == 2


@pytest.mark.parametrize("key,value", [("epochs", 0), ("lr", -1.0), ("lr", "1e-3"), ("device", "cuda"),
                                       ("weight_decay", -0.1), ("run_name", "a/b")])
def test_config_rejects_bad_values(key, value):
    with pytest.raises(ValueError):
        TrainConfig(**{**VALID, key: value})


def test_config_rejects_unknown_key():
    with pytest.raises(TypeError):
        TrainConfig(**VALID, epochz=3)


# --- model prior ---

def test_heatmap_head_starts_at_the_empty_prior():
    bias = BaselineCNN().heatmap_head.bias
    assert torch.sigmoid(bias).item() == pytest.approx(HEATMAP_PRIOR, rel=1e-5)


# --- train_one_epoch ---

def test_epoch_runs_and_calls_on_step_once_per_batch():
    model = BaselineCNN()
    optimizer, scheduler = _opt(model)
    calls = []
    stats = train_one_epoch(model, [_batch(torch.rand(2, 3, 32, 32))] * 3, optimizer, scheduler,
                            torch.device("cpu"), 0.1, calls.append)
    assert len(calls) == 3 and math.isfinite(stats.mean_total) and 0 <= stats.class_acc <= 1


def test_nan_loss_stops_training():
    model = BaselineCNN()
    optimizer, scheduler = _opt(model)
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, [_batch(torch.full((2, 3, 32, 32), float("nan")))], optimizer, scheduler,
                        torch.device("cpu"), 0.1, lambda parts: None)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))  # bad batch never applied


# --- checkpoint ---

def test_checkpoint_round_trip_reproduces_outputs(tmp_path):
    model = BaselineCNN().eval()
    optimizer, scheduler = _opt(model)
    path = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=3, config={"a": [1, 2]}, metrics={"joint_acc": 0.5})
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    x = torch.rand(1, 3, 64, 64)
    reloaded = load_model(path, torch.device("cpu"))
    assert torch.equal(model(x).heatmap_logits, reloaded(x).heatmap_logits)
    assert not reloaded.training


def test_checkpoint_info_returns_what_was_saved(tmp_path):
    model = BaselineCNN()
    optimizer, scheduler = _opt(model)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=7, config={"train": {"lr": 0.001}},
                    metrics={"joint_acc": 0.25})
    info = load_checkpoint_info(path)
    assert (info.epoch, info.config, info.metrics) == (7, {"train": {"lr": 0.001}}, {"joint_acc": 0.25})


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt", torch.device("cpu"))
