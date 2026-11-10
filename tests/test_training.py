import math

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch import nn

from findobject.data.dataset import IGNORE_CLASS, Sample
from findobject.models.baseline import HEATMAP_PRIOR, BaselineCNN, ModelOutput
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


def test_config_num_workers_defaults_to_zero_and_accepts_positive():
    assert TrainConfig(**VALID).num_workers == 0
    assert TrainConfig(**VALID, num_workers=4).num_workers == 4


@pytest.mark.parametrize("value", [-1, 1.5, "4"])
def test_config_rejects_bad_num_workers(value):
    with pytest.raises(ValueError):
        TrainConfig(**VALID, num_workers=value)


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


class _FixedModel(nn.Module):
    """Always predicts class 0 and puts the heatmap peak in cell (0, 0), centered at scene px (2, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))  # something for the optimizer to step

    def forward(self, image: torch.Tensor) -> ModelOutput:
        b = len(image)
        heat = torch.zeros(b, 1, 8, 8)
        heat[:, 0, 0, 0] = 5.0
        logits = torch.zeros(b, 43)
        logits[:, 0] = 5.0
        return ModelOutput(heat + 0 * self.w, logits + 0 * self.w)


def test_epoch_accuracies_count_sign_scenes_only():
    # Sign rows: row 0 right on both (class 0, box holds the peak), row 1 wrong on both. Empties can never
    # be right, so counting them would give 1/4 instead of 1/2.
    batch = Sample(torch.rand(4, 3, 32, 32), torch.zeros(4, 1, 8, 8), torch.tensor([0, 1, IGNORE_CLASS, IGNORE_CLASS]),
                   torch.tensor([[0, 0, 8, 8], [16, 16, 32, 32], [0, 0, 0, 0], [0, 0, 0, 0]]))
    model = _FixedModel()
    optimizer, scheduler = _opt(model)
    stats = train_one_epoch(model, [batch], optimizer, scheduler, torch.device("cpu"), 0.1, lambda parts: None)
    assert stats.class_acc == 0.5 and stats.pointing_acc == 0.5


def test_all_empty_batch_adds_no_rows_to_the_accuracies():
    sign_batch = Sample(torch.rand(2, 3, 32, 32), torch.zeros(2, 1, 8, 8), torch.tensor([0, 1]),
                        torch.tensor([[0, 0, 8, 8], [16, 16, 32, 32]]))
    empty_batch = Sample(torch.rand(3, 3, 32, 32), torch.zeros(3, 1, 8, 8), torch.tensor([IGNORE_CLASS] * 3),
                         torch.zeros(3, 4, dtype=torch.int64))
    model = _FixedModel()
    optimizer, scheduler = _opt(model)
    stats = train_one_epoch(model, [empty_batch, sign_batch], optimizer, scheduler, torch.device("cpu"), 0.1,
                            lambda parts: None)
    assert stats.class_acc == 0.5 and stats.pointing_acc == 0.5


def test_epoch_with_no_sign_scene_raises():
    batch = Sample(torch.rand(2, 3, 32, 32), torch.zeros(2, 1, 8, 8), torch.tensor([IGNORE_CLASS] * 2),
                   torch.zeros(2, 4, dtype=torch.int64))
    model = _FixedModel()
    optimizer, scheduler = _opt(model)
    with pytest.raises(ValueError):
        train_one_epoch(model, [batch], optimizer, scheduler, torch.device("cpu"), 0.1, lambda parts: None)


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
