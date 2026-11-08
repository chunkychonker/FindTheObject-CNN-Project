"""Typed, validated training settings (the `train:` section of a config YAML)."""

from dataclasses import dataclass
from typing import Literal

DEVICES = ("auto", "mps", "cpu")


@dataclass(frozen=True)
class TrainConfig:
    """Raises TypeError on unknown/missing keys (dataclass init), ValueError on bad values."""

    run_name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    heatmap_loss_weight: float
    seed: int
    device: Literal["auto", "mps", "cpu"]
    log_every_steps: int
    histogram_every_epochs: int
    num_workers: int = 0  # DataLoader worker processes for training scenes; 0 = load in the main process

    def __post_init__(self) -> None:
        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be non-empty and contain no '/'")
        for name in ("epochs", "batch_size", "log_every_steps", "histogram_every_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        for name in ("lr", "heatmap_loss_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a number > 0, got {value!r}")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError(f"weight_decay must be a number >= 0, got {self.weight_decay!r}")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(f"num_workers must be an int >= 0, got {self.num_workers!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
