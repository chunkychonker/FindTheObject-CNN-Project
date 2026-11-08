"""Picks each epoch's training scenes according to the config's data.train_scenes."""

from collections.abc import Callable
from enum import Enum
from pathlib import Path

from torch.utils.data import Dataset

from findobject.data.data_config import train_synth_config
from findobject.data.dataset import Sample, SignScenes
from findobject.data.fresh import FreshScenes
from findobject.data.splits import split_sources


class TrainScenes(str, Enum):
    """data.train_scenes. TrainScenes("x") raises ValueError for an unknown value."""

    FIXED = "fixed"  # the stored train.npz every epoch (v1, v1.5)
    FRESH = "fresh"  # newly generated scenes every epoch, optionally with data.train_augment


def epoch_datasets(data_cfg: dict, seed: int) -> Callable[[int], Dataset[Sample]]:
    """Return `for_epoch(epoch) -> dataset` for 1-based epochs.

    fixed (the default when data.train_scenes is absent): the same SignScenes(processed_dir/train.npz)
    for every epoch, loaded once, here. fresh: FreshScenes of data.n_train scenes from the train
    split's signs and backgrounds, a new set each epoch, seeded by (seed, epoch).

    Raises ValueError for an unknown data.train_scenes, or if data.train_augment is set with fixed
    (stored scenes cannot be re-augmented); otherwise raises as SignScenes / split_sources /
    train_synth_config do. Errors surface here, before training starts, not at the first epoch.
    """
    mode = TrainScenes(data_cfg.get("train_scenes", TrainScenes.FIXED.value))
    if mode is TrainScenes.FIXED:
        if data_cfg.get("train_augment"):
            raise ValueError("data.train_augment needs data.train_scenes: fresh (stored scenes can't be re-augmented)")
        stored = SignScenes(Path(data_cfg["processed_dir"]) / "train.npz")
        return lambda epoch: stored

    source = split_sources(data_cfg)["train"]
    cfg = train_synth_config(data_cfg)
    n = data_cfg["n_train"]
    return lambda epoch: FreshScenes(source, cfg, n, seed, epoch)
