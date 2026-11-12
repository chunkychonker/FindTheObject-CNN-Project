"""Picks each epoch's training scenes according to the config's data.train_scenes."""

from collections.abc import Callable
from enum import Enum
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from findobject.data.data_config import train_synth_config
from findobject.data.dataset import Sample, SignScenes
from findobject.data.decoy_patch import DecoySources, eligible_patch_sources
from findobject.data.fresh import FreshScenes
from findobject.data.splits import SceneSource, split_sources
from findobject.data.synth_negatives import roi_size


class TrainScenes(str, Enum):
    """data.train_scenes. TrainScenes("x") raises ValueError for an unknown value."""

    FIXED = "fixed"  # the stored train.npz every epoch (v1, v1.5)
    FRESH = "fresh"  # newly generated scenes every epoch, optionally with data.train_augment


def epoch_datasets(data_cfg: dict, seed: int) -> Callable[[int], Dataset[Sample]]:
    """Return `for_epoch(epoch) -> dataset` for 1-based epochs.

    fixed (the default when data.train_scenes is absent): the same SignScenes(processed_dir/train.npz)
    for every epoch, loaded once, here. fresh: FreshScenes of data.n_train scenes from the train
    split's signs and backgrounds, a new set each epoch, seeded by (seed, epoch). The last
    data.n_train_empty of them (default 0) hold no sign (v3a); the data.n_train_decoy before those
    (default 0) are decoys (v3a.1). n_train stays the epoch length.

    Raises ValueError for an unknown data.train_scenes, if data.train_augment or a nonzero
    data.n_train_empty or data.n_train_decoy is set with fixed (stored scenes cannot change), if either
    count is not an int >= 0, if n_train_empty + n_train_decoy >= n_train, or if fewer than 2 train
    backgrounds are big enough to cut every decoy patch from; otherwise raises as SignScenes /
    split_sources / train_synth_config do, and OSError for an unreadable background header. Errors
    surface here, before training starts, not at the first epoch.
    """
    mode = TrainScenes(data_cfg.get("train_scenes", TrainScenes.FIXED.value))
    n_empty = data_cfg.get("n_train_empty", 0)
    n_decoy = data_cfg.get("n_train_decoy", 0)
    if mode is TrainScenes.FIXED:
        if n_empty:
            raise ValueError("data.n_train_empty needs data.train_scenes: fresh (stored scenes can't gain empties)")
        if n_decoy:
            raise ValueError("data.n_train_decoy needs data.train_scenes: fresh (stored scenes can't gain decoys)")
        if data_cfg.get("train_augment"):
            raise ValueError("data.train_augment needs data.train_scenes: fresh (stored scenes can't be re-augmented)")
        stored = SignScenes(Path(data_cfg["processed_dir"]) / "train.npz")
        return lambda epoch: stored

    n = data_cfg["n_train"]
    if not isinstance(n_empty, int) or isinstance(n_empty, bool) or not 0 <= n_empty < n:
        raise ValueError(f"data.n_train_empty must be an int in [0, n_train={n}), got {n_empty!r}")
    if not _is_count(n_decoy) or n_empty + n_decoy >= n:
        raise ValueError(f"data.n_train_decoy must be an int >= 0 with n_train_empty + n_train_decoy < n_train={n}, "
                         f"got {n_decoy!r}")
    source = split_sources(data_cfg)["train"]
    cfg = train_synth_config(data_cfg)
    patches = decoy_sources(source) if n_decoy else None
    return lambda epoch: FreshScenes(source, cfg, n, seed, epoch, n_empty, n_decoy, patches)


def decoy_sources(source: SceneSource) -> DecoySources:
    """The backgrounds of `source` big enough to cut any of its signs' ROI-size patches from. Reads each
    background's header (not its pixels). Raises ValueError if fewer than 2 qualify; OSError if a header is
    unreadable."""
    sizes = [roi_size(s) for s in source.signs]
    need = (max(w for w, _ in sizes), max(h for _, h in sizes))
    bg_sizes = []
    for path in source.backgrounds:
        with Image.open(path) as img:
            bg_sizes.append(img.size)
    return eligible_patch_sources(bg_sizes, need)


def _is_count(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0
