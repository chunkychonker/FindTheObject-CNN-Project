"""Generates never-repeating training scenes on demand, one epoch's worth per instance."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from findobject.data.dataset import Sample
from findobject.data.gtsrb import load_sign
from findobject.data.splits import SceneSource
from findobject.data.synth import SynthConfig, make_scene


class FreshScenes(Dataset[Sample]):
    """Scene i of `epoch` is generated when asked for, from an RNG seeded with (seed, epoch, i).

    So a scene depends only on its own (seed, epoch, i): identical in any DataLoader worker, in any
    order, with any num_workers; a different epoch gives different scenes. Signs and backgrounds are
    drawn uniformly from `source`, as make_dataset.generate does for the stored splits.

    Holds only file paths, so it pickles cheaply into spawned DataLoader workers (unlike SignScenes).
    """

    def __init__(self, source: SceneSource, cfg: SynthConfig, n: int, seed: int, epoch: int) -> None:
        """Raises ValueError if n < 1, seed or epoch is negative, or source has no signs or backgrounds."""
        if n < 1 or seed < 0 or epoch < 0:
            raise ValueError(f"need n >= 1, seed >= 0, epoch >= 0; got {n}, {seed}, {epoch}")
        if not source.signs or not source.backgrounds:
            raise ValueError("source needs at least one sign and one background")
        self._source = source
        self._cfg = cfg
        self._n = n
        self._seed = seed
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> Sample:
        """Raises IndexError if i is out of range; OSError if a sign or background file is unreadable."""
        if not 0 <= i < self._n:
            raise IndexError(f"scene {i} out of range for {self._n}")
        rng = np.random.default_rng([self._seed, self._epoch, i])
        signs, backgrounds = self._source
        sign = signs[rng.integers(len(signs))]
        with Image.open(backgrounds[rng.integers(len(backgrounds))]) as bg:
            scene = make_scene(load_sign(sign), bg, rng, self._cfg)
        return Sample(
            image=torch.from_numpy(scene.image).permute(2, 0, 1).float().div_(255),  # HWC -> CHW
            heatmap=torch.from_numpy(scene.heatmap).float().unsqueeze(0),
            class_id=torch.tensor(sign.class_id, dtype=torch.int64),
            box=torch.tensor(scene.box, dtype=torch.int64),
        )
