"""Serve one generated split (data/processed/{train,test}.npz) as PyTorch tensors."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from findobject.data.gtsrb import NUM_CLASSES

_KEYS = ("images", "class_ids", "boxes", "heatmaps")

IGNORE_CLASS = -100  # class_id of a scene with no sign; equals torch cross_entropy's default ignore_index


class Sample(NamedTuple):
    """One scene. DataLoader's default collate batches each field (leading dim B)."""

    image: Tensor  # float32 [3, S, S] in [0, 1]
    heatmap: Tensor  # float32 [1, H, H] in [0, 1]
    class_id: Tensor  # int64 scalar in [0, NUM_CLASSES), or IGNORE_CLASS for a scene with no sign
    box: Tensor  # int64 [4] x1, y1, x2, y2 in scene px; x2/y2 exclusive

    def to(self, device: torch.device) -> "Sample":
        return Sample(*(t.to(device) for t in self))


def has_sign(class_id: Tensor) -> Tensor:
    """Bool tensor of class_id's shape: True where the scene holds a sign. The only test for "has a sign",
    so it can never disagree with the stored class id."""
    return class_id != IGNORE_CLASS


class SignScenes(Dataset[Sample]):
    """Every scene of one split, held in RAM in its stored dtypes (uint8 images, float16
    heatmaps); conversion to float32 happens per item.

    Use num_workers=0 on macOS: spawned DataLoader workers would each get a pickled copy
    of the arrays (~1 GB for train).
    """

    def __init__(self, npz_path: Path) -> None:
        """Read and validate all arrays.

        Raises FileNotFoundError if npz_path does not exist, KeyError if an expected array
        is missing, ValueError if shapes, dtypes, lengths or class ids are inconsistent.
        """
        with np.load(npz_path) as d:
            missing = [k for k in _KEYS if k not in d]
            if missing:
                raise KeyError(f"{npz_path}: missing arrays {missing}")
            images, class_ids, boxes, heatmaps = (d[k] for k in _KEYS)
        _validate(npz_path, images, class_ids, boxes, heatmaps)
        self._images = images
        self._class_ids = class_ids
        self._boxes = boxes
        self._heatmaps = heatmaps

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int) -> Sample:
        """Raises IndexError if i is out of range. Returned tensors never alias the stored arrays."""
        return Sample(
            image=torch.from_numpy(self._images[i]).permute(2, 0, 1).float().div_(255),  # HWC -> CHW
            heatmap=torch.from_numpy(self._heatmaps[i]).float().unsqueeze(0),
            class_id=torch.tensor(self._class_ids[i], dtype=torch.int64),
            box=torch.from_numpy(self._boxes[i]).long(),
        )


def _validate(path: Path, images: np.ndarray, class_ids: np.ndarray, boxes: np.ndarray, heatmaps: np.ndarray) -> None:
    def fail(msg: str) -> None:
        raise ValueError(f"{path}: {msg}")

    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[3] != 3 or images.shape[1] != images.shape[2]:
        fail(f"images must be uint8 [N, S, S, 3], got {images.dtype} {images.shape}")
    if not np.issubdtype(heatmaps.dtype, np.floating) or heatmaps.ndim != 3 or heatmaps.shape[1] != heatmaps.shape[2]:
        fail(f"heatmaps must be float [N, H, H], got {heatmaps.dtype} {heatmaps.shape}")
    if images.shape[1] % heatmaps.shape[1]:
        fail(f"scene size {images.shape[1]} is not a multiple of heatmap size {heatmaps.shape[1]}")
    if not np.issubdtype(class_ids.dtype, np.integer) or class_ids.ndim != 1:
        fail(f"class_ids must be integer [N], got {class_ids.dtype} {class_ids.shape}")
    if not np.issubdtype(boxes.dtype, np.integer) or boxes.ndim != 2 or boxes.shape[1] != 4:
        fail(f"boxes must be integer [N, 4], got {boxes.dtype} {boxes.shape}")
    lengths = {len(images), len(class_ids), len(boxes), len(heatmaps)}
    if len(lengths) != 1:
        fail(f"arrays disagree on N: {sorted(lengths)}")
    if len(class_ids) and not (class_ids.min() >= 0 and class_ids.max() < NUM_CLASSES):
        fail(f"class ids must be in [0, {NUM_CLASSES})")
