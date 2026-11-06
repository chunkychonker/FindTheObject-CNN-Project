"""Seed every global RNG a training run touches."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seeds Python, NumPy and torch (weight init); returns a fresh generator for DataLoader
    shuffling. MPS kernels are not bit-for-bit deterministic, so GPU reruns still differ slightly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)
