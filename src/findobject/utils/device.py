"""Choose the torch device to run on."""

import torch


def pick_device(preference: str) -> torch.device:
    """'auto' -> MPS if available, else CPU. 'mps' -> MPS or RuntimeError if unavailable.
    'cpu' -> CPU. Raises ValueError on any other value."""
    if preference not in ("auto", "mps", "cpu"):
        raise ValueError(f"unknown device preference {preference!r}")
    mps = torch.backends.mps.is_available()
    if preference == "mps" and not mps:
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device("mps" if mps and preference != "cpu" else "cpu")
