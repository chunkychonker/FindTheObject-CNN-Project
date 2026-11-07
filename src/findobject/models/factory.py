"""Model architectures by name, so configs and checkpoints can say which one to build."""

from enum import Enum

from torch import nn

from findobject.models.baseline import BaselineCNN
from findobject.models.peak import PeakCNN


class Arch(str, Enum):
    """The `model.arch` value in a config. Arch("x") raises ValueError for an unknown name."""

    BASELINE = "baseline"  # class from the average of all cells (v1)
    PEAK = "peak"  # class from the cells around the heatmap peak


_BUILDERS = {Arch.BASELINE: BaselineCNN, Arch.PEAK: PeakCNN}


def build_model(arch: Arch) -> nn.Module:
    """A fresh, untrained model; its forward returns models.baseline.ModelOutput."""
    return _BUILDERS[arch]()
