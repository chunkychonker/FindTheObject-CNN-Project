"""Baseline CNN: a shared conv trunk feeding a sign-center heatmap head and a sign-class head."""

import math
from typing import NamedTuple

from torch import Tensor, nn

from findobject.data.gtsrb import NUM_CLASSES

STRIDE = 4  # input px per heatmap cell (two 2x2 max-pools); matches data/synth.py at 128 -> 32
FEATURE_CHANNELS = 128  # length of the trunk's description of each heatmap cell
# Initial per-cell sign probability. Starting near "empty" (instead of sigmoid(0) = 0.5) cuts the
# initial heatmap loss from ~710 to ~39, so early gradients don't just shout "most cells are empty".
HEATMAP_PRIOR = 0.01


class ModelOutput(NamedTuple):
    heatmap_logits: Tensor  # [B, 1, H/4, W/4], raw scores; sigmoid -> [0, 1]
    class_logits: Tensor  # [B, num_classes], raw scores; softmax -> probabilities


class BaselineCNN(nn.Module):
    """Fully convolutional, so any H, W divisible by STRIDE works.

    Each heatmap cell has a 64 px receptive field (stage 3 uses dilations 1, 1, 2, 2), enough
    to see a whole sign up to 48 px. The class head averages trunk features over the whole
    scene (global average pooling).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.trunk = build_trunk()
        self.heatmap_head = build_heatmap_head()
        self.class_head = nn.Linear(FEATURE_CHANNELS, num_classes)

    def forward(self, images: Tensor) -> ModelOutput:
        """images: float [B, 3, H, W] in [0, 1]. Raises ValueError as check_images does."""
        check_images(images)
        features = self.trunk(images)  # [B, 128, H/4, W/4]
        return ModelOutput(
            heatmap_logits=self.heatmap_head(features),
            class_logits=self.class_head(features.mean(dim=(2, 3))),
        )


def build_trunk() -> nn.Sequential:
    """Conv trunk: [B, 3, H, W] -> [B, FEATURE_CHANNELS, H/STRIDE, W/STRIDE]."""
    return nn.Sequential(
        # stage 1: full resolution
        _conv(3, 32), _conv(32, 32), nn.MaxPool2d(2),
        # stage 2: 1/2 resolution
        _conv(32, 64), _conv(64, 64), nn.MaxPool2d(2),
        # stage 3: 1/4 resolution = heatmap grid; dilation widens the view without pooling
        _conv(64, 128), _conv(128, 128), _conv(128, 128, dilation=2), _conv(128, FEATURE_CHANNELS, dilation=2),
    )


def build_heatmap_head() -> nn.Conv2d:
    """1x1 conv scoring each cell's features as 'sign center here', biased to start at HEATMAP_PRIOR."""
    head = nn.Conv2d(FEATURE_CHANNELS, 1, kernel_size=1)
    nn.init.constant_(head.bias, math.log(HEATMAP_PRIOR / (1 - HEATMAP_PRIOR)))
    return head


def check_images(images: Tensor) -> None:
    """Raises ValueError if images is not 4D with 3 channels, or H or W is not divisible
    by STRIDE (heatmap cells would no longer line up with scene pixels)."""
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(images.shape)}")
    if images.shape[2] % STRIDE or images.shape[3] % STRIDE:
        raise ValueError(f"H and W must be divisible by {STRIDE}, got {tuple(images.shape[2:])}")


def _conv(c_in: int, c_out: int, dilation: int = 1) -> nn.Sequential:
    """Size-preserving 3x3 conv -> BatchNorm -> ReLU. No conv bias: BatchNorm's shift replaces it."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )
