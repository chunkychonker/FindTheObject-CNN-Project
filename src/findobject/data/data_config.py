"""Parses the YAML `data:` section into typed scene-generation settings."""

from dataclasses import replace

from findobject.data.sign_augment import SignAugment
from findobject.data.synth import SynthConfig

_PAIRS = ("contrast", "saturation", "occlusion_area")  # YAML lists -> tuples


def synth_config(data_cfg: dict) -> SynthConfig:
    """The shared generation settings (data.synth), used for every stored split. No sign augmentation.

    Raises KeyError if data.synth is missing, TypeError for unknown keys, ValueError for bad values.
    """
    synth = data_cfg["synth"]
    return SynthConfig(**{**synth, "brightness": tuple(synth["brightness"])})


def train_synth_config(data_cfg: dict) -> SynthConfig:
    """data.synth plus data.train_augment (if present) as the sign augmentation, for fresh training scenes.

    Raises as synth_config does; TypeError / ValueError for a bad data.train_augment.
    """
    aug = data_cfg.get("train_augment") or {}
    augment = SignAugment(**{k: tuple(v) if k in _PAIRS else v for k, v in aug.items()})
    return replace(synth_config(data_cfg), sign_augment=augment)
