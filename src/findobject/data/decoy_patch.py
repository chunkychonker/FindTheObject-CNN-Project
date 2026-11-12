"""Choose where a training decoy's patch is cut from: which background, and which corner (docs/v3a1-spec.md §2.1). Pure.

A decoy replaces a sign's pixels with a same-size patch of another background. Only backgrounds at least
as large as every possible patch may be cut from, so a draw can never fail mid-training; the sizes are
read once, at the edge, and passed in.
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PATCH_STREAM = 1  # fourth word of a training decoy's patch RNG key [seed, epoch, i, PATCH_STREAM]; fixed once


@dataclass(frozen=True)
class DecoySources:
    """The backgrounds a patch may be cut from. Build it with eligible_patch_sources.

    need: (width, height) no patch exceeds. index: eligible background indices, ascending, at least 2.
    sizes[k]: (width, height) of background index[k].
    """

    need: tuple[int, int]
    index: tuple[int, ...]
    sizes: tuple[tuple[int, int], ...]


def eligible_patch_sources(sizes: Sequence[tuple[int, int]], need: tuple[int, int]) -> DecoySources:
    """The backgrounds (sizes[j] is background j's (width, height)) with width >= need[0] and height >= need[1].
    Raises ValueError if fewer than 2 qualify: a scene's own background is never its patch source, so one
    is not enough."""
    keep = [j for j, (w, h) in enumerate(sizes) if w >= need[0] and h >= need[1]]
    if len(keep) < 2:
        raise ValueError(f"need >= 2 backgrounds of at least {need[0]}x{need[1]} to cut decoy patches from, got {len(keep)}")
    return DecoySources(need=(int(need[0]), int(need[1])), index=tuple(keep),
                        sizes=tuple((int(sizes[j][0]), int(sizes[j][1])) for j in keep))


def draw_patch(sources: DecoySources, scene_bg: int, size: tuple[int, int],
               rng: np.random.Generator) -> tuple[int, tuple[int, int]]:
    """(source background index, (x, y) top-left corner) for a patch of `size` (width, height).

    Draws, in order: a source uniform over the eligible backgrounds other than scene_bg; then x, then y,
    uniform over the corners that keep the patch inside the source. Never fails for a size within
    sources.need. Raises ValueError if size is not positive or exceeds sources.need (the caller computed
    need from the wrong signs).
    """
    w, h = size
    if not (1 <= w <= sources.need[0] and 1 <= h <= sources.need[1]):
        raise ValueError(f"patch size {size} outside 1..{sources.need}")
    m = len(sources.index)
    k = bisect_left(sources.index, scene_bg)
    if k < m and sources.index[k] == scene_bg:  # skip the scene's own background
        j = int(rng.integers(m - 1))
        j += j >= k
    else:
        j = int(rng.integers(m))
    src_w, src_h = sources.sizes[j]
    x = int(rng.integers(0, src_w - w + 1))
    y = int(rng.integers(0, src_h - h + 1))
    return sources.index[j], (x, y)
