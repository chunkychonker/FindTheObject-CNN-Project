"""Count P6's sign labels among the top-scoring and control tiles (docs/v3a-spec.md §2.9 step 3). Pure."""

from enum import Enum

from findobject.eval.p6_sheet import N_CONTROL, N_TOP, Tile

P6_NEEDS = 10  # P6 holds if at least this many of the N_TOP tiles are A or B


class Label(Enum):
    A = "A"  # unlabeled sign outside the 43 classes, at least half its face in the window
    B = "B"  # unlabeled sign of one of the 43 classes, same half-face rule
    C = "C"  # sign-like non-sign
    D = "D"  # nothing sign-like


SIGN = frozenset({Label.A, Label.B})


def parse_labels(raw: dict[str, str], n_tiles: int) -> dict[int, Label]:
    """{"1": "A", ...} -> {1: Label.A, ...}. Raises ValueError unless the keys are exactly "1".."n_tiles" and
    every value is one of A-D (a missing or extra tile, or a typo, must not be silently dropped)."""
    expected = {str(i) for i in range(1, n_tiles + 1)}
    if set(raw) != expected:
        raise ValueError(f"labels must cover tiles 1..{n_tiles} exactly; missing {sorted(expected - set(raw))}, "
                         f"extra {sorted(set(raw) - expected)}")
    return {int(k): Label(v) for k, v in raw.items()}  # Label(v) raises ValueError for anything but A-D


def tally(labels: dict[int, Label], tiles: list[Tile]) -> dict:
    """Per group (top / control): count of each label, the A+B count, and for top whether P6 holds.
    Raises ValueError if the tiles are not N_TOP top + N_CONTROL control, or a tile has no label."""
    if sum(t.top for t in tiles) != N_TOP or sum(not t.top for t in tiles) != N_CONTROL:
        raise ValueError(f"need {N_TOP} top and {N_CONTROL} control tiles")
    missing = [t.number for t in tiles if t.number not in labels]
    if missing:
        raise ValueError(f"no label for tiles {missing}")
    out = {}
    for group, is_top in (("top", True), ("control", False)):
        got = [labels[t.number] for t in tiles if t.top == is_top]
        out[group] = {"counts": {lab.value: got.count(lab) for lab in Label},
                      "sign": sum(lab in SIGN for lab in got), "n": len(got)}
    out["p6_holds"] = out["top"]["sign"] >= P6_NEEDS
    return out
