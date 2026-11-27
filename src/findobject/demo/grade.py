"""Grade the demo's answer against known signs, with the project's own matching rules (eval/photo_match). Pure."""

from collections.abc import Sequence
from dataclasses import dataclass

from findobject.eval.photo_match import FalsePositive, match_photo
from findobject.eval.photo_truth import TruthSign
from findobject.inference.detection import Detection


@dataclass(frozen=True)
class AnswerGrade:
    found: tuple[int | None, ...]  # per detection: the gt_index it found (point in box, right class), or None
    fp_kind: tuple[FalsePositive | None, ...]  # per detection; None exactly where found is not None
    missed: frozenset[int]  # gt_index of every sign no detection found


@dataclass(frozen=True)
class SignOutcome:
    found_by: Detection | None  # the detection that found the sign
    located_by: Detection | None  # the detection pointing at it, class ignored (set whenever found_by is)


def grade_answer(dets: Sequence[Detection], truth: Sequence[TruthSign]) -> AnswerGrade:
    """Match dets (in detection_order, as detect() returns them) to truth exactly as the graded v3b numbers
    were. Raises ValueError as match_photo does if dets are out of order."""
    m = match_photo(dets, truth)
    hit = {g for g in m.found if g is not None}
    return AnswerGrade(found=m.found, fp_kind=m.fp_kind, missed=frozenset(s.gt_index for s in truth) - hit)


def sign_outcome(dets: Sequence[Detection], truth: Sequence[TruthSign], gt_index: int) -> SignOutcome:
    """What happened to one sign: which detection found it and which pointed at it ignoring class.
    Raises ValueError as match_photo does."""
    m = match_photo(dets, truth)
    found = next((d for d, g in zip(dets, m.found) if g == gt_index), None)
    located = next((d for d, g in zip(dets, m.located) if g == gt_index), None)
    return SignOutcome(found_by=found, located_by=located)
