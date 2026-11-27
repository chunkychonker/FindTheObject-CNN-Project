"""Turn a Trace and a threshold into everything one demo tab shows (docs/demo-spec.md §2). Pure: never calls
the model, so moving the threshold slider only re-runs these (A5)."""

from dataclasses import dataclass

from PIL import Image

from findobject.demo.captions import (answer_caption, detection_rows, fit_caption, pasted_caption, peaks_caption,
                                      suppression_caption)
from findobject.demo.draw import SCALE_LABELS, draw_answer, draw_heatmap, draw_points, draw_scales
from findobject.demo.grade import grade_answer, sign_outcome
from findobject.demo.paste import center_from_fractions, paste_sign, pasted_size
from findobject.demo.trace import Trace, above
from findobject.eval.photo_truth import TruthSign
from findobject.models.baseline import STRIDE

PASTED_GT_INDEX = 10_000  # gt_index of the pasted sign; above any real gt.txt row index (GTSDB photos have < 10)

Gallery = list[tuple[Image.Image, str]]


@dataclass(frozen=True)
class WatchView:
    scales: Image.Image
    fit_text: str
    heatmaps: Gallery
    peaks: Image.Image
    peaks_text: str
    suppression: Image.Image
    suppression_text: str
    answer: Image.Image
    answer_text: str
    rows: list[list]


@dataclass(frozen=True)
class Placed:
    image: Image.Image  # the background with the sign pasted on
    truth: tuple[TruthSign, ...]  # the background's labeled signs, then the pasted one (gt_index PASTED_GT_INDEX)
    class_id: int
    long_side: int  # as pasted, after clamping to fit
    asked_long_side: int  # as requested


@dataclass(frozen=True)
class FoolView:
    answer: Image.Image
    text: str
    heatmaps: Gallery


def heatmap_gallery(tr: Trace) -> Gallery:
    """One overlay per pyramid scale, captioned with the input size the model saw and its heatmap size."""
    w, h = tr.fitted.image.size
    out = []
    for v in tr.views:
        rows, cols = v.heatmap.shape
        cap = (f"×{SCALE_LABELS[v.scale]}: the model sees {round(w * v.scale)}×{round(h * v.scale)} px "
               f"→ {cols}×{rows} cells, one per {STRIDE / v.scale:.0f} photo px")
        out.append((draw_heatmap(tr.fitted.content, v), cap))
    return out


def watch_view(tr: Trace, truth: tuple[TruthSign, ...] | None, tau: float) -> WatchView:
    """Every stage of tab 1 at threshold tau. The answer is tr.result.kept filtered to score >= tau, which is
    exactly what detect() answers at that threshold (A1).
    Raises ValueError if truth is given for a photo that was shrunk (its boxes would be in the wrong px)."""
    if truth is not None and tr.fitted.content_size != tr.fitted.original_size:
        raise ValueError("truth boxes are in original px, but this photo was shrunk to fit")
    photo = tr.fitted.content
    peaks, answer = above(tr.result.raw, tau), above(tr.result.kept, tau)
    grade = grade_answer(answer, truth) if truth is not None else None
    return WatchView(
        scales=draw_scales(photo), fit_text=fit_caption(tr.fitted), heatmaps=heatmap_gallery(tr),
        peaks=draw_points(photo, peaks), peaks_text=peaks_caption(len(peaks), len(tr.result.raw), tau),
        suppression=draw_points(photo, answer, show_suppression=True),
        suppression_text=suppression_caption(len(answer), len(peaks)),
        answer=draw_answer(photo, answer, truth, grade), answer_text=answer_caption(answer, truth, grade),
        rows=detection_rows(answer, tr.fitted))


def place_sign(background: Image.Image, background_truth: tuple[TruthSign, ...], sign: Image.Image,
               class_id: int, long_side: int, fx: float, fy: float) -> Placed:
    """Paste the sign on the background with its long side clamped to the background's shorter side, so every
    slider value fits. background must already be fitted (no bigger than FIT_MAX) so the trace won't shrink it.
    Raises ValueError for long_side < 1 or fractions outside [0, 1]."""
    side = min(long_side, *background.size)
    size = pasted_size(sign.size, side)
    pasted = paste_sign(background, sign, center_from_fractions(background.size, size, fx, fy), side)
    new = TruthSign(gt_index=PASTED_GT_INDEX, box=pasted.box, class_id=class_id,
                    long_side=max(size), stacked=False)
    return Placed(image=pasted.image, truth=(*background_truth, new), class_id=class_id,
                  long_side=max(size), asked_long_side=long_side)


def fool_view(tr: Trace, placed: Placed, tau: float) -> FoolView:
    """Tab 2 at threshold tau: the graded answer and what happened to the pasted sign.
    Raises ValueError if the trace isn't of placed.image at its own size."""
    if tr.fitted.content_size != placed.image.size or tr.fitted.original_size != placed.image.size:
        raise ValueError("the trace must be of the placed image, unshrunk")
    answer = above(tr.result.kept, tau)
    grade = grade_answer(answer, placed.truth)
    text = pasted_caption(sign_outcome(answer, placed.truth, PASTED_GT_INDEX), placed.class_id, placed.long_side)
    if placed.long_side < placed.asked_long_side:
        text += f" (Asked for {placed.asked_long_side} px; capped at {placed.long_side} to fit the photo.)"
    return FoolView(answer=draw_answer(tr.fitted.content, answer, placed.truth, grade), text=text,
                    heatmaps=heatmap_gallery(tr))
