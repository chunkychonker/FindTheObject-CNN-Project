"""Write the plain-language text shown under each demo stage. Pure."""

from collections.abc import Sequence

from findobject.data.class_names import CLASS_NAMES
from findobject.data.synth import SynthConfig
from findobject.demo.draw import SCALE_LABELS
from findobject.demo.fit import FittedPhoto, to_original
from findobject.demo.grade import AnswerGrade, SignOutcome
from findobject.eval.photo_truth import TruthSign
from findobject.inference.detection import PYRAMID_SCALES, Detection
from findobject.inference.peaks import MIN_PEAK_SCORE

TRAIN_SIGN_RANGE = (SynthConfig().sign_min, SynthConfig().sign_max)  # (16, 48) px: sign sizes the model trained on
TABLE_HEADERS = ["class", "score", "class prob", "scale", "x (px)", "y (px)"]


def fit_caption(fitted: FittedPhoto) -> str:
    (w, h), (cw, ch) = fitted.original_size, fitted.content_size
    if (w, h) == (cw, ch):
        return f"Photo: {w}×{h} px, used as is."
    return f"Photo: {w}×{h} px, shrunk to {cw}×{ch} so signs come out near the sizes the model was graded on."


def peaks_caption(n_above: int, n_raw: int, tau: float) -> str:
    return (f"**{n_above}** peaks score ≥ {tau:.2f} across the three scales (of {n_raw} peaks ≥ {MIN_PEAK_SCORE}). "
            "A peak is a heatmap cell higher than its 8 neighbours. "
            "Ring colour = the scale it came from: cyan ×1, yellow ×½, magenta ×¼.")


def suppression_caption(n_kept: int, n_above: int) -> str:
    return (f"Suppression keeps **{n_kept}** of {n_above}. Starting from the highest score, each kept point clears "
            "the square drawn around it (3 heatmap cells at its own scale); weaker points inside are dropped as "
            "repeats of the same sign. One sign often peaks at two scales, so this is where duplicates go.")


def answer_caption(dets: Sequence[Detection], truth: Sequence[TruthSign] | None, grade: AnswerGrade | None) -> str:
    if truth is None or grade is None:
        return f"**{len(dets)}** detection(s). This photo has no labels, so there's nothing to grade against."
    n_found = sum(g is not None for g in grade.found)
    n_wrong = len(dets) - n_found
    text = f"Found **{n_found} of {len(truth)}** labeled signs, with **{n_wrong}** false positive(s)."
    missed = sorted((s for s in truth if s.gt_index in grade.missed), key=lambda s: s.long_side)
    if missed:
        text += " Missed (orange): " + ", ".join(f"{CLASS_NAMES[s.class_id]} ({s.long_side} px)" for s in missed) + "."
    if n_wrong:
        text += (" Red rings found no labeled sign. Some are real signs the dataset never labeled, "
                 "like parking signs, which the 43 classes don't include.")
    return text


def detection_rows(dets: Sequence[Detection], fitted: FittedPhoto) -> list[list]:
    """One table row per detection, with its point in the original photo's pixels."""
    rows = []
    for d in dets:
        x, y = to_original(d.x, d.y, fitted)
        rows.append([CLASS_NAMES[d.class_id], round(d.score, 3), round(d.class_prob, 3),
                     f"×{SCALE_LABELS[d.scale]}", round(x), round(y)])
    return rows


def pasted_caption(outcome: SignOutcome, class_id: int, long_side: int) -> str:
    """What happened to the pasted sign, and the size each pyramid scale shows it at."""
    lo, hi = TRAIN_SIGN_RANGE
    sizes = ", ".join(f"×{SCALE_LABELS[s]} → {long_side * s:.0f} px" for s in PYRAMID_SCALES)
    seen = f"Each scale shows your {long_side} px sign at: {sizes}. The model trained on signs {lo}-{hi} px."
    if outcome.found_by is not None:
        d = outcome.found_by
        return (f"✅ **Found** {CLASS_NAMES[class_id]} at scale ×{SCALE_LABELS[d.scale]} "
                f"(score {d.score:.2f}), where it looked {long_side * d.scale:.0f} px. {seen}")
    if outcome.located_by is not None:
        d = outcome.located_by
        return (f"🟠 **Pointed at it but named it wrong**: called it {CLASS_NAMES[d.class_id]} "
                f"(score {d.score:.2f}). {seen}")
    return f"❌ **Missed.** No detection pointed at your {CLASS_NAMES[class_id]} sign. {seen}"
