"""Place one sign on a photo at a chosen size and position (docs/demo-spec.md §2, tab 2). Pure."""

from dataclasses import dataclass

from PIL import Image

from findobject.data.gtsdb import Box


@dataclass(frozen=True)
class Pasted:
    image: Image.Image  # the photo with the sign on it
    box: Box  # x1, y1, x2, y2 of the sign in the photo; x2/y2 exclusive


def paste_sign(photo: Image.Image, sign: Image.Image, center: tuple[float, float], long_side: int) -> Pasted:
    """Resize the sign (PIL bilinear, shape kept) so its longer side is long_side px, and paste it opaque with
    its center at `center` (rounded to the nearest whole pixel). The input photo is not changed.

    Raises ValueError if long_side < 1 or the sign would not lie wholly inside the photo.
    """
    w, h = pasted_size(sign.size, long_side)
    x1, y1 = round(center[0] - w / 2), round(center[1] - h / 2)
    if x1 < 0 or y1 < 0 or x1 + w > photo.width or y1 + h > photo.height:
        raise ValueError(f"a {w}x{h} sign centered at {center} doesn't fit in a {photo.width}x{photo.height} photo")
    out = photo.convert("RGB")  # convert always returns a copy
    out.paste(sign.convert("RGB").resize((w, h), Image.BILINEAR), (x1, y1))
    return Pasted(image=out, box=(x1, y1, x1 + w, y1 + h))


def pasted_size(sign_size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """(w, h) the sign will have once its longer side is long_side px, shape kept, each side at least 1.
    Raises ValueError if long_side < 1."""
    if long_side < 1:
        raise ValueError(f"long_side must be >= 1, got {long_side}")
    sw, sh = sign_size
    k = long_side / max(sw, sh)
    return max(1, round(sw * k)), max(1, round(sh * k))


def center_from_fractions(photo_size: tuple[int, int], sign_size: tuple[int, int],
                          fx: float, fy: float) -> tuple[float, float]:
    """Slider positions (0 = sign touches the left/top edge, 1 = the right/bottom edge) -> the sign's center,
    so every slider value keeps the whole sign on the photo. Raises ValueError if the sign is bigger than the
    photo or a fraction is outside [0, 1]."""
    (pw, ph), (w, h) = photo_size, sign_size
    if w > pw or h > ph:
        raise ValueError(f"a {w}x{h} sign doesn't fit in a {pw}x{ph} photo")
    if not (0 <= fx <= 1 and 0 <= fy <= 1):
        raise ValueError(f"fractions must be in [0, 1], got ({fx}, {fy})")
    return w / 2 + fx * (pw - w), h / 2 + fy * (ph - h)
