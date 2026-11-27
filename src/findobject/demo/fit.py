"""Shrink and pad any photo to a size detect() accepts, and map points back to the original (docs/demo-spec.md §2). Pure."""

import math
from dataclasses import dataclass

from PIL import Image

from findobject.data.gtsdb import PHOTO_SIZE
from findobject.inference.detection import PYRAMID_SCALES
from findobject.models.baseline import STRIDE

FIT_MAX = PHOTO_SIZE  # (1360, 800): the photo size the model was graded on
FIT_MULTIPLE = round(STRIDE / min(PYRAMID_SCALES))  # 16: every pyramid scale must give whole heatmap cells


@dataclass(frozen=True)
class FittedPhoto:
    image: Image.Image  # RGB; width and height are multiples of FIT_MULTIPLE; padding is black, right and bottom
    content_size: tuple[int, int]  # (w, h) of the shrunk photo inside image, top-left aligned
    original_size: tuple[int, int]  # (w, h) of the photo as given

    @property
    def content(self) -> Image.Image:
        """The shrunk photo without its padding."""
        return self.image.crop((0, 0, *self.content_size))

    def in_content(self, x: float, y: float) -> bool:
        """The point (fitted px) lies on the photo, not on the padding."""
        return 0 <= x < self.content_size[0] and 0 <= y < self.content_size[1]


def fit_photo(img: Image.Image) -> FittedPhoto:
    """Convert to RGB, shrink (PIL bilinear, never enlarge) to fit inside FIT_MAX keeping the shape, then pad
    right and bottom with black up to the next multiple of FIT_MULTIPLE. A FIT_MAX photo comes back unchanged.

    Raises ValueError for an image with a zero side (PIL can't make one, but a caller might pass a view).
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w == 0 or h == 0:
        raise ValueError(f"image has a zero side: {rgb.size}")
    factor = min(1.0, FIT_MAX[0] / w, FIT_MAX[1] / h)
    cw, ch = max(1, round(w * factor)), max(1, round(h * factor))
    small = rgb if (cw, ch) == (w, h) else rgb.resize((cw, ch), Image.BILINEAR)
    pw, ph = (math.ceil(n / FIT_MULTIPLE) * FIT_MULTIPLE for n in (cw, ch))
    if (pw, ph) == (cw, ch):
        padded = small
    else:
        padded = Image.new("RGB", (pw, ph))
        padded.paste(small, (0, 0))
    return FittedPhoto(image=padded, content_size=(cw, ch), original_size=(w, h))


def to_original(x: float, y: float, fitted: FittedPhoto) -> tuple[float, float]:
    """A point in fitted px -> the same point in the original photo's px (each axis scaled separately, since
    rounding makes the two shrink factors differ slightly). A point on the content maps inside the original."""
    (cw, ch), (w, h) = fitted.content_size, fitted.original_size
    return x * w / cw, y * h / ch
