"""Renders a scene with its predicted heatmap and label for humans to look at."""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch import Tensor

from findobject.data.class_names import CLASS_NAMES
from findobject.inference.predict import Prediction

COLORMAP = "inferno"
TRUE_COLOR = "lime"
WRONG_COLOR = "red"
PEAK_COLOR = "cyan"


def heatmap_overlay(image: Tensor, heatmap: Tensor, alpha: float = 0.6) -> np.ndarray:
    """Float [3, S, S] image in [0, 1] + [h, w] heatmap in [0, 1] -> uint8 [S, S, 3].

    The heatmap is upsampled (bilinear) to S x S and colored with COLORMAP. Each pixel's blend
    weight is alpha * heatmap value, so where the heatmap is 0 the scene is left exactly as it was.
    Raises ValueError if the image is not [3, S, S] or alpha is outside [0, 1].
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image [3, S, S], got {tuple(image.shape)}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    size = tuple(image.shape[1:])
    up = F.interpolate(heatmap[None, None].float().cpu(), size=size, mode="bilinear", align_corners=False)
    up = up[0, 0].clamp(0, 1).numpy()  # [S, S]
    colored = plt.get_cmap(COLORMAP)(up)[..., :3]  # [S, S, 3] in [0, 1]
    scene = image.detach().cpu().permute(1, 2, 0).numpy()
    w = alpha * up[..., None]
    blended = (1 - w) * scene + w * colored
    return np.round(np.clip(blended, 0, 1) * 255).astype(np.uint8)


def draw_prediction(
    ax: Axes,
    image: Tensor,
    pred: Prediction,
    true_box: tuple[int, int, int, int] | None = None,
    true_class: int | None = None,
) -> None:
    """Draw the heatmap overlay, a marker at the predicted peak, and the predicted label as the title.

    true_box (x1, y1, x2, y2, scene px, x2/y2 exclusive) is outlined in TRUE_COLOR. If true_class is
    given and differs from the prediction, the title turns WRONG_COLOR and names the true class.
    Raises ValueError as heatmap_overlay does.
    """
    ax.imshow(heatmap_overlay(image, pred.heatmap), interpolation="nearest")  # show real pixels, not a blur
    x, y = pred.peak_xy
    # imshow centers pixel i at coordinate i, so a scene-px position p sits at p - 0.5.
    # Hollow ring, not a cross: the heatmap glow is only ~3 cells wide and must stay visible inside it.
    ax.plot(x - 0.5, y - 0.5, marker="o", fillstyle="none", color=PEAK_COLOR, markersize=14, markeredgewidth=1.2)
    if true_box is not None:
        x1, y1, x2, y2 = true_box
        ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, edgecolor=TRUE_COLOR, linewidth=1))
    title = f"{CLASS_NAMES[pred.class_id]} ({pred.class_prob:.0%})"
    color = "black"
    if true_class is not None and true_class != pred.class_id:
        title += f"\ntrue: {CLASS_NAMES[true_class]}"
        color = WRONG_COLOR
    ax.set_title(title, fontsize=8, color=color)
    ax.set_axis_off()
