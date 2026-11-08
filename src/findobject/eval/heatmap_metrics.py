"""Localization metrics for predicted heatmaps.

Coordinate convention matches data/synth.py: heatmap cell (row r, col c) covers scene
pixels [c*stride, (c+1)*stride) and is centered at ((c + 0.5)*stride, (r + 0.5)*stride).
Boxes are (x1, y1, x2, y2) in scene pixels with x2/y2 exclusive.
"""

import torch


def peak_centers(heatmaps: torch.Tensor, scene_size: int) -> torch.Tensor:
    """[B, H, W] or [B, 1, H, W] heatmaps -> [B, 2] float (x, y) of each argmax cell center, in scene px."""
    if heatmaps.dim() == 4:
        heatmaps = heatmaps.squeeze(1)
    b, h, w = heatmaps.shape
    idx = heatmaps.reshape(b, -1).argmax(dim=1)
    rows, cols = idx // w, idx % w
    x = (cols.float() + 0.5) * (scene_size / w)
    y = (rows.float() + 0.5) * (scene_size / h)
    return torch.stack([x, y], dim=1)


def box_centers(boxes: torch.Tensor) -> torch.Tensor:
    """[B, 4] (x1, y1, x2, y2) -> [B, 2] float (x, y) centers."""
    boxes = boxes.float()
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], dim=1)


def center_error_px(pred_heatmaps: torch.Tensor, boxes: torch.Tensor, scene_size: int) -> torch.Tensor:
    """[B] Euclidean distance (scene px) from the predicted peak to the true sign center."""
    return (peak_centers(pred_heatmaps, scene_size) - box_centers(boxes)).norm(dim=1)


def peak_in_box(pred_heatmaps: torch.Tensor, boxes: torch.Tensor, scene_size: int) -> torch.Tensor:
    """[B] bool: does the predicted peak land inside the true sign box? Mean of this = hit rate."""
    return point_in_box(peak_centers(pred_heatmaps, scene_size), boxes)


def point_in_box(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """[B, 2] (x, y) points, [B, 4] (x1, y1, x2, y2) boxes -> [B] bool, x2/y2 exclusive."""
    p = points.float()
    b = boxes.float()
    return (p[:, 0] >= b[:, 0]) & (p[:, 0] < b[:, 2]) & (p[:, 1] >= b[:, 1]) & (p[:, 1] < b[:, 3])
