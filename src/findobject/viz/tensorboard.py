"""Thin TensorBoard wrapper for this project.

Usage in a train loop:

    from findobject.viz.tensorboard import TBLogger

    with TBLogger.for_run("baseline") as tb:          # -> outputs/runs/baseline-<timestamp>/
        tb.scalars({"loss/total": loss, "loss/heatmap": hm_loss}, step)
        ...
        tb.heatmap_panels("val/heatmaps", images, pred_hm, epoch, gt_heatmaps=gt_hm, boxes=boxes)
        tb.confusion_matrix("val/confusion", y_true, y_pred, num_classes=43, step=epoch)

Then: tensorboard --logdir outputs/runs
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are rendered straight into TensorBoard
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid

from findobject.eval.heatmap_metrics import peak_centers

RUNS_ROOT = Path("outputs/runs")


class TBLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(str(self.log_dir))

    @classmethod
    def for_run(cls, name: str, root: str | Path = RUNS_ROOT) -> "TBLogger":
        """One subdir per run, so baseline vs resnet (vs reruns) overlay on the same charts."""
        return cls(Path(root) / f"{name}-{datetime.now():%Y%m%d-%H%M%S}")

    # --- scalars -------------------------------------------------------------------

    def scalars(self, values: dict[str, float | torch.Tensor], step: int) -> None:
        """Log several scalars. Use 'group/name' tags (e.g. 'loss/cls', 'val/hit_rate') to group charts."""
        for tag, v in values.items():
            self.writer.add_scalar(tag, float(v), step)

    def config(self, cfg: dict) -> None:
        """Record the run's config as text so you can tell runs apart later."""
        lines = [f"    {k}: {v}" for k, v in _flatten(cfg).items()]
        self.writer.add_text("config", "\n".join(lines), 0)

    # --- heatmaps ------------------------------------------------------------------

    def heatmap_panels(
        self,
        tag: str,
        images: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        step: int,
        gt_heatmaps: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        max_images: int = 8,
        alpha: float = 0.8,
    ) -> None:
        """One row per sample: [scene + boxes | predicted heatmap overlay | GT heatmap overlay].

        images: [B, 3, S, S] float in [0, 1] (un-normalize first if you normalized inputs).
        pred_heatmaps / gt_heatmaps: [B, H, W] or [B, 1, H, W], already in [0, 1]
            (pass sigmoid(logits), not raw logits).
        boxes: [B, 4] true (x1, y1, x2, y2) in scene px -> drawn green; predicted peak drawn cyan.

        Log the SAME fixed val batch every epoch so you can scrub the slider and watch it learn.
        """
        n = min(max_images, images.shape[0])
        images = images[:n].detach().float().cpu().clamp(0, 1)
        scene = images.shape[-1]
        pred = _as_b1hw(pred_heatmaps[:n])

        annotated = []
        peaks = peak_centers(pred, scene)
        for i in range(n):
            img = (images[i] * 255).to(torch.uint8)
            r = max(2, scene // 64)
            px, py = peaks[i].tolist()
            draw_boxes = [torch.tensor([px - r, py - r, px + r, py + r])]
            colors = ["cyan"]
            if boxes is not None:
                x1, y1, x2, y2 = boxes[i].float().tolist()
                draw_boxes.insert(0, torch.tensor([x1, y1, x2 - 1, y2 - 1]))  # x2/y2 exclusive
                colors.insert(0, "lime")
            img = draw_bounding_boxes(img, torch.stack(draw_boxes), colors=colors, width=1)
            annotated.append(img.float() / 255)

        cols = [torch.stack(annotated), _overlay(images, pred, alpha)]
        if gt_heatmaps is not None:
            cols.append(_overlay(images, _as_b1hw(gt_heatmaps[:n]), alpha))

        # interleave so each row of the grid is one sample
        tiles = torch.stack(cols, dim=1).flatten(0, 1)
        grid = make_grid(tiles, nrow=len(cols), padding=2, pad_value=1.0)
        self.writer.add_image(tag, grid, step)

    # --- classification ------------------------------------------------------------

    def confusion_matrix(
        self,
        tag: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        num_classes: int,
        step: int,
        class_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Row-normalized confusion matrix (row = true class) as a heatmap figure. Returns raw counts."""
        y_true = y_true.detach().cpu().long().flatten()
        y_pred = y_pred.detach().cpu().long().flatten()
        counts = torch.bincount(y_true * num_classes + y_pred, minlength=num_classes**2)
        counts = counts.reshape(num_classes, num_classes)
        norm = counts.float() / counts.sum(dim=1, keepdim=True).clamp(min=1)

        size = max(5.0, num_classes * 0.22)
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(norm.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = (y_true == y_pred).float().mean().item() if len(y_true) else 0.0
        ax.set_title(f"acc {acc:.3f}  (n={len(y_true)})")
        if class_names is not None and num_classes <= 20:
            ax.set_xticks(range(num_classes), class_names, rotation=90, fontsize=7)
            ax.set_yticks(range(num_classes), class_names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        self.writer.add_figure(tag, fig, step)  # closes the figure
        return counts

    def histograms(self, model: torch.nn.Module, step: int) -> None:
        """Weight + grad histograms per layer. Cheap way to spot dead layers / exploding grads. Log every few epochs."""
        for name, p in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", p.detach().cpu(), step)
            if p.grad is not None:
                self.writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), step)

    # --- lifecycle -----------------------------------------------------------------

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _as_b1hw(hm: torch.Tensor) -> torch.Tensor:
    hm = hm.detach().float().cpu()
    return hm.unsqueeze(1) if hm.dim() == 3 else hm


def _overlay(images: torch.Tensor, heatmaps: torch.Tensor, alpha: float) -> torch.Tensor:
    """Upsample [B,1,H,W] heatmaps to scene size, colorize with 'inferno', blend onto images.

    Blend weight scales with heatmap value, so cold regions keep the original scene visible.
    """
    scene = images.shape[-1]
    up = F.interpolate(heatmaps, size=(scene, scene), mode="bilinear", align_corners=False)
    up = up.clamp(0, 1).squeeze(1).numpy()
    colored = torch.from_numpy(plt.get_cmap("inferno")(up)[..., :3]).permute(0, 3, 1, 2).float()
    w = alpha * (0.2 + 0.8 * torch.from_numpy(up).unsqueeze(1).float())  # max blend = alpha at the peak
    return ((1 - w) * images + w * colored).clamp(0, 1)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
