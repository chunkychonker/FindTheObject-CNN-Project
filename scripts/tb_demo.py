"""Write a fake 'training run' to TensorBoard so you can check the dashboard works before train.py exists.

Everything here is synthetic noise that 'improves' over epochs. Delete this once real runs exist.

    python scripts/tb_demo.py
    tensorboard --logdir outputs/runs
"""

import math

import torch
import torch.nn.functional as F

from findobject.eval.heatmap_metrics import center_error_px, peak_in_box
from findobject.viz.tensorboard import TBLogger

SCENE, HM, N_CLASSES, EPOCHS, STEPS_PER_EPOCH = 128, 32, 43, 15, 50


def fake_batch(g: torch.Generator, n: int = 64):
    images = F.avg_pool2d(torch.rand(n, 3, SCENE, SCENE, generator=g), 9, stride=1, padding=4)
    images = (images - images.amin()) / (images.amax() - images.amin())
    sizes = torch.randint(16, 49, (n,), generator=g)
    x1 = (torch.rand(n, generator=g) * (SCENE - sizes)).long()
    y1 = (torch.rand(n, generator=g) * (SCENE - sizes)).long()
    boxes = torch.stack([x1, y1, x1 + sizes, y1 + sizes], dim=1)
    for i in range(n):  # paint a red square where the 'sign' is
        a, b, c, d = boxes[i].tolist()
        images[i, :, b:d, a:c] = torch.tensor([0.9, 0.1, 0.1]).view(3, 1, 1)
    # crude GT blob: one-hot at the center cell, blurred (NOT the real gaussian_heatmap — that's your TODO)
    gt = torch.zeros(n, 1, HM, HM)
    cx = ((x1 + sizes / 2) / (SCENE / HM)).long().clamp(max=HM - 1)
    cy = ((y1 + sizes / 2) / (SCENE / HM)).long().clamp(max=HM - 1)
    gt[torch.arange(n), 0, cy, cx] = 1.0
    gt = F.avg_pool2d(F.avg_pool2d(gt, 3, 1, 1), 3, 1, 1)
    gt = gt / gt.amax(dim=(2, 3), keepdim=True)
    labels = torch.randint(0, N_CLASSES, (n,), generator=g)
    return images, gt, boxes, labels


def main() -> None:
    g = torch.Generator().manual_seed(0)
    val_images, val_gt, val_boxes, val_labels = fake_batch(g)

    with TBLogger.for_run("demo") as tb:
        tb.config({"model": "fake", "lr": 1e-3, "data": {"scene_size": SCENE, "heatmap_size": HM}})
        step = 0
        for epoch in range(EPOCHS):
            progress = epoch / (EPOCHS - 1)  # 0 -> 1
            for _ in range(STEPS_PER_EPOCH):
                decay = math.exp(-step / 250)
                tb.scalars({
                    "loss/heatmap": 0.05 + 0.5 * decay + 0.02 * torch.rand(1, generator=g).item(),
                    "loss/cls": 0.1 + 3.6 * decay + 0.1 * torch.rand(1, generator=g).item(),
                }, step)
                step += 1

            noise = torch.rand(val_gt.shape, generator=g)
            pred = (progress * val_gt + (1 - progress) * noise).clamp(0, 1)
            right = torch.rand(len(val_labels), generator=g) < 0.05 + 0.9 * progress
            pred_labels = torch.where(right, val_labels, torch.randint(0, N_CLASSES, val_labels.shape, generator=g))

            tb.scalars({
                "val/center_err_px": center_error_px(pred, val_boxes, SCENE).mean(),
                "val/hit_rate": peak_in_box(pred, val_boxes, SCENE).float().mean(),
                "val/cls_acc": (pred_labels == val_labels).float().mean(),
            }, epoch)
            tb.heatmap_panels("val/heatmaps", val_images, pred, epoch, gt_heatmaps=val_gt, boxes=val_boxes)
            tb.confusion_matrix("val/confusion", val_labels, pred_labels, N_CLASSES, epoch)

        print(f"wrote {tb.log_dir}\nnow run: tensorboard --logdir outputs/runs")


if __name__ == "__main__":
    main()
