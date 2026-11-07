"""CLI: train the config's model.arch on data/processed/train.npz, validating on val.npz every epoch.

Usage:
    python scripts/train.py --config configs/baseline.yaml              # full run
    python scripts/train.py --config configs/peak.yaml                  # full run, class read at the peak
    python scripts/train.py --config configs/baseline.yaml --overfit 32  # sanity check, ~1 min

Raises KeyError if the config has no model.arch, ValueError if it names an unknown architecture.

Full run writes TensorBoard logs to outputs/runs/<run_id>/ and last.pt / best.pt (by val joint
accuracy) to outputs/checkpoints/<run_id>/, where run_id = <run_name>-<timestamp>. Never touches
test.npz. Not idempotent: each invocation is a new run, and MPS is not bit-for-bit deterministic;
safe to re-run because it never overwrites an earlier run. No resume: a crashed run restarts.

--overfit N trains on the first N train scenes only and exits 0 if the model reaches 100% class
and pointing accuracy on them (in eval mode), else 1. Failure means a pipeline bug.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, default_collate

from findobject.data.dataset import Sample, SignScenes
from findobject.data.gtsrb import NUM_CLASSES
from findobject.eval.evaluate import evaluate
from findobject.models.factory import Arch, build_model
from findobject.training.checkpoint import save_checkpoint
from findobject.training.config import TrainConfig
from findobject.training.epoch import train_one_epoch
from findobject.training.losses import LossParts, detection_loss
from findobject.utils.device import pick_device
from findobject.utils.seed import seed_everything
from findobject.viz.tensorboard import RUNS_ROOT, TBLogger

CHECKPOINTS_ROOT = Path("outputs/checkpoints")
VAL_BATCH_SIZE = 256  # no gradients at eval time, so bigger batches are cheap
PANEL_COUNT = 8  # fixed val scenes drawn every epoch, so the TensorBoard slider shows progress
OVERFIT_STEPS = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--overfit", type=int, default=0, metavar="N",
                        help="sanity check: memorize the first N train scenes, then exit")
    args = parser.parse_args()

    raw_cfg = yaml.safe_load(args.config.read_text())
    cfg = TrainConfig(**raw_cfg["train"])
    arch = Arch(raw_cfg["model"]["arch"])
    processed = Path(raw_cfg["data"]["processed_dir"])
    device = pick_device(cfg.device)
    shuffle_gen = seed_everything(cfg.seed)
    train_ds = SignScenes(processed / "train.npz")
    print(f"device {device}, model {arch.value}, {len(train_ds)} train scenes")

    if args.overfit:
        sys.exit(0 if overfit(arch, train_ds, args.overfit, cfg, device) else 1)
    val_ds = SignScenes(processed / "val.npz")
    train(arch, cfg, raw_cfg, train_ds, val_ds, device, shuffle_gen)


def train(arch: Arch, cfg: TrainConfig, raw_cfg: dict, train_ds: SignScenes, val_ds: SignScenes,
          device: torch.device, shuffle_gen: torch.Generator) -> None:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=shuffle_gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, num_workers=0)
    panels = default_collate([val_ds[i] for i in range(PANEL_COUNT)])

    model = build_model(arch).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))  # per-step decay to 0

    run_id = f"{cfg.run_name}-{datetime.now():%Y%m%d-%H%M%S}"
    ckpt_dir = CHECKPOINTS_ROOT / run_id
    print(f"run {run_id}: {cfg.epochs} epochs x {len(train_loader)} steps, {len(val_ds)} val scenes")

    step = 0
    best_joint = -1.0
    with TBLogger(RUNS_ROOT / run_id) as tb:
        tb.config(raw_cfg)

        def on_step(parts: LossParts) -> None:
            nonlocal step
            step += 1
            if step % cfg.log_every_steps == 0:
                tb.scalars({
                    "train/loss_total": parts.total.detach(),
                    "train/loss_class": parts.class_loss.detach(),
                    "train/loss_heatmap": parts.heatmap_loss.detach(),
                    "train/heatmap_excess": (parts.heatmap_loss - parts.heatmap_floor).detach(),
                    "train/lr": scheduler.get_last_lr()[0],
                }, step)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.monotonic()
            stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg.heatmap_loss_weight, on_step)
            val = evaluate(model, val_loader, device, cfg.heatmap_loss_weight)
            metrics = val.scalars()

            tb.scalars({f"val/{k}": v for k, v in metrics.items()}, epoch)
            tb.scalars({"train_epoch/loss_total": stats.mean_total, "train_epoch/class_acc": stats.class_acc,
                        "train_epoch/pointing_acc": stats.pointing_acc}, epoch)
            tb.confusion_matrix("val/confusion", val.y_true, val.y_pred, NUM_CLASSES, epoch)
            _log_panels(tb, model, panels, device, epoch)
            if epoch % cfg.histogram_every_epochs == 0:
                tb.histograms(model, epoch)
            tb.flush()

            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, raw_cfg, metrics)
            is_best = val.joint_acc > best_joint
            if is_best:
                best_joint = val.joint_acc
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, raw_cfg, metrics)
            print(
                f"epoch {epoch:2d}/{cfg.epochs} {time.monotonic() - t0:4.0f}s | "
                f"train loss {stats.mean_total:.3f} cls {stats.class_acc:.3f} pt {stats.pointing_acc:.3f} | "
                f"val cls {val.class_acc:.3f} pt {val.pointing_acc:.3f} joint {val.joint_acc:.3f} "
                f"err med {val.center_err_median:.1f}px hm_excess {val.heatmap_excess:.2f}"
                f"{'  *best' if is_best else ''}",
                flush=True,
            )
    print(f"done. best val joint acc {best_joint:.3f}; checkpoints in {ckpt_dir}")


@torch.no_grad()
def _log_panels(tb: TBLogger, model: nn.Module, panels: Sample, device: torch.device, epoch: int) -> None:
    model.eval()
    out = model(panels.image.to(device))
    tb.heatmap_panels("val/heatmaps", panels.image, torch.sigmoid(out.heatmap_logits).cpu(), epoch,
                      gt_heatmaps=panels.heatmap, boxes=panels.box)


def overfit(arch: Arch, train_ds: SignScenes, n: int, cfg: TrainConfig, device: torch.device) -> bool:
    batch: Sample = default_collate([train_ds[i] for i in range(n)]).to(device)
    model = build_model(arch).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for step in range(1, OVERFIT_STEPS + 1):
        model.train()
        parts = detection_loss(model(batch.image), batch, cfg.heatmap_loss_weight)
        if not torch.isfinite(parts.total):
            raise FloatingPointError(f"non-finite loss at overfit step {step}")
        optimizer.zero_grad(set_to_none=True)
        parts.total.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            r = evaluate(model, [batch], device, cfg.heatmap_loss_weight)  # same batch, eval mode
            print(f"step {step:3d} | loss cls {parts.class_loss.item():.3f} hm {parts.heatmap_loss.item():.2f} "
                  f"(floor {parts.heatmap_floor.item():.2f}) | eval cls {r.class_acc:.3f} pt {r.pointing_acc:.3f}")
    ok = r.class_acc == 1.0 and r.pointing_acc == 1.0
    print("OVERFIT PASS" if ok else "OVERFIT FAIL: pipeline cannot memorize a tiny set; look for a bug")
    return ok


if __name__ == "__main__":
    main()
