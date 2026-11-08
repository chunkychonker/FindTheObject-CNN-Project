"""CLI: draw a grid of val scenes with the model's heatmap, peak and label, for the README.

Usage:
    python scripts/make_demo_figure.py --checkpoint outputs/checkpoints/<run>/best.pt
    python scripts/make_demo_figure.py --checkpoint ... --n-right 8 --n-wrong 4 --seed 0 --out docs/demo_val.png

Scores every val scene, then draws --n-right random correct scenes followed by --n-wrong random wrong
ones. "Correct" is the joint-accuracy rule from evaluate.py: right class AND heatmap peak inside the
true box. Each panel: heatmap overlay, true box (green), peak (cyan ring), predicted class and confidence;
misses with the wrong class also name the true class in red.

Val only, like show_data.py: test scenes stay unseen so they can't steer decisions. The val file comes
from the config stored in the checkpoint. Same checkpoint + seed -> same scenes picked. If there are
fewer correct or wrong scenes than asked for, all of them are drawn and the shortfall is printed.

Raises FileNotFoundError / ValueError / RuntimeError for a bad checkpoint or data file (as load_model
and SignScenes). The PNG is written atomically, so a failed run leaves the previous figure in place.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no window backend needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from findobject.data.dataset import SignScenes
from findobject.eval.heatmap_metrics import point_in_box
from findobject.inference.predict import Prediction, predict
from findobject.training.checkpoint import load_checkpoint_info, load_model
from findobject.utils.device import pick_device
from findobject.viz.overlay import draw_prediction

DEFAULT_OUT = Path("docs/demo_val.png")
BATCH_SIZE = 256
COLUMNS = 4
PANEL_INCHES = 2.4
DPI = 110


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-right", type=int, default=8)
    parser.add_argument("--n-wrong", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.n_right < 0 or args.n_wrong < 0 or args.n_right + args.n_wrong == 0:
        parser.error("--n-right and --n-wrong must be >= 0 and not both 0")

    info = load_checkpoint_info(args.checkpoint)
    data_path = Path(info.config["data"]["processed_dir"]) / "val.npz"
    dataset = SignScenes(data_path)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    print(f"scoring {len(dataset)} val scenes from {data_path} with {args.checkpoint} on {device}")

    preds, correct = score_all(model, dataset, device)
    right, wrong = pick_examples(correct, args.n_right, args.n_wrong, args.seed)
    if len(right) < args.n_right or len(wrong) < args.n_wrong:
        print(f"only {int(correct.sum())} correct / {int((~correct).sum())} wrong scenes available; "
              f"drawing {len(right)} + {len(wrong)}")

    run_id = args.checkpoint.parent.name
    title = (f"{run_id} on val ({correct.float().mean():.1%} joint acc): {len(right)} random hits, "
             f"{len(wrong)} random misses\ngreen box = true sign, cyan ring = heatmap peak, title = predicted class")
    fig = draw_grid(dataset, preds, right + wrong, title)
    save_figure_atomic(fig, args.out)
    print(f"wrote {args.out}")


def score_all(model: torch.nn.Module, dataset: SignScenes, device: torch.device) -> tuple[list[Prediction], torch.Tensor]:
    """Predictions for every scene, and a [N] bool of joint correctness (class right and peak in box)."""
    preds: list[Prediction] = []
    hits: list[torch.Tensor] = []
    for batch in DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0):
        batch_preds = predict(model, batch.image.to(device))
        class_ok = torch.tensor([p.class_id for p in batch_preds]) == batch.class_id
        peak_ok = point_in_box(torch.tensor([p.peak_xy for p in batch_preds]), batch.box)
        preds += batch_preds
        hits.append(class_ok & peak_ok)
    return preds, torch.cat(hits)


def pick_examples(correct: torch.Tensor, n_right: int, n_wrong: int, seed: int) -> tuple[list[int], list[int]]:
    """Random scene indices: up to n_right where correct is True, up to n_wrong where it is False.

    Each list is in ascending index order. Deterministic for a given (correct, n_right, n_wrong, seed).
    """
    rng = np.random.default_rng(seed)
    pools = (torch.nonzero(correct).flatten().numpy(), torch.nonzero(~correct).flatten().numpy())
    picks = [sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
             for pool, n in zip(pools, (n_right, n_wrong))]
    return picks[0], picks[1]


def draw_grid(dataset: SignScenes, preds: list[Prediction], indices: list[int], title: str) -> plt.Figure:
    """One draw_prediction panel per index, COLUMNS wide; unused grid cells are hidden."""
    rows = -(-len(indices) // COLUMNS)  # ceiling division
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(COLUMNS * PANEL_INCHES, rows * (PANEL_INCHES + 0.35) + 0.6),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, i in zip(axes.flat, indices):
        sample = dataset[i]
        draw_prediction(ax, sample.image, preds[i], true_box=tuple(sample.box.tolist()),
                        true_class=int(sample.class_id))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def save_figure_atomic(fig: plt.Figure, path: Path) -> None:
    """Save as PNG to <path>.tmp, then rename over path. Creates parent dirs. Raises OSError on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fig.savefig(tmp, dpi=DPI, format="png")
    plt.close(fig)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
