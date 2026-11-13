# Notebooks: the whole project, in order

These nine notebooks walk through the project from raw data to a model you can point at your own photos. They
are a readable front end to the code in `src/findobject/` and `scripts/`: they call that code, and they don't
replace it. You can run the whole project from here, without typing any terminal command.

| # | notebook | what you'll do | quick run* |
|---|---|---|---|
| 00 | `00_setup` | Check Python, the packages, the datasets and the trained runs on disk | ~30 s |
| 01 | `01_synthetic_scenes` | See how a scene is made: a GTSRB sign pasted onto a COCO photo, plus its heatmap target | ~5 s |
| 02 | `02_model` | Go through the CNN layer by layer and push one scene through it | ~2 s |
| 03 | `03_train` | Train a small model for a few minutes and watch the loss fall; optionally start a full run | ~2 min |
| 04 | `04_evaluate_synthetic` | Score a trained run on val: joint, class and pointing accuracy, and its mistakes | ~10 s |
| 05 | `05_real_photos` | Move to real road photos (GTSDB) and the real-vs-hybrid comparison (v2) | ~5 s |
| 06 | `06_presence_empty_decoys` | Teach the model to say "no sign": empty scenes, decoys, and τ (v3a, v3a.1) | ~30 s |
| 07 | `07_inference_demo` | Run a trained model on image files, including your own | ~5 s |
| 08 | `08_diagnostics` | The one-off checks: data pictures, shortcut test, memorization probe | ~30 s |

\*With every flag off, on an Apple-silicon Mac with the datasets and trained runs already on disk.

Read them in order the first time. From 01 on, each one starts with a short note saying what it needs and what it
writes, and ends by pointing to the next one.

## Opening them

**VS Code (recommended):** open the repo folder, open a notebook, click **Select Kernel** in the top-right
corner and pick the project's `.venv` interpreter. `ipykernel` (in `requirements.txt`) is what lets VS Code run
Python from that environment.

**Jupyter in the browser:** `pip install jupyterlab` inside the `.venv`, then run `jupyter lab` from the repo
root. JupyterLab isn't in `requirements.txt`, because VS Code doesn't need it.

Wherever you open them, each notebook's setup cell finds the repo root and works from there, so file paths
mean the same thing in a notebook as in a terminal.

## The Settings cell and its flags

The first code cell of each notebook (from 01 on) is **Settings**, the only cell you need to edit. Change a value
there and re-run the notebook from the top. Anything slow, anything that writes results the project keeps, and
anything that touches the test set sits behind a flag that is **off by default**:

| flag | in | when it's on |
|---|---|---|
| `REBUILD` | 01, 05, 06 | Regenerate the stored `.npz` datasets (minutes). They come out byte-identical, because generation is seeded. |
| `FULL_RUN` | 03 | Start a full training run (~1 hour) in the background. |
| `RUN_FOR_REAL` | 04, 05, 06, 08 | Run the real scripts and record their results under `outputs/eval/` (or `results/`). |
| `SCORE_TEST` | 04, 05, 06 | Score the **test** set, once per run. A second attempt is refused with an error, so test results can't be tuned toward. |

With a flag off, its cell prints the exact command it *would* run, for example
`$ python scripts/evaluate.py --checkpoint ... --split val`, followed by `(not run: ...)`. You can copy that command to a terminal
if you prefer.

## What a quick run (all flags off) does

* It **reads** the datasets in `data/processed/`, the trained runs in `outputs/checkpoints/` and the stored reports in
  `outputs/eval/`.
* It **computes live** wherever that takes seconds: scoring val, finding τ, running the model on images. Where
  a result was recorded earlier, the notebook shows it too, so you can see the two agree.
* It **writes** only under `outputs/notebooks/`: the mini training run, demo images, figures and logs. That folder
  is gitignored and safe to delete at any time.

If a file a notebook needs is missing, the notebook stops with a message listing what's missing and the command that makes
it. It doesn't crash halfway through.

## Full training in the background

In notebook 03, `FULL_RUN = True` starts `scripts/train.py` as a separate process with its output going to
`outputs/notebooks/train-<run>.log`. That process isn't tied to the notebook: restarting the kernel or closing
VS Code doesn't stop it. Shutting the Mac down does, and sleep pauses it. The cell after it shows whether the job is
still running and the last lines of its log. Run it whenever you want an update. Running the start cell again while
a job is going is refused, so you can't start two copies by accident.

## For contributors

* Every code cell starts with a `#` comment saying what it does.
* Notebooks are committed with **outputs cleared** (in VS Code, "Clear All Outputs" before saving).
* `nbhelpers.py` holds the small helpers the notebooks share: finding runs, checking files exist, launching scripts.

`pytest` checks the comments, the cleared outputs and the default-off flags, and also tests the helpers.
`pytest -m notebooks` runs every notebook top to bottom. That takes a few minutes and needs the data and the
trained runs.
