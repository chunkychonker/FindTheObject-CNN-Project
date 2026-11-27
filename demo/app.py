"""The glass-box detector: a local Gradio app that shows every stage of the traffic-sign detector and lets you
paste signs to find where it breaks (docs/demo-spec.md).

Usage (from the repo root):
    python scripts/export_demo_model.py      # once: copies seed 0's weights and τ_W into demo/assets/
    .venv/bin/python demo/app.py             # then open the printed http://127.0.0.1 address

This file only wires inputs to findobject.demo and lays out the page. Startup fails loudly if the exported
model or the GTSDB/GTSRB data is missing.
"""

import time
from pathlib import Path

import gradio as gr
from PIL import Image

from findobject.data.class_names import CLASS_NAMES
from findobject.data.gtsdb import load_photo, read_gtsdb
from findobject.data.gtsrb import load_sign, read_train_index
from findobject.demo.assets import load_assets
from findobject.demo.captions import TABLE_HEADERS
from findobject.demo.examples import EXAMPLE_PHOTOS, example_photos, sign_per_class
from findobject.demo.fit import fit_photo
from findobject.demo.trace import trace
from findobject.demo.views import fool_view, place_sign, watch_view
from findobject.eval.photo_truth import photo_truth
from findobject.utils.device import pick_device

ROOT = Path(__file__).resolve().parents[1]
GTSDB_DIR = ROOT / "data/raw/gtsdb/FullIJCNN2013"
GTSRB_DIR = ROOT / "data/raw/gtsrb"
ASSET_DIR = ROOT / "demo/assets"
WATCH_DEFAULT = 206  # five signs at three sizes
FOOL_DEFAULT = 108  # no signs, so the pasted one is the only thing to find
FOOL_DEFAULT_CLASS = 14  # stop
SIZE_RANGE = (8, 200)  # px, long side of the pasted sign

DEVICE = pick_device("auto")
MODEL, META = load_assets(ASSET_DIR, DEVICE)
EXAMPLES = example_photos(read_gtsdb(GTSDB_DIR))
SIGNS = {c: load_sign(s) for c, s in sign_per_class(read_train_index(GTSRB_DIR)).items()}
EXAMPLE_CHOICES = [(f"photo {i}: {text}", i) for i, text in EXAMPLE_PHOTOS.items()]
CLASS_CHOICES = [f"{c}: {name}" for c, name in enumerate(CLASS_NAMES)]

INTRO = f"""
# The glass-box sign detector
A small CNN that finds German traffic signs in street photos, with every stage shown. Weights from run
`{META.run}`; the threshold starts at τ_W = {META.tau_w:.2f}, chosen on dev photos. Example photos are dev
photos only (never the held-out test set). Running on `{DEVICE.type}`.
"""


def load_example(index: int) -> tuple[Image.Image, tuple]:
    """An example photo and its labeled signs, as (image, truth)."""
    photo = EXAMPLES[index]
    return load_photo(photo), photo_truth(photo)


# ---- tab 1: watch it look ----

def watch_outputs(state: tuple, tau: float) -> list:
    tr, truth, seconds = state
    v = watch_view(tr, truth, tau)
    timing = f"Model time: {seconds:.2f} s on `{DEVICE.type}` (detect plus a second pass for the heatmaps)."
    return [v.scales, v.fit_text, v.heatmaps, v.peaks, v.peaks_text, v.suppression, v.suppression_text,
            v.answer, v.answer_text, v.rows, timing]


def watch_run(photo: tuple | None, tau: float) -> list:
    if photo is None:
        raise gr.Error("Pick an example or upload a photo first.")
    image, truth = photo
    start = time.perf_counter()
    tr = trace(MODEL, image)
    state = (tr, truth, time.perf_counter() - start)
    return [state, *watch_outputs(state, tau)]


def watch_pick(index: int) -> tuple:
    image, truth = load_example(index)
    return image, (image, truth)


def watch_upload(image: Image.Image | None) -> tuple | None:
    return None if image is None else (image, None)  # uploads have no labels to grade against


def watch_rerender(state: tuple | None, tau: float) -> list:
    if state is None:
        raise gr.Error("Run the detector first.")
    return watch_outputs(state, tau)  # no model call: the trace is reused (A5)


# ---- tab 2: try to fool it ----

def fool_pick(index: int) -> tuple:
    image, truth = load_example(index)
    return image, (image, truth)


def fool_upload(image: Image.Image | None) -> tuple | None:
    if image is None:
        return None
    return fit_photo(image).content, ()  # shrunk to fit first, so the pasted size is the size the model gets


def fool_run(background: tuple | None, class_index: int, long_side: float, fx: float, fy: float,
             tau: float) -> list:
    if background is None:
        raise gr.Error("Pick a background or upload one first.")
    image, truth = background
    placed = place_sign(image, truth, SIGNS[class_index], class_index, int(long_side), fx, fy)
    tr = trace(MODEL, placed.image)
    return [(tr, placed), *fool_outputs((tr, placed), tau)]


def fool_outputs(state: tuple, tau: float) -> list:
    v = fool_view(*state, tau)
    return [v.answer, v.text, v.heatmaps]


def fool_rerender(state: tuple | None, tau: float) -> list:
    if state is None:
        raise gr.Error("Paste a sign first.")
    return fool_outputs(state, tau)


def stage(title: str, text: str) -> None:
    gr.Markdown(f"### {title}\n{text}")


def build() -> gr.Blocks:
    image = dict(type="pil", format="png", interactive=False)
    with gr.Blocks(title="Glass-box sign detector", fill_width=True) as app:
        gr.Markdown(INTRO)
        with gr.Tab("Watch it look"):
            w_photo, w_state = gr.State(), gr.State()
            with gr.Row():
                with gr.Column(scale=1):
                    w_pick = gr.Dropdown(EXAMPLE_CHOICES, value=WATCH_DEFAULT, label="Example photo")
                    w_input = gr.Image(type="pil", sources=["upload", "clipboard"], label="…or upload your own",
                                       height=260)
                    w_tau = gr.Slider(0.05, 0.95, value=META.tau_w, step=0.01, label="Threshold τ")
                    w_go = gr.Button("Run detector", variant="primary")
                    w_time = gr.Markdown()
                with gr.Column(scale=3):
                    stage("1 · Three sizes of the same photo",
                          "The model only knows signs 16-48 px wide, so it looks at the photo at full, half and "
                          "quarter size. A big sign shrinks into that range at a smaller scale.")
                    w_scales = gr.Image(label="Pyramid", **image)
                    w_fit = gr.Markdown()
            stage("2 · Where it thinks signs are",
                  "One heatmap per scale. Each cell is the model's guess that a sign's center is in it; bright = sure.")
            w_heat = gr.Gallery(label="Heatmaps", columns=3, height="auto", object_fit="contain")
            with gr.Row():
                with gr.Column():
                    stage("3 · Peaks above the threshold", "")
                    w_peaks = gr.Image(label="Peaks", **image)
                    w_peaks_text = gr.Markdown()
                with gr.Column():
                    stage("4 · Drop the repeats", "")
                    w_supp = gr.Image(label="Suppression", **image)
                    w_supp_text = gr.Markdown()
            stage("5 · The answer", "Green box: labeled sign found. Orange box: missed. Red ring: found nothing.")
            w_answer = gr.Image(label="Answer", **image)
            w_answer_text = gr.Markdown()
            w_table = gr.Dataframe(headers=TABLE_HEADERS, label="Detections (points in the photo's own px)",
                                   interactive=False)

            outs = [w_scales, w_fit, w_heat, w_peaks, w_peaks_text, w_supp, w_supp_text, w_answer, w_answer_text,
                    w_table, w_time]
            run = dict(fn=watch_run, inputs=[w_photo, w_tau], outputs=[w_state, *outs])
            w_pick.input(watch_pick, w_pick, [w_input, w_photo]).then(**run)
            w_input.upload(watch_upload, w_input, w_photo).then(**run)
            w_go.click(**run)
            w_tau.release(watch_rerender, [w_state, w_tau], outs)
            app.load(watch_pick, w_pick, [w_input, w_photo]).then(**run)

        with gr.Tab("Try to fool it"):
            f_bg, f_state = gr.State(), gr.State()
            gr.Markdown("Paste a real sign (a GTSRB training image) onto a photo at any size and place, and see "
                        "if the detector still finds it. Tip: try very small and very big signs.")
            with gr.Row():
                with gr.Column(scale=1):
                    f_pick = gr.Dropdown(EXAMPLE_CHOICES, value=FOOL_DEFAULT, label="Background photo")
                    f_input = gr.Image(type="pil", sources=["upload", "clipboard"], label="…or upload your own",
                                       height=200)
                    f_class = gr.Dropdown(CLASS_CHOICES, value=CLASS_CHOICES[FOOL_DEFAULT_CLASS], type="index",
                                          label="Sign")
                    f_size = gr.Slider(*SIZE_RANGE, value=32, step=1, label="Sign size (px, long side)")
                    f_x = gr.Slider(0, 1, value=0.5, step=0.01, label="Left ↔ right")
                    f_y = gr.Slider(0, 1, value=0.5, step=0.01, label="Top ↕ bottom")
                    f_tau = gr.Slider(0.05, 0.95, value=META.tau_w, step=0.01, label="Threshold τ")
                    f_go = gr.Button("Paste and detect", variant="primary")
                with gr.Column(scale=3):
                    f_text = gr.Markdown()
                    f_answer = gr.Image(label="Answer", **image)
            f_heat = gr.Gallery(label="Heatmaps", columns=3, height="auto", object_fit="contain")

            run = dict(fn=fool_run, inputs=[f_bg, f_class, f_size, f_x, f_y, f_tau],
                       outputs=[f_state, f_answer, f_text, f_heat])
            f_pick.input(fool_pick, f_pick, [f_input, f_bg]).then(**run)
            f_input.upload(fool_upload, f_input, f_bg).then(**run)
            f_go.click(**run)
            f_class.input(**run)
            for s in (f_size, f_x, f_y):
                s.release(**run)
            f_tau.release(fool_rerender, [f_state, f_tau], [f_answer, f_text, f_heat])
            app.load(fool_pick, f_pick, [f_input, f_bg]).then(**run)
    return app


if __name__ == "__main__":
    build().launch()
