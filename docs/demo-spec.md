# Demo spec: the glass-box detector (Gradio)

Draft for review, 2026-11-26. Layers 1-3 only (intent, behaviour, interfaces); no code yet.

## 1. Intent

Build a local web app where anyone can watch the final model (seed 0, `peak_decoy-*`) find signs in a whole
photo, one stage at a time, and try to break it by placing a sign on a photo at any size. The app shows what
the project measured: signs are found when the pyramid brings them into the 16-48 px range, small signs
(< 24 px) are the weak spot, and the model only knows German signs.

**Out of scope:** running in the browser (ONNX), a training replay, video, drawing boxes, US signs, and
**deploying to Hugging Face Spaces**. Deploying is a separate step after the local app works, because it
publishes the weights and images (see open questions).

## 2. Behaviour

### Tab 1: "Watch it look"

- **Input:** one image: an uploaded photo (any size, any format PIL reads), or one of about 8 example GTSDB
  **dev** photos. Plus a score threshold slider, default τ_W = 0.42658 (seed 0, from the stored dev report).
- **Fit:** the photo is shrunk (never enlarged) to fit inside 1360×800, keeping its shape, then padded on
  the right and bottom to a multiple of 16 px, the size `detect()` needs at scale ¼.
- **Output, in order:**
  1. the photo at scales 1, ½ and ¼;
  2. the heatmap at each scale, laid over the photo;
  3. every peak at or above the threshold, before suppression (with a count);
  4. what suppression keeps;
  5. the final answer: a point and class name per detection. For an example photo, the labeled signs are
     drawn too, each marked found or missed.
  - plus a table: class, score, class probability, scale, position.
- Moving the threshold re-filters the stored result. It doesn't run the model again.

### Tab 2: "Try to fool it"

- **Input:** a background (an example dev photo or an upload), a sign class (one GTSRB **train** crop per
  class), its size (8-200 px long side) and its position (sliders that keep the whole sign on the photo).
- **Output:** the photo with the sign pasted in, then the same final answer as stage 5, plus one line saying
  whether the pasted sign was found, and at which scale.

### Acceptance criteria

- **A1, parity:** on any 1360×800 dev photo, the final answer is exactly `detect(model, photo).kept`
  filtered to score ≥ threshold. Same detections, same order.
- **A2, any size:** uploads from 16×16 to 6000×6000, in any shape, run without error, and every reported
  point lies inside the original image (it's reported in the original's pixels, not the fitted ones).
- **A3, fit:** the fitted image's width and height are multiples of 16; the shrink factor is ≤ 1; the
  content fits in 1360×800; a photo that's already 1360×800 is unchanged.
- **A4, paste:** the pasted sign's long side is the requested size (±1 px) and its center is the requested
  point (±1 px). Nothing else in the photo changes.
- **A5, threshold:** changing the threshold never calls the model.
- **A6, no test data:** examples come only from `photos_in(photos, RealSplit.DEV)`, and signs only from the
  GTSRB train split.
- **A7, speed:** time per photo on CPU is measured and reported. A target, not a promise: under 3 s.

### Failure modes

- Missing checkpoint or exported weights: the app refuses to start and names the missing file.
- Grayscale, RGBA or palette images are converted to RGB. A file PIL can't read shows an error message in
  the app; it doesn't crash.
- Padding adds black strips at the right and bottom of odd-sized uploads. A peak could land near that edge;
  this is accepted, and the padding is shown in the stage images.

## 3. Interfaces

```python
# src/findobject/demo/fit.py: shrink and pad any photo to a size detect() accepts. Pure.
FIT_MAX = (1360, 800)
FIT_MULTIPLE = 16  # STRIDE (4) / smallest pyramid scale (1/4)

@dataclass(frozen=True)
class FittedPhoto:
    image: Image.Image  # RGB, sides multiples of FIT_MULTIPLE
    factor: float       # fitted px = original px * factor, 0 < factor <= 1
    original_size: tuple[int, int]

def fit_photo(img: Image.Image) -> FittedPhoto: ...
def to_original(x: float, y: float, fitted: FittedPhoto) -> tuple[float, float]: ...


# src/findobject/demo/trace.py: run detection and keep each scale's heatmap for display.
@dataclass(frozen=True)
class ScaleView:
    scale: float
    heatmap: np.ndarray  # sigmoid probabilities, [h, w]

@dataclass(frozen=True)
class Trace:
    views: tuple[ScaleView, ...]
    result: DetectResult  # straight from detect(), unchanged

def trace(model: PeakCNN, photo: Image.Image) -> Trace: ...


# src/findobject/demo/paste.py: place one sign on a photo. Pure.
def paste_sign(photo: Image.Image, sign: Image.Image, center: tuple[int, int], long_side: int) -> Image.Image: ...


# src/findobject/demo/draw.py: render each stage as an image. Pure (in-memory images only).
def draw_scales(photo: Image.Image) -> Image.Image: ...
def draw_heatmaps(photo: Image.Image, views: Sequence[ScaleView]) -> Image.Image: ...
def draw_points(photo: Image.Image, dets: Sequence[Detection], title: str) -> Image.Image: ...
def draw_answer(photo: Image.Image, dets: Sequence[Detection], truth: Sequence[RealSign] | None) -> Image.Image: ...


# scripts/export_demo_model.py: write demo/assets/model.pt (weights only, about 2.3 MB) and meta.json
#   (run name, τ_W, class names), so the app doesn't depend on outputs/.
# demo/app.py: the Gradio UI. Loads the model once, wires the two tabs to the functions above.
# demo/requirements.txt: gradio pinned to an exact release at least two weeks old, plus the repo's pins.
```

### Design choices to review

- **`trace()` runs the model twice per photo:** once inside `detect()`, unchanged, and once more to keep
  the heatmaps. That leaves the graded code untouched, at the cost of about double the time. The
  alternative is changing `detect()` to also return its heatmaps, which touches tested code that produced
  the graded numbers.
- **New dependency, Gradio:** large (dozens of packages), installed only for the demo (`demo/requirements.txt`),
  not added to the main `requirements.txt`. The alternative, Streamlit, is similar in size. Gradio is the
  native choice for Hugging Face Spaces and has ready-made image inputs and galleries.

## Open questions (needed before deploying, not before building locally)

- **Images in a public Space:** the examples and sign crops come from GTSDB and GTSRB. I haven't checked
  whether their terms allow redistributing images. Local use reads them from `data/raw/` as now.
- **Weights in a public Space:** publishing `model.pt` makes the trained model public.
