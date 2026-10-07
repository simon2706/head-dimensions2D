# head-dimensions — facial measurement prototype

An end-to-end **experimental research prototype** that estimates facial/head dimensions from a
frontal photo. MediaPipe Face Landmarker and the MediaPipe Iris landmark model run in the browser.
Pixel distances are converted to millimetres using the iris as an approximate size reference. A Python engine computes a set of
direct measurements and deliberately labelled *proxies*, and runs capture-quality checks on them.

> ⚠️ **Not a medical or anthropometric measurement device.** All values are image-based estimates
> with subject-specific scale uncertainty. Several metrics are *proxies* intended for later
> experimental validation against real head measurements. None of them are skeletal
> anthropometric dimensions.

The goal is a tool that answers a later research question:

> Which combination of frontal-image measurements best predicts the actual width of a person's
> head at the plane where glasses temples sit?

To that end, every capture exports a stable feature vector, all landmarks and every intermediate
value as JSON.

---

## What the application does

1. Upload a frontal photo, or take one with the webcam (live alignment feedback, mirrored preview).
2. Run MediaPipe Face Landmarker **locally in the browser**. It returns 478 landmarks (468 mesh +
   10 iris), 52 blendshapes and a 4×4 facial transformation matrix.
3. Run the dedicated MediaPipe **Iris landmark model** (`iris_landmark.tflite`, via LiteRT.js) on a
   64×64 crop of each eye. It returns 5 iris points and 71 eye-contour points per eye.
4. Send only the landmarks and metadata to the local FastAPI backend. **The photo is never sent.**
5. The backend:
   - converts landmarks to pixels;
   - builds a roll-corrected *face frame*;
   - estimates mm/px for each eye from the assumed iris diameter (Iris-model iris by default);
   - computes 9 measurements in pixels, then converts them to mm;
   - estimates head pose;
   - runs the quality checks.
6. The frontend draws everything over the photo. Each overlay layer can be toggled. Clicking a
   measurement highlights it.
7. A debug panel shows every intermediate value. **Export JSON** saves the complete record without
   the image.

## Architecture

```text
Photo ──► Browser: MediaPipe Face Landmarker (WASM + model served locally)
            │  478 normalized landmarks, transformation matrix, blendshapes,
            │  face-crop sharpness (computed in the browser)
            ├─► MediaPipe Iris landmark model on 64×64 eye crops (LiteRT.js, served locally)
            │  5 iris + 71 eye-contour points per eye
            ▼
          POST /api/measure  (JSON only — no image)
            ▼
          Python measurement engine (FastAPI + Pydantic + numpy)
            landmarks → pixel geometry → measurement_px → ScaleProvider → measurement_mm
            + pose + quality checks
            ▼
          Measurements, feature vector, quality diagnostics, debug data
            ▼
          Browser: SVG overlay, results/quality/debug panels, JSON export
```

**Why MediaPipe runs in the browser.**
- The photo never leaves the user's machine.
- Landmark detection uses the browser's GPU, and live webcam feedback needs low latency.
- The models (`frontend/public/models/face_landmarker.task`, `frontend/public/models/iris_landmark.tflite`)
  and the WASM runtimes (`frontend/public/mediapipe/wasm` and `frontend/public/litert/wasm`, copied
  from `node_modules` on install) are all served by the app itself. No CDN or cloud inference is
  involved.

**Why the measurement logic is in Python.**
- The engine is a pure function of the landmarks (`pipeline.measure_frame`), so it can be
  unit-tested with synthetic data.
- It can be re-run offline on exported JSON.
- It can later take landmarks from another source, aggregate several frames, or swap the scale
  provider, all without touching the UI.
- The UI contains no measurement geometry. It draws the endpoints that the backend returns.
- The webcam's live quality badges also call the backend (throttled to about 3 Hz), so there is
  one source of truth for the checks.

### Repository layout

```text
backend/                         Python measurement engine (uv project)
  src/facial_measurement/
    main.py                      FastAPI app
    api/routes.py                GET /api/health, GET /api/config, POST /api/measure
    config.py                    MeasurementConfig — every threshold lives here
    topology.py                  All MediaPipe landmark IDs, with justification
    pipeline.py                  measure_frame(): the single-frame pipeline
    models/schemas.py            Pydantic request/response models, feature vector
    measurements/
      geometry.py                pixel conversion, distance, roll-corrected FaceFrame
      iris.py                    ScaleProvider protocol, IrisModel/FaceLandmarkerIris scale providers,
                                 ReferenceObject stub
      distances.py               point-to-point measurements
      face_contour.py            contour intersection, width_at_y, max width, contour metrics
      pose.py                    yaw/pitch/roll from the transformation matrix
      quality.py                 PASS/WARNING/FAIL quality checks
  tests/                         pytest, synthetic landmarks only
frontend/                        React + TypeScript + Vite
  src/mediapipe/                 Face Landmarker + Iris model (LiteRT.js) loaders, eye crops,
                                 result conversion, sharpness
  src/api/                       backend client
  src/visualization/             SVG overlay, styles, label layout
  src/components/                panels, webcam capture, config editor
  src/export/                    JSON export
  src/types/                     TS mirrors of backend schemas
  public/models/                 face_landmarker.task (~3.7 MB) and iris_landmark.tflite (~2.6 MB), committed
  scripts/setup-assets.mjs       copies MediaPipe and LiteRT.js WASM into public/ on npm install
scripts/
  dev.sh                         start backend + frontend
  fetch-model.sh                 re-download the Face Landmarker and Iris landmark models
```

## Installation

Requirements:
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically if needed.
- Node.js ≥ 20 with npm.
- A Chromium-based browser or Firefox. WebGL is recommended.

```bash
git clone <repo-url> head-dimensions
cd head-dimensions

cd backend
uv sync                 # creates .venv and installs locked dependencies
cd ../frontend
npm install             # also copies the MediaPipe and LiteRT.js WASM runtimes into public/
```

The model files are committed. If they are missing, run `./scripts/fetch-model.sh`.

## Running

Option 1: start both servers with one command (Ctrl+C stops both):

```bash
./scripts/dev.sh
```

Option 2: start them separately, in two terminals:

```bash
cd backend
uv run uvicorn facial_measurement.main:app --reload
```

```bash
cd frontend
npm run dev
```

Then open <http://localhost:5173>. The Vite dev server proxies `/api` to `http://127.0.0.1:8000`;
set `BACKEND_URL` to use another backend address. Interactive API docs are at
<http://127.0.0.1:8000/docs>.

### Tests and checks

```bash
cd backend
uv run pytest                 # measurement engine
uv run ruff check .           # lint

cd frontend
npm test                      # vitest unit tests
npm run build                 # type check + production build
npm run lint                  # oxlint
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + engine version |
| GET | `/api/config` | default `MeasurementConfig` |
| POST | `/api/measure` | landmarks + metadata → scale, pose, quality, measurements, features, debug |

Request body for `POST /api/measure`:

```json
{
  "image_width": 1920,
  "image_height": 2560,
  "landmarks": [{"x": 0.51, "y": 0.23, "z": -0.01}],
  "face_count": 1,
  "facial_transformation_matrix": ["16 numbers, column-major"],
  "blendshapes": {"eyeBlinkLeft": 0.03},
  "image_metrics": {"sharpness_laplacian_var": 607.2, "sharpness_crop_width_px": 256},
  "iris_model": {
    "model": "iris_landmark.tflite",
    "right": {"iris": ["5 normalized points: centre, then 4 contour points"], "roi": {"cx": 365, "cy": 177, "size": 81, "rotation": 0.02}},
    "left": {"iris": ["..."], "roi": null}
  },
  "config": {"assumed_iris_diameter_mm": 11.7, "scale_source": "iris_model"}
}
```

Notes on the fields:
- `landmarks` must contain exactly 478 points.
- `iris_model` is optional. When present, each eye must have exactly 5 iris points. The 71
  eye-contour points are kept in the JSON export only.
- `config` is a partial override; omitted keys use the defaults. Unknown keys are rejected.

---

## Coordinate conventions

- **Sides:** "Left" and "right" always mean the **subject's** left and right, as in MediaPipe's
  naming. In a non-mirrored photo the subject's right side appears on the image's left.
- **Landmark coordinates:** landmarks are normalized to the original (non-mirrored) image. The
  webcam preview is mirrored with CSS only. The captured frame and all coordinates are not
  mirrored.
- **Face frame:** the image is rotated about the outer-canthal midpoint so that the line from
  landmark 33 to 263 is horizontal. All "horizontal levels" and contour intersections are computed
  in this frame, which removes small in-plane roll from the widths. Endpoints are rotated back to
  image pixels for display.
- **Z coordinate:** MediaPipe z is **never** used for any measurement. It is not metric depth.

## Iris-based metric scale

### Where the iris points come from

There are two sources of iris landmarks. `scale_source` in the configuration selects one:

- **`iris_model` (default): the dedicated MediaPipe Iris landmark model.** This is the model of the
  [MediaPipe Iris](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md)
  solution (`mediapipe/modules/iris_landmark/iris_landmark.tflite`). It is not part of the Face
  Landmarker bundle, so the browser runs it separately with LiteRT.js. The crop reproduces
  `iris_landmark_landmarks_to_roi.pbtxt`:
  - The eye region is a square centred between two eye corners: 33→133 for the subject's right eye,
    362→263 for the left.
  - The square is rotated along the corner line. Its side is 2.3 × the corner span.
  - It is resized to 64×64. The subject's left eye is mirrored, because the model was trained on
    one eye only.
  - The model returns 5 iris points (centre plus 4 contour points) and 71 eye-contour points, which
    are projected back to the image.
- **`face_landmarker`: Face Landmarker points 468–477.** These come from the iris head of the face
  mesh model, which sees the whole face at 256×256.

**Why the dedicated model is the default.**
- It sees each eye at a much higher effective resolution: a 64-px input covers about 2.3 eye widths.
- On a test portrait, the mesh iris came out about 10 % larger than the Iris-model iris
  (17.2 vs 15.5 px). The mesh-based iris-centre distance was then a low 58.7 mm, against 65.0 mm
  with the Iris model.
- This is a single image, not a validation.

**Fallback.** If the Iris model is selected but its output is missing (it failed to load, or the
input is an older export), the backend falls back to `face_landmarker`. The `scale_source` quality
check then reports a warning, and `scale.fallback_reason` says why. Changing `scale_source` in the
UI re-runs only the backend, because the frontend always sends both sources.

**Iris-model point pairs.** The 4 contour points go around the iris in the same order as 469–472,
so the opposite pairs are (1, 3) and (2, 4). MediaPipe's own `iris_to_depth_calculator.cc` measures
(1, 2) and (3, 4). Those are adjacent points, so that pairing is not used here. For each eye, the
pair closest to the face-frame x axis is taken as horizontal.

### Scale

For each eye, the horizontal iris diameter is the distance between the two horizontal iris contour
points:
- Face Landmarker: 469 ↔ 471 for the subject's right eye, 474 ↔ 476 for the left.
- Iris model: the horizontal opposite pair.

The vertical pairs are reported for debugging only, because the eyelids often cover the iris
vertically. The scale is then:

```text
left_mm_per_px  = assumed_iris_diameter_mm / left_iris_diameter_px
right_mm_per_px = assumed_iris_diameter_mm / right_iris_diameter_px
final_mm_per_px = mean(left, right)
difference_percent = |left − right| / mean(left, right) × 100
```

The default iris diameter is 11.7 mm and can be changed in the UI's configuration panel. Both
per-eye results are always shown. The scale is computed through a `ScaleProvider` interface, so a
`ReferenceObjectScaleProvider` (ArUco, AprilTag or a calibration card) can replace it later. A stub
for it already exists. `scale.method` (`iris_model` or `face_landmarker_iris`) records which
source produced a result. `scale.endpoints_px` holds the diameter endpoints that the overlay
draws.

### Iris calibration limitation

Human horizontal visible iris diameter varies between people, by roughly ±0.5 mm (1 SD) around
11.7 mm. Assuming a fixed value therefore introduces a **subject-specific scale error of about
±4–5 % (1 SD)**. This error applies to every mm value from that capture, and it does not average
out across frames of the same person. MediaPipe's iris contour is also a model estimate, not an
edge measurement. The left/right agreement check catches detection and pose problems, but it
cannot catch an atypical iris size. Both iris sources are model estimates and differ by several
percent from each other, so record which one was used. For validation work, record a reference-object scale alongside
the photo.

### Perspective limitation

The scale is measured at the eyes, so it is correct only for structures at roughly the same depth
as the irises:
- **Eye and brow-region features are the most reliable:** canthal widths, iris-centre distance and
  brow span.
- **Widths that reach the lateral face contour are less reliable.** The temples, cheeks and the
  face oval at the brow or eye level lie further from the camera than the irises, and they are
  partly silhouette edges whose position depends on yaw, focal length and hair.
- **Camera distance matters.** A close selfie (wide-angle, short distance) shrinks lateral widths
  relative to the eye-based scale. That is why the capture instructions recommend 1–1.5 m.

Lateral-contour measurements are flagged as `warning` once |yaw| exceeds
`lateral_measurement_max_yaw_deg` (2.5° by default), even if the global yaw check passes.

## Measurements

Every measurement returns:
- `id`, `name`, `type`, `value_px`, `value_mm`;
- the exact `landmarks` used and `endpoints_px` (image pixels);
- a `definition` and a `formula`;
- `quality` (`good`/`warning`/`poor`) with `quality_notes`;
- calculation `details`, such as the interpolated contour segments and their parameters.

Contour widths use `width_at_y(face_contour, target_y)`. It splits the MediaPipe face oval into
the subject's right chain and left chain (forehead 10 → chin 152) and linearly interpolates the
contour segment that crosses the horizontal line on each side. If a chain crosses the line more
than once, the most lateral crossing is used. Both intersection points are returned for rendering.

| Metric (`id`) | Type | Definition | Landmarks | Known limitations |
|---|---|---|---|---|
| **Outer canthal width** (`outer_canthal_width`) | direct | Distance between the outer eye corners | 33 ↔ 263 | Corner localisation is affected by squinting and by eyelid shape. |
| **Lateral eye mesh width** (`lateral_eye_mesh_width`) | experimental proxy | Distance between mesh vertices 130 and 359, the first points of MediaPipe's `rightEyeLower1` / `leftEyeLower1` rings, just lateral to the outer eye corners | 130 ↔ 359 | Mesh vertex, not a canthus; follows eyelid/periocular shape and squinting. |
| **Inner canthal width** (`inner_canthal_width`) | direct | Distance between the inner eye corners | 133 ↔ 362 | Small value, so the relative error is larger; the caruncle region is low-contrast. |
| **Iris-center distance** (`iris_center_distance`) | direct | Distance between the iris centres | 468 ↔ 473 | Depends on gaze and vergence; **not** a clinical distance PD. |
| **Outer brow span** (`outer_brow_span`) | surface proxy | Distance between the lateral-most landmark of each MediaPipe eyebrow contour (picked per image in the face frame) | Right candidates {46,53,52,65,55,70,63,105,66,107}, left {276,283,282,295,285,300,293,334,296,336}; usually 70 ↔ 300 | Semantic contour point, not the last visible hair. Biased by brow raising or frowning. |
| **Upper face width @ brow level** (`upper_face_width_at_brow_level`) | surface proxy | Face-oval width on the horizontal line at the mean height of the two outer-brow points | Brow points + interpolated oval segments (typically 162–127 and 389–356) | Silhouette edge; hair, yaw and perspective affect it. |
| **Upper temporal mesh width** (`upper_temporal_mesh_width`) | experimental proxy | Euclidean distance between face-oval vertices 162 and 389 | 162 ↔ 389 | Fixed mesh vertices on the silhouette; **not** anatomical temple breadth. |
| **Face width @ eye level** (`face_width_at_eye_level`) | surface proxy | Face-oval width on the horizontal line at the mean height of the outer canthi | 33, 263 + interpolated oval segments (typically 127–234 and 356–454) | Silhouette edge; yaw-sensitive. |
| **Midface width proxy** (`midface_width_proxy`) | experimental proxy | Face-oval width at level `eye_y + f·(y[2] − eye_y)`, with f = `midface_level_fraction` = 0.4 | 33, 263, 2 + interpolated oval segments (typically 234–93 and 454–323) | Reproducible cheek-region level; **not** bizygomatic breadth. |
| **Maximum upper-face width** (`maximum_upper_face_width`) | surface proxy | Maximum oval width over levels from the brow level down to the midface level | Brow points, 33, 263, 2 + oval segments at the maximising level | Visible silhouette maximum, not a skeletal measurement. |

**Maximum upper-face width.** Contour width is piecewise linear in y, with breakpoints at the
contour vertex levels. The search therefore evaluates `max_width_samples` uniform levels plus
every vertex level inside the range. This finds the exact maximum, not a sampled approximation.
The response also reports the level y and both intersection points.

### Landmark choices for the upper temporal mesh width and the midface level

These were checked against MediaPipe's official canonical face model
(`mediapipe/modules/face_geometry/data/canonical_face_model.obj`, units in cm, y up). In that model
the outer canthi are at y = 2.66 and the brow tails (46/70) at y = 3.88–4.25.

- **Upper temporal mesh width uses 162/389.** They sit at y = 4.11 and |x| = 7.56: on the lateral
  silhouette at brow-tail height, which is approximately where glasses temples pass. The
  neighbouring vertices were rejected:
  - 127/356 (y = 2.36) are at eye level and would duplicate the eye-level width.
  - 21/251 (y = 5.43) are on the upper forehead.
- **The midface level uses landmark 2** (lowest midline nose point, ≈ subnasale, y = −2.09). It
  moves less with pitch than the nose tip (1). With f = 0.4 the level lands at canonical
  y ≈ 0.76, the height of 234/454, which is the widest lateral part of the oval in the cheek
  region.

### Feature vector

`features` holds the 10 mm values under stable keys, plus `feature_schema_version` (currently
`1.2`) and `scale_method` (added in 1.1; it records which iris source produced the mm values).
`lateral_eye_mesh_width_mm` was added in 1.2:
- `outer_canthal_width_mm`
- `lateral_eye_mesh_width_mm`
- `inner_canthal_width_mm`
- `iris_center_distance_mm`
- `outer_brow_span_mm`
- `upper_face_width_at_brow_level_mm`
- `upper_temporal_mesh_width_mm`
- `face_width_at_eye_level_mm`
- `midface_width_proxy_mm`
- `maximum_upper_face_width_mm`

The values are deliberately not combined into any prediction.

### Anthropometric terminology

The metrics `midface_width_proxy`, `upper_temporal_mesh_width`, `upper_face_width_at_brow_level`
and `maximum_upper_face_width` are **not** claims of true:
- bizygomatic breadth (MediaPipe cannot locate zygion; no landmark is called zygion here);
- maximum head breadth (euryon is usually hidden behind hair and not visible frontally);
- skeletal temple / bifrontotemporal breadth.

They remain repeatable image features until they are validated experimentally against
ground-truth measurements.

## Capture-quality checks

Each check reports `pass`, `warning` or `fail`. A check whose input is missing reports `skipped`.
The overall result is the worst individual status.

Failed captures are still measured, so they can be inspected. Their measurements are flagged
`poor`, and the UI shows a retake banner. All thresholds are in `MeasurementConfig`
(`backend/src/facial_measurement/config.py`). They are heuristics, not validated limits.

| Check | Default thresholds |
|---|---|
| One face | exactly 1 face, otherwise fail |
| Head roll / yaw / pitch | warn above 3° / 5° / 5°, fail above 6° / 10° / 10° |
| Scale source | warning when the Iris model was selected but its output is missing (fallback to Face Landmarker iris) |
| Iris agreement | warn above 5 %, fail above 8 % |
| Iris resolution | smallest iris diameter: warn below 20 px, fail below 10 px |
| Eyes open | lid opening ÷ iris diameter: warn below 0.55, fail below 0.35; blink blendshape: warn above 0.4, fail above 0.6 |
| Face not clipped | forehead (10), chin (152) and the whole oval stay at least 1 % inside the border; otherwise fail |
| Neutral expression | brow-raise, brow-down or squint blendshapes above 0.5 / 0.5 / 0.6 give a warning (affects brow measurements) |
| Image sharpness | Laplacian variance of the face crop resampled to 256 px wide: warn below 30 (diagnostic only) |

**Head pose.** Pose comes from the MediaPipe facial transformation matrix (Euler order
R = Rz(roll)·Ry(yaw)·Rx(pitch)). The sign conventions are:
- yaw > 0: the nose turns towards the image right;
- pitch > 0: the chin goes down;
- roll > 0: the image-right eye goes up.

The angles are measured **relative to the camera→face line of sight**, not to the optical axis.
A face that looks straight into the lens but sits in the upper part of the frame is rotated about
10–15° relative to the optical axis. Without this correction it would wrongly fail the pitch check
(observed on a standard portrait: 11.6° raw → −3.5° corrected). The correction uses the matrix
translation, which depends on MediaPipe's assumed camera field of view, so it is approximate.

The raw camera-axis angles, the translation and a roll value from the eye-corner line are all
shown in the debug panel as cross-checks.

## Export JSON

Export JSON writes one file per capture. It contains:
- export schema version and timestamp;
- source (upload or webcam, plus the file name);
- image width and height;
- the MediaPipe model description, face count, all 478 normalized landmarks, the column-major
  transformation matrix and all 52 blendshapes;
- `iris_model`: for each eye, the 5 iris points, the 71 eye-contour points and the eye crop (ROI),
  or `null` if the model did not run (export schema 1.1);
- image metrics (sharpness);
- the complete measurement configuration;
- iris calibration, pose, quality checks, every measurement with its landmarks, endpoints and
  details, the feature vector and the debug data.

**The image is never embedded.**

An export can be re-measured offline. Build a `MeasureRequest` from `image`,
`mediapipe.landmarks_normalized`, the matrix and the blendshapes, then call
`facial_measurement.pipeline.measure_frame`. To use the Iris-model scale, add `iris_model` with
the `iris_normalized` points of each eye as `iris`.

## Known limitations

- Iris-size and perspective limitations, as described above.
- Contour widths come from the face-oval *model* output, not from detected image edges. Hair,
  beards, ears and background contrast can shift them.
- The expression and eye-open thresholds were set by eye on a few images. A broad smile, for
  example, raises `browDown` and `eyeSquint` scores, which produces an expression warning.
- Sharpness thresholds depend on the camera and resolution.
- The webcam flow is intentionally basic. It shows live pose, iris and eye badges, and a capture
  button. It does not estimate camera distance.
- Single frame only. The pure `measure_frame` function is designed to be wrapped by a multi-frame
  median aggregator later.

## Suggested first validation experiment

Recruit **30–50 adults**, varied in sex, age, ethnicity, head size and hair style.

**Ground truth, per participant:**
- Head width at the glasses plane: spreading calipers at the height where glasses temples rest,
  about 10 mm above the ear-root junction. Take 3 repeats and record the mean and SD.
- Horizontal visible iris diameter, ideally from a ruler-calibrated close-up or a clinical device.
- Bitragion and bizygomatic breadth, as anatomical anchors.
- Optionally, the frame width that fits the participant.

**Captures, per participant:**
- 5 photos at 1.0 m and 5 at 1.5 m, with the camera at eye height and a fixed camera and focal
  length.
- Include an ArUco or credit-card-sized reference held at eye depth in some of the frames.
- Export JSON for every frame.

**Analysis:**
1. **Repeatability.** Compute the within-subject coefficient of variation (CV) of each feature
   across frames. Drop features with a high CV.
2. **Scale error.** Iris-scaled mm compared with reference-scaled mm, per participant. This
   isolates the error from the fixed-iris assumption.
3. **Univariate.** Pearson or Spearman correlation and Bland–Altman analysis of each feature
   against the glasses-plane width.
4. **Multivariate.** Ridge regression or PLS on the feature vector, with leave-one-subject-out
   cross-validation. Compare models on RMSE and MAE in mm against a baseline that predicts the
   population mean. Also compare models using only direct eye features against models that add
   the contour proxies.
5. **Ratio features.** Test ratio features (for example `upper_face_width_at_brow_level /
   outer_canthal_width`), which cancel the iris scale error.

---

## Development notes

- Python dependencies are managed only with **uv** (`uv add`, `uv sync`). There is no pip or
  requirements.txt.
- Every threshold belongs in `MeasurementConfig`. Every landmark ID belongs in `topology.py`.
- Each new measurement is a small function that returns a `PixelMeasurement`. Register it in
  `pipeline.py`, and add its key to `Features` only if it should join the stable feature vector.
  If you change existing feature semantics, bump `FEATURE_SCHEMA_VERSION`.
