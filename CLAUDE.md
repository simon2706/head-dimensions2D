# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is an experimental prototype that estimates facial and head dimensions from a frontal photo:
- MediaPipe Face Landmarker and the MediaPipe Iris landmark model (`iris_landmark.tflite`, via
  LiteRT.js) run in the browser.
- Landmarks (never the image) go to a Python measurement engine.
- Iris-based mm/px scaling converts pixel values to mm.

See README.md for the measurement definitions and limitations.

## Commands

Backend (Python, managed **only** with uv; never use pip):

```bash
cd backend
uv sync
uv run uvicorn facial_measurement.main:app --reload    # API on :8000
uv run pytest                                          # all tests
uv run pytest tests/test_face_contour.py::test_width_at_y   # single test
uv run ruff check . && uv run ruff format .
```

Frontend (React + TS + Vite, npm):

```bash
cd frontend
npm install          # postinstall copies MediaPipe + LiteRT.js WASM to public/
npm run dev          # :5173, proxies /api to :8000
npm test             # vitest
npx vitest run src/mediapipe/detection.test.ts   # single test file
npm run build        # tsc -b + vite build
npm run lint         # oxlint
```

To start both: `./scripts/dev.sh`. To re-download the models: `./scripts/fetch-model.sh`.

## Architecture

The pipeline runs in this order:
1. Browser: `frontend/src/mediapipe/`, giving 478 landmarks, the transformation matrix and
   blendshapes (`faceLandmarker.ts`), plus 5 iris + 71 eye-contour points per eye from the Iris
   model on 64×64 eye crops (`irisLandmark.ts`).
2. `POST /api/measure`.
3. `backend/src/facial_measurement/pipeline.py:measure_frame`, a pure function.
4. Response: scale, pose, quality, measurements and features.
5. The SVG overlay draws the `endpoints_px` returned by the backend. The UI contains no
   measurement geometry.

Backend layout:
- `config.py` (`MeasurementConfig`) holds every threshold.
- `topology.py` holds every landmark ID, with its justification.
- `measurements/` contains small pure functions. Pixel geometry is kept separate from the mm
  conversion, which goes through a `ScaleProvider` in `iris.py`. `config.scale_source` selects
  the Iris-model iris (default) or the Face Landmarker iris (468–477).
- Horizontal levels and contour intersections are computed in a roll-corrected face frame
  (`geometry.FaceFrame`).
- Pose (`pose.py`) is measured relative to the camera→face line of sight.
- Quality checks (`quality.py`) return pass, warning, fail or skipped.

Conventions:
- "left" and "right" mean the subject's sides (MediaPipe naming).
- MediaPipe z is never used.
- Proxies must not be named as anatomical measurements (no "zygion", "bizygomatic" or similar).
- The `Features` keys form a stable export schema. Bump `FEATURE_SCHEMA_VERSION` when their
  semantics change.
