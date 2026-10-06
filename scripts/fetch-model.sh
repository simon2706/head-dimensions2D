#!/usr/bin/env bash
# Re-download the MediaPipe Face Landmarker and Iris landmark models into frontend/public/models/.
# The models are committed to the repository; this script is only needed to refresh them.
# Inference always runs locally in the browser - this is a one-time file download.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/frontend/public/models"
mkdir -p "$DIR"

fetch() {
  curl -fL --progress-bar "$1" -o "$DIR/$2"
  echo "Saved $DIR/$2 ($(wc -c <"$DIR/$2" | tr -d ' ') bytes)"
}

fetch "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" face_landmarker.task
# Legacy MediaPipe Iris landmark model (mediapipe/modules/iris_landmark), run with LiteRT.js.
fetch "https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite" iris_landmark.tflite
