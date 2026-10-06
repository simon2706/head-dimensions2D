#!/usr/bin/env bash
# Start the FastAPI backend (:8000) and the Vite frontend (:5173) together.
# Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  (cd "$ROOT/frontend" && npm install)
fi
(cd "$ROOT/backend" && uv sync --quiet)

cleanup() {
  trap - INT TERM EXIT
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(cd "$ROOT/backend" && uv run uvicorn facial_measurement.main:app --reload --host 127.0.0.1 --port 8000) &
(cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1) &

echo "Backend:  http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:5173"
wait
