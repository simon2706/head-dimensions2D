"""FastAPI application entry point: ``uvicorn facial_measurement.main:app``."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facial_measurement import __version__
from facial_measurement.api.routes import router

app = FastAPI(
    title="Facial Measurement Prototype",
    version=__version__,
    description="Experimental measurement engine for MediaPipe Face Landmarker landmarks. "
    "Not a medical or anthropometric measurement device.",
)
# The Vite dev server proxies /api, so CORS is only needed when the frontend is
# served from a different origin (e.g. `vite preview`).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
