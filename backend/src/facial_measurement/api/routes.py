"""HTTP API. Only landmarks and metadata are accepted; images are never received."""

from fastapi import APIRouter

from facial_measurement import __version__
from facial_measurement.config import MeasurementConfig
from facial_measurement.models import MeasureRequest, MeasureResponse
from facial_measurement.pipeline import measure_frame

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/config", response_model=MeasurementConfig)
def default_config() -> MeasurementConfig:
    return MeasurementConfig()


@router.post("/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> MeasureResponse:
    return measure_frame(request)
