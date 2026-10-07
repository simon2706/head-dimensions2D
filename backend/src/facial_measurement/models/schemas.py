"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.base import MeasurementType
from facial_measurement.measurements.iris import ScaleResult
from facial_measurement.measurements.pose import Pose
from facial_measurement.measurements.quality import QualityReport

FEATURE_SCHEMA_VERSION = "1.2"  # 1.1: added scale_method; 1.2: added lateral_eye_mesh_width


class Landmark(BaseModel):
    """MediaPipe normalized landmark (x by image width, y by image height)."""

    x: float
    y: float
    z: float = 0.0


class ImageMetrics(BaseModel):
    """Image statistics computed in the browser (the image itself is never sent)."""

    sharpness_laplacian_var: float | None = None
    sharpness_crop_width_px: int | None = None


class IrisModelEye(BaseModel):
    """MediaPipe Iris landmark model output for one eye (normalized to the full image)."""

    iris: list[Landmark] = Field(description="Iris centre followed by 4 contour points.")
    roi: dict[str, float] | None = Field(
        None, description="Eye crop the model ran on (image px); debug only."
    )

    @field_validator("iris")
    @classmethod
    def _iris_count(cls, v: list[Landmark]) -> list[Landmark]:
        if len(v) != T.IRIS_MODEL_NUM_IRIS:
            raise ValueError(f"expected {T.IRIS_MODEL_NUM_IRIS} iris points, got {len(v)}")
        return v


class IrisModelOutput(BaseModel):
    model: str = "iris_landmark.tflite"
    right: IrisModelEye = Field(description="Subject's right eye.")
    left: IrisModelEye = Field(description="Subject's left eye.")


class MeasureRequest(BaseModel):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    landmarks: list[Landmark]
    face_count: int = Field(1, ge=0, description="Number of faces detected in the image.")
    facial_transformation_matrix: list[float] | None = Field(
        None, description="4x4 matrix as 16 values in column-major order (MediaPipe Matrix.data)."
    )
    blendshapes: dict[str, float] = Field(default_factory=dict)
    image_metrics: ImageMetrics | None = None
    iris_model: IrisModelOutput | None = Field(
        None, description="Output of the MediaPipe Iris landmark model, if it ran."
    )
    config: MeasurementConfig = Field(default_factory=MeasurementConfig)

    @field_validator("landmarks")
    @classmethod
    def _landmark_count(cls, v: list[Landmark]) -> list[Landmark]:
        if len(v) != T.NUM_LANDMARKS:
            raise ValueError(
                f"expected {T.NUM_LANDMARKS} landmarks (468 mesh + 10 iris), got {len(v)}"
            )
        return v

    @field_validator("facial_transformation_matrix")
    @classmethod
    def _matrix_size(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 16:
            raise ValueError("facial_transformation_matrix must have 16 values")
        return v


class Measurement(BaseModel):
    id: str
    name: str
    type: MeasurementType
    value_px: float | None
    value_mm: float | None
    landmarks: list[int]
    endpoints_px: list[list[float]] = Field(
        description="Line endpoints in original image pixel coordinates."
    )
    quality: Literal["good", "warning", "poor"]
    quality_notes: list[str] = []
    definition: str
    formula: str
    level_y_face_px: float | None = None
    details: dict[str, Any] = {}
    error: str | None = None


class Features(BaseModel):
    """Stable feature vector for downstream regression (all in mm)."""

    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    scale_method: str = Field(description="ScaleResult.method that produced the mm values.")
    outer_canthal_width_mm: float | None
    lateral_eye_mesh_width_mm: float | None
    inner_canthal_width_mm: float | None
    iris_center_distance_mm: float | None
    outer_brow_span_mm: float | None
    upper_face_width_at_brow_level_mm: float | None
    upper_temporal_mesh_width_mm: float | None
    face_width_at_eye_level_mm: float | None
    midface_width_proxy_mm: float | None
    maximum_upper_face_width_mm: float | None


class Debug(BaseModel):
    image_width: int
    image_height: int
    face_bbox_px: dict[str, float]
    face_size_px: dict[str, float]
    face_frame: dict[str, Any]
    eye_opening_ratios: dict[str, float]
    landmark_coordinates_px: dict[int, list[float]]
    sharpness_laplacian_var: float | None = None


class MeasureResponse(BaseModel):
    engine_version: str
    scale: ScaleResult
    pose: Pose
    quality: QualityReport
    measurements: list[Measurement]
    features: Features
    config: MeasurementConfig
    debug: Debug
