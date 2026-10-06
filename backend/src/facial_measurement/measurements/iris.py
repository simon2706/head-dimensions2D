"""Metric scale estimation.

The engine converts pixels to millimetres through a ``ScaleProvider``. Both
implemented providers assume a fixed horizontal visible iris diameter and differ
only in where the iris points come from (the dedicated MediaPipe Iris model or
the Face Landmarker mesh); a reference-object provider (ArUco/AprilTag/
calibration card) can be added later behind the same interface.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.geometry import FaceGeometry, distance


class ScaleResult(BaseModel):
    method: str
    assumed_iris_diameter_mm: float | None = None
    left_iris_diameter_px: float | None = None
    right_iris_diameter_px: float | None = None
    left_iris_vertical_diameter_px: float | None = None
    right_iris_vertical_diameter_px: float | None = None
    left_mm_per_px: float | None = None
    right_mm_per_px: float | None = None
    final_mm_per_px: float
    difference_percent: float | None = None
    landmarks: dict[str, list[int]] = {}
    endpoints_px: dict[str, list[list[float]]] = Field(
        default_factory=dict,
        description="Iris diameter endpoints in image pixels, keyed e.g. 'right_horizontal'.",
    )
    iris_centers_px: dict[str, list[float]] = {}
    fallback_reason: str | None = Field(
        None, description="Why the configured scale source could not be used, if it was not."
    )


class ScaleProvider(Protocol):
    name: str

    def estimate(self, geom: FaceGeometry, config: MeasurementConfig) -> ScaleResult: ...


def iris_diameter_px(geom: FaceGeometry, pair: tuple[int, int]) -> float:
    """Distance between two opposite iris contour landmarks, in pixels."""
    return distance(geom.image_px[pair[0]], geom.image_px[pair[1]])


def mm_per_px(assumed_diameter_mm: float, diameter_px: float) -> float:
    if diameter_px <= 0:
        raise ValueError("iris diameter must be positive")
    return assumed_diameter_mm / diameter_px


def scale_difference_percent(a: float, b: float) -> float:
    """Absolute difference of two scales relative to their mean, in percent."""
    mean = (a + b) / 2
    return abs(a - b) / mean * 100.0


def px_to_mm(value_px: float | None, scale_mm_per_px: float) -> float | None:
    return None if value_px is None else value_px * scale_mm_per_px


def _points(*pts: np.ndarray) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in pts]


def iris_scale_result(
    method: str,
    *,
    right_horizontal: tuple[np.ndarray, np.ndarray],
    left_horizontal: tuple[np.ndarray, np.ndarray],
    right_vertical: tuple[np.ndarray, np.ndarray],
    left_vertical: tuple[np.ndarray, np.ndarray],
    right_center: np.ndarray,
    left_center: np.ndarray,
    config: MeasurementConfig,
    landmarks: dict[str, list[int]] | None = None,
) -> ScaleResult:
    """Per-eye and mean mm/px from the horizontal iris diameters (endpoints in image px)."""
    d_mm = config.assumed_iris_diameter_mm
    left_px = distance(*left_horizontal)
    right_px = distance(*right_horizontal)
    left_scale = mm_per_px(d_mm, left_px)
    right_scale = mm_per_px(d_mm, right_px)
    return ScaleResult(
        method=method,
        assumed_iris_diameter_mm=d_mm,
        left_iris_diameter_px=left_px,
        right_iris_diameter_px=right_px,
        left_iris_vertical_diameter_px=distance(*left_vertical),
        right_iris_vertical_diameter_px=distance(*right_vertical),
        left_mm_per_px=left_scale,
        right_mm_per_px=right_scale,
        final_mm_per_px=(left_scale + right_scale) / 2,
        difference_percent=scale_difference_percent(left_scale, right_scale),
        landmarks=landmarks or {},
        endpoints_px={
            "right_horizontal": _points(*right_horizontal),
            "left_horizontal": _points(*left_horizontal),
            "right_vertical": _points(*right_vertical),
            "left_vertical": _points(*left_vertical),
        },
        iris_centers_px={
            "right": _points(right_center)[0],
            "left": _points(left_center)[0],
        },
    )


class FaceLandmarkerIrisScaleProvider:
    """Scale from the iris points 468-477 of the Face Landmarker mesh.

    Each eye yields its own mm/px scale; the final scale is their mean. The
    left/right mismatch is reported and used as a quality metric.
    """

    name = "face_landmarker_iris"

    def estimate(self, geom: FaceGeometry, config: MeasurementConfig) -> ScaleResult:
        px = geom.image_px

        def pair(ids: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            return px[ids[0]], px[ids[1]]

        return iris_scale_result(
            self.name,
            right_horizontal=pair(T.RIGHT_IRIS_HORIZONTAL),
            left_horizontal=pair(T.LEFT_IRIS_HORIZONTAL),
            right_vertical=pair(T.RIGHT_IRIS_VERTICAL),
            left_vertical=pair(T.LEFT_IRIS_VERTICAL),
            right_center=px[T.RIGHT_IRIS_CENTER],
            left_center=px[T.LEFT_IRIS_CENTER],
            config=config,
            landmarks={
                "left_horizontal": list(T.LEFT_IRIS_HORIZONTAL),
                "right_horizontal": list(T.RIGHT_IRIS_HORIZONTAL),
                "left_vertical": list(T.LEFT_IRIS_VERTICAL),
                "right_vertical": list(T.RIGHT_IRIS_VERTICAL),
            },
        )


def split_iris_pairs(
    geom: FaceGeometry, iris_px: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """(horizontal, vertical) endpoint pairs of a 5-point iris-model result.

    The pair whose direction is closer to the face-frame x axis is horizontal.
    """
    pairs = [(iris_px[a], iris_px[b]) for a, b in T.IRIS_MODEL_OPPOSITE_PAIRS]

    def horizontalness(pair: tuple[np.ndarray, np.ndarray]) -> float:
        a, b = geom.frame.to_face(np.array(pair))
        v = b - a
        return abs(float(v[0])) / max(float(np.hypot(*v)), 1e-9)

    horizontal, vertical = sorted(pairs, key=horizontalness, reverse=True)
    return horizontal, vertical


class IrisModelScaleProvider:
    """Scale from the dedicated MediaPipe Iris landmark model (``iris_landmark.tflite``).

    The model runs in the browser on 64x64 crops of each eye and sends 5 iris
    points per eye (image pixels here). Scale maths are identical to
    ``FaceLandmarkerIrisScaleProvider``; only the iris points differ.
    """

    name = "iris_model"

    def __init__(self, right_iris_px: np.ndarray, left_iris_px: np.ndarray) -> None:
        self.right_iris_px = np.asarray(right_iris_px, dtype=float)
        self.left_iris_px = np.asarray(left_iris_px, dtype=float)
        for pts in (self.right_iris_px, self.left_iris_px):
            if pts.shape != (T.IRIS_MODEL_NUM_IRIS, 2):
                raise ValueError(f"expected {T.IRIS_MODEL_NUM_IRIS} iris points per eye")

    def estimate(self, geom: FaceGeometry, config: MeasurementConfig) -> ScaleResult:
        right_h, right_v = split_iris_pairs(geom, self.right_iris_px)
        left_h, left_v = split_iris_pairs(geom, self.left_iris_px)
        return iris_scale_result(
            self.name,
            right_horizontal=right_h,
            left_horizontal=left_h,
            right_vertical=right_v,
            left_vertical=left_v,
            right_center=self.right_iris_px[T.IRIS_MODEL_CENTER],
            left_center=self.left_iris_px[T.IRIS_MODEL_CENTER],
            config=config,
        )


class ReferenceObjectScaleProvider:
    """Placeholder for calibration against an object of known size.

    A future implementation would receive the detected corners of an ArUco /
    AprilTag marker or calibration card (in image pixels) together with its
    physical size, and return a ``ScaleResult`` with ``method="reference_object"``.
    """

    name = "reference_object"

    def __init__(self, object_width_mm: float, object_width_px: float) -> None:
        self.object_width_mm = object_width_mm
        self.object_width_px = object_width_px

    def estimate(self, geom: FaceGeometry, config: MeasurementConfig) -> ScaleResult:
        raise NotImplementedError("Reference-object calibration is not implemented yet.")
