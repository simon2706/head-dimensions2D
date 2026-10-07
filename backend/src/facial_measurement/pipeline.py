"""Single-frame measurement pipeline.

landmarks -> pixel geometry -> measurement_px -> scale -> measurement_mm,
plus pose and quality checks. ``measure_frame`` is pure (no I/O) so it can be
reused for offline datasets or wrapped by a future multi-frame aggregator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from facial_measurement import __version__
from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements import distances, face_contour
from facial_measurement.measurements.base import PixelMeasurement
from facial_measurement.measurements.face_contour import FaceContour, build_face_contour
from facial_measurement.measurements.geometry import FaceGeometry, normalized_to_pixels
from facial_measurement.measurements.iris import (
    FaceLandmarkerIrisScaleProvider,
    IrisModelScaleProvider,
    ScaleProvider,
    ScaleResult,
    px_to_mm,
)
from facial_measurement.measurements.pose import Pose, estimate_pose
from facial_measurement.measurements.quality import (
    QualityReport,
    Status,
    eye_opening_ratios,
    run_quality_checks,
)
from facial_measurement.models.schemas import (
    Debug,
    Features,
    IrisModelEye,
    Measurement,
    MeasureRequest,
    MeasureResponse,
)

PointMeasurementFn = Callable[[FaceGeometry, MeasurementConfig], PixelMeasurement]
ContourMeasurementFn = Callable[[FaceGeometry, FaceContour, MeasurementConfig], PixelMeasurement]

POINT_MEASUREMENTS: list[PointMeasurementFn] = [
    distances.outer_canthal_width,
    distances.lateral_eye_mesh_width,
    distances.inner_canthal_width,
    distances.iris_center_distance,
    distances.outer_brow_span,
    distances.upper_temporal_mesh_width,
]
CONTOUR_MEASUREMENTS: list[ContourMeasurementFn] = [
    face_contour.upper_face_width_at_brow_level,
    face_contour.face_width_at_eye_level,
    face_contour.midface_width_proxy,
    face_contour.maximum_upper_face_width,
]
# Order of the exported feature vector (matches ``Features`` fields).
FEATURE_IDS = [
    "outer_canthal_width",
    "lateral_eye_mesh_width",
    "inner_canthal_width",
    "iris_center_distance",
    "outer_brow_span",
    "upper_face_width_at_brow_level",
    "upper_temporal_mesh_width",
    "face_width_at_eye_level",
    "midface_width_proxy",
    "maximum_upper_face_width",
]
# Checks whose failure invalidates every measurement.
CORE_CHECKS = {
    "face_count",
    "head_roll",
    "head_yaw",
    "head_pitch",
    "iris_agreement",
    "iris_resolution",
    "eyes_open",
    "face_not_clipped",
}


def compute_pixel_measurements(
    geom: FaceGeometry, config: MeasurementConfig
) -> list[PixelMeasurement]:
    contour = build_face_contour(geom)
    return [fn(geom, config) for fn in POINT_MEASUREMENTS] + [
        fn(geom, contour, config) for fn in CONTOUR_MEASUREMENTS
    ]


def grade_measurement(
    m: PixelMeasurement, quality: QualityReport, pose: Pose, config: MeasurementConfig
) -> tuple[str, list[str]]:
    by_id = {c.id: c for c in quality.checks}
    notes: list[str] = []
    level = 0  # 0 good, 1 warning, 2 poor
    if m.error:
        return "poor", [m.error]
    for cid in sorted(CORE_CHECKS):
        c = by_id.get(cid)
        if c is None:
            continue
        if c.status == Status.FAIL:
            level = 2
            notes.append(f"{c.label}: fail")
        elif c.status == Status.WARNING:
            level = max(level, 1)
            notes.append(f"{c.label}: warning")
    expr = by_id.get("neutral_expression")
    if m.uses_brows and expr is not None and expr.status == Status.WARNING:
        level = max(level, 1)
        notes.append("Non-neutral brow/eye expression")
    if (
        m.uses_lateral_contour
        and pose.yaw_deg is not None
        and abs(pose.yaw_deg) > config.lateral_measurement_max_yaw_deg
    ):
        level = max(level, 1)
        notes.append(
            f"|yaw| {abs(pose.yaw_deg):.1f}° > {config.lateral_measurement_max_yaw_deg:g}° "
            "for lateral-contour measurement"
        )
    return ("good", "warning", "poor")[level], notes


def to_measurement(
    m: PixelMeasurement, scale_mm_per_px: float, grade: str, notes: list[str]
) -> Measurement:
    return Measurement(
        id=m.id,
        name=m.name,
        type=m.type,
        value_px=m.value_px,
        value_mm=px_to_mm(m.value_px, scale_mm_per_px),
        landmarks=m.landmarks,
        endpoints_px=m.endpoints_px,
        quality=grade,  # type: ignore[arg-type]
        quality_notes=notes,
        definition=m.definition,
        formula=m.formula,
        level_y_face_px=m.level_y_face_px,
        details=m.details,
        error=m.error,
    )


def build_features(measurements: list[Measurement], scale: ScaleResult) -> Features:
    by_id = {m.id: m.value_mm for m in measurements}
    return Features(
        scale_method=scale.method, **{f"{mid}_mm": by_id.get(mid) for mid in FEATURE_IDS}
    )


def select_scale_provider(request: MeasureRequest) -> tuple[ScaleProvider, str | None]:
    """Scale provider for ``config.scale_source``, plus a fallback reason if it is unavailable."""
    if request.config.scale_source == "face_landmarker":
        return FaceLandmarkerIrisScaleProvider(), None
    if request.iris_model is None:
        return FaceLandmarkerIrisScaleProvider(), "Iris landmark model output is missing."

    def to_px(eye: IrisModelEye) -> np.ndarray:
        raw = np.array([[p.x, p.y] for p in eye.iris])
        return normalized_to_pixels(raw, request.image_width, request.image_height)

    return IrisModelScaleProvider(
        to_px(request.iris_model.right), to_px(request.iris_model.left)
    ), None


def build_debug(
    geom: FaceGeometry,
    request: MeasureRequest,
    measurements: list[Measurement],
    scale_landmarks: list[int],
    eye_ratios: tuple[float, float],
) -> Debug:
    px = geom.image_px
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    relevant = sorted(
        {i for m in measurements for i in m.landmarks}
        | set(scale_landmarks)
        | {T.RIGHT_IRIS_CENTER, T.LEFT_IRIS_CENTER, T.FOREHEAD_TOP, T.CHIN}
    )
    return Debug(
        image_width=request.image_width,
        image_height=request.image_height,
        face_bbox_px={
            "x": float(x0),
            "y": float(y0),
            "width": float(x1 - x0),
            "height": float(y1 - y0),
        },
        face_size_px={
            "bbox_width": float(x1 - x0),
            "bbox_height": float(y1 - y0),
            "bbox_diagonal": float(np.hypot(x1 - x0, y1 - y0)),
        },
        face_frame={
            "origin_px": [float(geom.frame.origin[0]), float(geom.frame.origin[1])],
            "roll_correction_deg": float(np.degrees(geom.frame.roll_rad)),
            "description": "Image rotated about the outer-canthal midpoint so 33-263 is level.",
        },
        eye_opening_ratios={"right": eye_ratios[0], "left": eye_ratios[1]},
        landmark_coordinates_px={i: [float(px[i, 0]), float(px[i, 1])] for i in relevant},
        sharpness_laplacian_var=request.image_metrics.sharpness_laplacian_var
        if request.image_metrics
        else None,
    )


def measure_frame(
    request: MeasureRequest, scale_provider: ScaleProvider | None = None
) -> MeasureResponse:
    config = request.config
    fallback_reason = None
    if scale_provider is None:
        scale_provider, fallback_reason = select_scale_provider(request)

    raw = np.array([[lm.x, lm.y] for lm in request.landmarks])
    geom = FaceGeometry.from_pixels(
        normalized_to_pixels(raw, request.image_width, request.image_height)
    )

    scale = scale_provider.estimate(geom, config)
    if fallback_reason:
        scale = scale.model_copy(update={"fallback_reason": fallback_reason})
    pose = estimate_pose(geom, request.facial_transformation_matrix)
    quality = run_quality_checks(
        geom=geom,
        scale=scale,
        pose=pose,
        face_count=request.face_count,
        image_width=request.image_width,
        image_height=request.image_height,
        blendshapes=request.blendshapes,
        sharpness=request.image_metrics.sharpness_laplacian_var if request.image_metrics else None,
        config=config,
    )

    measurements = []
    for pm in compute_pixel_measurements(geom, config):
        grade, notes = grade_measurement(pm, quality, pose, config)
        measurements.append(to_measurement(pm, scale.final_mm_per_px, grade, notes))

    scale_landmarks = [i for ids in scale.landmarks.values() for i in ids]
    return MeasureResponse(
        engine_version=__version__,
        scale=scale,
        pose=pose,
        quality=quality,
        measurements=measurements,
        features=build_features(measurements, scale),
        config=config,
        debug=build_debug(
            geom, request, measurements, scale_landmarks, eye_opening_ratios(geom, scale)
        ),
    )
