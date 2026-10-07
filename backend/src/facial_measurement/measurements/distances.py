"""Point-to-point measurements between semantic MediaPipe landmarks."""

from __future__ import annotations

import numpy as np

from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.base import MeasurementType, PixelMeasurement, points_to_list
from facial_measurement.measurements.geometry import FaceGeometry, distance


def landmark_distance(
    geom: FaceGeometry,
    a: int,
    b: int,
    *,
    id: str,
    name: str,
    type: MeasurementType,
    definition: str,
    uses_lateral_contour: bool = False,
    uses_brows: bool = False,
    details: dict | None = None,
) -> PixelMeasurement:
    pa, pb = geom.image_px[a], geom.image_px[b]
    return PixelMeasurement(
        id=id,
        name=name,
        type=type,
        definition=definition,
        formula=f"euclidean(px[{a}], px[{b}])",
        value_px=distance(pa, pb),
        landmarks=[a, b],
        endpoints_px=points_to_list(np.array([pa, pb])),
        uses_lateral_contour=uses_lateral_contour,
        uses_brows=uses_brows,
        details=details or {},
    )


def lateral_brow_points(geom: FaceGeometry) -> tuple[int, int]:
    """IDs of the lateral-most eyebrow contour landmark on each side (face frame)."""
    right = list(T.RIGHT_EYEBROW)
    left = list(T.LEFT_EYEBROW)
    right_id = right[int(np.argmin(geom.aligned[right, 0]))]
    left_id = left[int(np.argmax(geom.aligned[left, 0]))]
    return right_id, left_id


def outer_canthal_width(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    return landmark_distance(
        geom,
        T.RIGHT_EYE_OUTER,
        T.LEFT_EYE_OUTER,
        id="outer_canthal_width",
        name="Outer canthal width",
        type=MeasurementType.DIRECT,
        definition="Distance between the outer (lateral) eye corners.",
    )


def lateral_eye_mesh_width(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    return landmark_distance(
        geom,
        T.RIGHT_EYE_LATERAL_MESH,
        T.LEFT_EYE_LATERAL_MESH,
        id="lateral_eye_mesh_width",
        name="Lateral eye mesh width",
        type=MeasurementType.EXPERIMENTAL_PROXY,
        definition=(
            "Distance between mesh vertices 130 and 359, just lateral to the outer eye "
            "corners. Repeatable mesh feature; not an anatomical landmark distance."
        ),
    )


def inner_canthal_width(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    return landmark_distance(
        geom,
        T.RIGHT_EYE_INNER,
        T.LEFT_EYE_INNER,
        id="inner_canthal_width",
        name="Inner canthal width",
        type=MeasurementType.DIRECT,
        definition="Distance between the inner (medial) eye corners.",
    )


def iris_center_distance(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    return landmark_distance(
        geom,
        T.RIGHT_IRIS_CENTER,
        T.LEFT_IRIS_CENTER,
        id="iris_center_distance",
        name="Iris-center distance",
        type=MeasurementType.DIRECT,
        definition=(
            "Distance between the two iris-centre landmarks. Geometric feature only; not a "
            "clinical (distance) interpupillary distance - it depends on gaze/vergence."
        ),
    )


def outer_brow_span(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    right_id, left_id = lateral_brow_points(geom)
    return landmark_distance(
        geom,
        right_id,
        left_id,
        id="outer_brow_span",
        name="Outer brow span",
        type=MeasurementType.SURFACE_PROXY,
        definition=(
            "Distance between the lateral-most landmarks of the MediaPipe eyebrow contours "
            "(selected per image in the roll-corrected face frame). Semantic contour point, "
            "not the last visible eyebrow hair."
        ),
        uses_brows=True,
        details={
            "right_brow_candidates": list(T.RIGHT_EYEBROW),
            "left_brow_candidates": list(T.LEFT_EYEBROW),
            "selected_right": right_id,
            "selected_left": left_id,
        },
    )


def upper_temporal_mesh_width(geom: FaceGeometry, config: MeasurementConfig) -> PixelMeasurement:
    return landmark_distance(
        geom,
        T.RIGHT_UPPER_TEMPORAL,
        T.LEFT_UPPER_TEMPORAL,
        id="upper_temporal_mesh_width",
        name="Upper temporal mesh width",
        type=MeasurementType.EXPERIMENTAL_PROXY,
        definition=(
            "Distance between face-oval mesh vertices 162 and 389, which lie on the lateral "
            "silhouette at approximately eyebrow-tail height in the canonical face model. "
            "Repeatable mesh feature; NOT anatomical temple breadth."
        ),
        uses_lateral_contour=True,
    )
