"""Synthetic landmark fixtures so tests never depend on MediaPipe itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from facial_measurement import topology as T

IMAGE_W = 1000
IMAGE_H = 1000
CENTER = np.array([500.0, 500.0])
OVAL_HALF_WIDTH = 150.0
OVAL_HALF_HEIGHT = 200.0
IRIS_RADIUS = 10.0


def synthetic_face_px(
    roll_deg: float = 0.0, left_iris_radius: float = IRIS_RADIUS, lid_opening: float = 16.0
) -> np.ndarray:
    """A symmetric, frontal synthetic face in image pixels (478, 2).

    The face oval is an ellipse sampled at the 36 MediaPipe oval landmarks, so the
    oval loop order matches MediaPipe (10 at the top, 152 at index 18 = chin).
    """
    pts = np.tile(CENTER, (T.NUM_LANDMARKS, 1))
    for k, idx in enumerate(T.FACE_OVAL):
        phi = 2 * math.pi * k / len(T.FACE_OVAL)
        pts[idx] = CENTER + [OVAL_HALF_WIDTH * math.sin(phi), -OVAL_HALF_HEIGHT * math.cos(phi)]

    eye_y = 460.0
    pts[T.RIGHT_EYE_OUTER] = [430, eye_y]
    pts[T.RIGHT_EYE_INNER] = [480, eye_y]
    pts[T.LEFT_EYE_INNER] = [520, eye_y]
    pts[T.LEFT_EYE_OUTER] = [570, eye_y]
    pts[T.RIGHT_EYE_LATERAL_MESH] = [420, eye_y + 2]
    pts[T.LEFT_EYE_LATERAL_MESH] = [580, eye_y + 2]

    for center_id, (h0, v0, h1, v1), cx, r in (
        (T.RIGHT_IRIS_CENTER, T.RIGHT_IRIS_CONTOUR, 455.0, IRIS_RADIUS),
        (T.LEFT_IRIS_CENTER, T.LEFT_IRIS_CONTOUR, 545.0, left_iris_radius),
    ):
        pts[center_id] = [cx, eye_y]
        pts[h0] = [cx + r, eye_y]
        pts[v0] = [cx, eye_y - r]
        pts[h1] = [cx - r, eye_y]
        pts[v1] = [cx, eye_y + r]

    half = lid_opening / 2
    pts[T.RIGHT_EYE_UPPER_LID] = [455, eye_y - half]
    pts[T.RIGHT_EYE_LOWER_LID] = [455, eye_y + half]
    pts[T.LEFT_EYE_UPPER_LID] = [545, eye_y - half]
    pts[T.LEFT_EYE_LOWER_LID] = [545, eye_y + half]

    # Eyebrows: 70/300 are the lateral-most points, 46/276 slightly medial and lower.
    brow_x = np.linspace(415, 488, len(T.RIGHT_EYEBROW))
    for i, idx in enumerate(T.RIGHT_EYEBROW):
        pts[idx] = [brow_x[i], 418.0]
    for i, idx in enumerate(T.LEFT_EYEBROW):
        pts[idx] = [2 * CENTER[0] - brow_x[i], 418.0]
    pts[70], pts[300] = [405, 416], [595, 416]
    pts[46], pts[276] = [410, 424], [590, 424]

    pts[T.SUBNASALE] = [500, 560]
    pts[T.NOSE_TIP] = [500, 545]

    if roll_deg:
        a = math.radians(roll_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts = (pts - CENTER) @ rot.T + CENTER
    return pts


def synthetic_iris_model_px(
    roll_deg: float = 0.0,
    right_radius: float = IRIS_RADIUS,
    left_radius: float = IRIS_RADIUS,
    swap_pairs: bool = False,
    vertical_squash: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Iris-model output (5, 2) per eye, (right, left), for ``synthetic_face_px``.

    Contour points go around the iris like the real model (opposite pairs (1, 3) and
    (2, 4)): by default (1, 3) is horizontal; ``swap_pairs`` starts at the top instead. The
    vertical diameter is ``vertical_squash`` x the horizontal one so the pairs are
    distinguishable.
    """
    eye_y = 460.0
    out = []
    for cx, r in ((455.0, right_radius), (545.0, left_radius)):
        rv = r * vertical_squash
        h = [[cx - r, eye_y], [cx + r, eye_y]]
        v = [[cx, eye_y - rv], [cx, eye_y + rv]]
        ring = [v[0], h[0], v[1], h[1]] if swap_pairs else [h[1], v[0], h[0], v[1]]
        pts = np.array([[cx, eye_y], *ring])
        if roll_deg:
            a = math.radians(roll_deg)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            pts = (pts - CENTER) @ rot.T + CENTER
        out.append(pts)
    return out[0], out[1]


def iris_model_payload(right_px: np.ndarray, left_px: np.ndarray) -> dict:
    return {
        "model": "iris_landmark.tflite",
        "right": {"iris": to_normalized(right_px)},
        "left": {"iris": to_normalized(left_px)},
    }


def to_normalized(px: np.ndarray) -> list[dict[str, float]]:
    return [{"x": p[0] / IMAGE_W, "y": p[1] / IMAGE_H, "z": 0.0} for p in px]


IDENTITY_MATRIX = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, -50.0, 1.0]


def make_request(px: np.ndarray | None = None, **overrides) -> dict:
    payload = {
        "image_width": IMAGE_W,
        "image_height": IMAGE_H,
        "landmarks": to_normalized(synthetic_face_px() if px is None else px),
        "face_count": 1,
        "facial_transformation_matrix": IDENTITY_MATRIX,
        "blendshapes": {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05},
        "image_metrics": {"sharpness_laplacian_var": 120.0},
        "iris_model": iris_model_payload(*synthetic_iris_model_px()),
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def face_px() -> np.ndarray:
    return synthetic_face_px()
