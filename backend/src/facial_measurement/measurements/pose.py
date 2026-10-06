"""Head pose from MediaPipe's facial transformation matrix.

The matrix maps the canonical face model into MediaPipe's metric camera space
(right-handed, OpenGL-style: x to the image right, y up, z towards the viewer).
It is sent from the browser as 16 numbers in column-major order (the layout of
``Matrix.data`` in ``@mediapipe/tasks-vision``).

Angles are extracted from the rotation part as R = Rz(roll) @ Ry(yaw) @ Rx(pitch):

* yaw   > 0: nose turns towards the image right (subject turns to their left)
* pitch > 0: chin moves down / face tilts forward
* roll  > 0: head tilts so the subject's left eye (image right) moves up

Line-of-sight correction: a face that looks straight into the lens but sits
away from the image centre is rotated relative to the camera *axis* by the
angle between that axis and the camera->face ray (e.g. ~10° pitch for a face in
the upper third of a portrait). What matters for frontal measurements is the
rotation relative to the *line of sight*, so the reported yaw/pitch/roll are
computed after removing the rotation that takes the optical axis onto the
camera->face ray (from the matrix translation). The raw camera-axis angles are
kept for debugging. The translation depends on MediaPipe's assumed virtual
camera field of view, so the correction is approximate.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from facial_measurement import topology as T
from facial_measurement.measurements.geometry import FaceGeometry


class Pose(BaseModel):
    # Relative to the camera->face line of sight (used by the quality checks).
    yaw_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    source: str
    landmark_roll_deg: float  # cross-check from the outer-canthal line, same sign convention
    # Raw angles relative to the camera optical axis.
    camera_yaw_deg: float | None = None
    camera_pitch_deg: float | None = None
    camera_roll_deg: float | None = None
    translation: list[float] | None = None  # MediaPipe metric camera space (cm)
    line_of_sight_offset_deg: float | None = None  # angle between optical axis and face ray


def matrix_from_column_major(values: list[float]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("facial transformation matrix must have 16 values")
    return np.asarray(values, dtype=float).reshape(4, 4).T


def rotation_from_matrix(m: np.ndarray) -> np.ndarray:
    """Nearest orthonormal rotation to the upper-left 3x3 block (removes any scale)."""
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def euler_from_rotation(r: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees for R = Rz(roll) Ry(yaw) Rx(pitch)."""
    yaw = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    pitch = math.atan2(r[2, 1], r[2, 2])
    roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def line_of_sight_rotation(translation: np.ndarray) -> np.ndarray:
    """Rotation taking the optical axis (+z towards viewer) onto the face->camera direction."""
    if np.linalg.norm(translation) < 1e-9:
        return np.eye(3)
    return rotation_between(np.array([0.0, 0.0, 1.0]), -translation)


def landmark_roll_deg(geom: FaceGeometry) -> float:
    """Roll from the outer-canthal line; positive when the image-right eye is higher."""
    v = geom.image_px[T.LEFT_EYE_OUTER] - geom.image_px[T.RIGHT_EYE_OUTER]
    return -math.degrees(math.atan2(v[1], v[0]))


def estimate_pose(geom: FaceGeometry, matrix: list[float] | None) -> Pose:
    lm_roll = landmark_roll_deg(geom)
    if matrix is None:
        return Pose(
            yaw_deg=None, pitch_deg=None, roll_deg=lm_roll, source="landmarks",
            landmark_roll_deg=lm_roll,
        )  # fmt: skip
    m = matrix_from_column_major(matrix)
    r = rotation_from_matrix(m)
    t = m[:3, 3]
    r_los = line_of_sight_rotation(t)
    cam = euler_from_rotation(r)
    yaw, pitch, roll = euler_from_rotation(r_los.T @ r)
    offset = math.degrees(math.acos(max(-1.0, min(1.0, float(np.trace(r_los) - 1) / 2))))
    return Pose(
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        source="transformation_matrix (line-of-sight corrected)",
        landmark_roll_deg=lm_roll,
        camera_yaw_deg=cam[0],
        camera_pitch_deg=cam[1],
        camera_roll_deg=cam[2],
        translation=[float(x) for x in t],
        line_of_sight_offset_deg=offset,
    )
