"""Pixel-space geometry primitives.

Coordinates are image pixels with x to the right and y down. The *face frame*
is the image frame rotated about the outer-canthal midpoint so that the line
between the outer eye corners is horizontal. Horizontal levels (brow level,
eye level, ...) are defined in that frame, which removes small in-plane head
roll from width measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from facial_measurement import topology as T


def normalized_to_pixels(landmarks: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert normalized (x, y[, z]) landmarks to (N, 2) pixel coordinates.

    MediaPipe normalizes x by image width and y by image height. z is ignored on
    purpose: Face Landmarker z is not metric depth.
    """
    pts = np.asarray(landmarks, dtype=float)
    return np.column_stack((pts[:, 0] * image_width, pts[:, 1] * image_height))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(math.hypot(float(b[0] - a[0]), float(b[1] - a[1])))


def rotate(points: np.ndarray, angle_rad: float, origin: np.ndarray) -> np.ndarray:
    """Rotate points (N, 2) or (2,) about origin; a positive angle turns +x towards +y."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    pts = np.asarray(points, dtype=float)
    return (pts - origin) @ rot.T + origin


@dataclass(frozen=True)
class FaceFrame:
    """Roll-corrected frame anchored at the outer-canthal midpoint."""

    origin: np.ndarray
    roll_rad: float  # angle of the 33 -> 263 line in image coordinates

    @classmethod
    def from_eye_corners(cls, right_outer: np.ndarray, left_outer: np.ndarray) -> FaceFrame:
        v = np.asarray(left_outer, dtype=float) - np.asarray(right_outer, dtype=float)
        origin = (np.asarray(left_outer, dtype=float) + np.asarray(right_outer, dtype=float)) / 2
        return cls(origin=origin, roll_rad=math.atan2(v[1], v[0]))

    def to_face(self, points: np.ndarray) -> np.ndarray:
        return rotate(points, -self.roll_rad, self.origin)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        return rotate(points, self.roll_rad, self.origin)


@dataclass(frozen=True)
class FaceGeometry:
    """Landmarks of one face in image pixels and in the roll-corrected face frame."""

    image_px: np.ndarray  # (478, 2)
    frame: FaceFrame
    aligned: np.ndarray  # (478, 2)

    @classmethod
    def from_pixels(cls, image_px: np.ndarray) -> FaceGeometry:
        image_px = np.asarray(image_px, dtype=float)
        frame = FaceFrame.from_eye_corners(image_px[T.RIGHT_EYE_OUTER], image_px[T.LEFT_EYE_OUTER])
        return cls(image_px=image_px, frame=frame, aligned=frame.to_face(image_px))

    def to_image(self, aligned_points: np.ndarray) -> np.ndarray:
        return self.frame.to_image(aligned_points)

    def mean_aligned_y(self, ids: tuple[int, ...] | list[int]) -> float:
        return float(np.mean(self.aligned[list(ids), 1]))
