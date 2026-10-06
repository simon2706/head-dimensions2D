import math

import numpy as np
import pytest

from facial_measurement import topology as T
from facial_measurement.measurements.geometry import (
    FaceFrame,
    FaceGeometry,
    distance,
    normalized_to_pixels,
    rotate,
)
from tests.conftest import synthetic_face_px


def test_euclidean_distance():
    assert distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert distance(np.array([10.0, 10.0]), np.array([10.0, 10.0])) == 0.0


def test_normalized_to_pixels_ignores_z():
    lm = np.array([[0.5, 0.25, -0.3], [1.0, 1.0, 0.9], [0.0, 0.0, 0.0]])
    px = normalized_to_pixels(lm, 1920, 1080)
    assert px.shape == (3, 2)
    np.testing.assert_allclose(px, [[960, 270], [1920, 1080], [0, 0]])


def test_rotate_quarter_turn():
    out = rotate(np.array([1.0, 0.0]), math.pi / 2, np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_face_frame_round_trip_and_levels_eyes():
    frame = FaceFrame.from_eye_corners(np.array([100.0, 100.0]), np.array([200.0, 120.0]))
    pts = np.array([[100.0, 100.0], [200.0, 120.0], [150.0, 300.0]])
    aligned = frame.to_face(pts)
    assert aligned[0, 1] == pytest.approx(aligned[1, 1])
    np.testing.assert_allclose(frame.to_image(aligned), pts)


def test_face_geometry_removes_roll():
    geom = FaceGeometry.from_pixels(synthetic_face_px(roll_deg=4.0))
    a = geom.aligned
    assert a[T.RIGHT_EYE_OUTER, 1] == pytest.approx(a[T.LEFT_EYE_OUTER, 1])
    assert math.degrees(geom.frame.roll_rad) == pytest.approx(4.0)
