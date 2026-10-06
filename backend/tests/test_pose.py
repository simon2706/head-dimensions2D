import math

import numpy as np
import pytest

from facial_measurement.measurements.geometry import FaceGeometry
from facial_measurement.measurements.pose import (
    estimate_pose,
    euler_from_rotation,
    line_of_sight_rotation,
    matrix_from_column_major,
    rotation_from_matrix,
)
from tests.conftest import IDENTITY_MATRIX, synthetic_face_px


def rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = map(math.radians, (yaw, pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_column_major_layout_puts_translation_in_last_column():
    m = matrix_from_column_major(IDENTITY_MATRIX)
    assert m[2, 3] == -50.0
    np.testing.assert_allclose(m[:3, :3], np.eye(3))


@pytest.mark.parametrize("angles", [(0, 0, 0), (4.0, -3.0, 1.5), (-12.0, 8.0, -5.0)])
def test_euler_round_trip(angles):
    assert euler_from_rotation(rot(*angles)) == pytest.approx(angles, abs=1e-9)


def test_scale_is_removed_from_rotation():
    m = np.eye(4)
    m[:3, :3] = 1.7 * rot(3, 2, 1)
    assert euler_from_rotation(rotation_from_matrix(m)) == pytest.approx((3, 2, 1), abs=1e-9)


def test_estimate_pose_from_matrix_and_landmark_roll():
    m = np.eye(4)
    m[:3, :3] = rot(2.0, -1.0, 0.5)
    pose = estimate_pose(FaceGeometry.from_pixels(synthetic_face_px()), list(m.T.flatten()))
    assert pose.source.startswith("transformation_matrix")
    assert (pose.yaw_deg, pose.pitch_deg, pose.roll_deg) == pytest.approx((2.0, -1.0, 0.5))
    assert pose.landmark_roll_deg == pytest.approx(0.0)


def test_landmark_roll_sign_matches_matrix_convention():
    # Image rotation by +3° (x towards y) lowers the image-right eye -> negative roll.
    pose = estimate_pose(FaceGeometry.from_pixels(synthetic_face_px(roll_deg=3.0)), None)
    assert pose.source == "landmarks"
    assert pose.landmark_roll_deg == pytest.approx(-3.0)
    assert pose.roll_deg == pytest.approx(-3.0)


def test_face_off_axis_but_facing_camera_is_frontal():
    # Face above the optical axis (+y) looking straight at the camera: its forward axis
    # points along the face->camera ray, i.e. it is pitched relative to the camera axis.
    t = np.array([0.0, 10.0, -50.0])
    r = line_of_sight_rotation(t)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    pose = estimate_pose(FaceGeometry.from_pixels(synthetic_face_px()), list(m.T.flatten()))
    assert abs(pose.camera_pitch_deg) == pytest.approx(math.degrees(math.atan(10 / 50)))
    assert (pose.yaw_deg, pose.pitch_deg, pose.roll_deg) == pytest.approx((0, 0, 0), abs=1e-9)
    assert pose.line_of_sight_offset_deg == pytest.approx(math.degrees(math.atan(10 / 50)))


def test_line_of_sight_identity_on_axis():
    np.testing.assert_allclose(line_of_sight_rotation(np.array([0, 0, -50.0])), np.eye(3))
