import pytest

from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.geometry import FaceGeometry
from facial_measurement.measurements.iris import FaceLandmarkerIrisScaleProvider, ScaleResult
from facial_measurement.measurements.quality import (
    Status,
    check_angle,
    check_clipping,
    check_eyes_open,
    check_face_count,
    check_iris_agreement,
    check_iris_resolution,
    check_neutral_expression,
    check_sharpness,
    grade_lower,
    grade_upper,
    worst,
)
from tests.conftest import IMAGE_H, IMAGE_W, synthetic_face_px

CFG = MeasurementConfig()


def scale_with(diff: float | None = 0.0, diameter: float = 40.0) -> ScaleResult:
    return ScaleResult(
        method="iris",
        left_iris_diameter_px=diameter,
        right_iris_diameter_px=diameter,
        final_mm_per_px=0.3,
        difference_percent=diff,
    )


def test_grade_helpers():
    assert grade_upper(4.9, 5, 8) == Status.PASS
    assert grade_upper(5.0, 5, 8) == Status.PASS
    assert grade_upper(5.1, 5, 8) == Status.WARNING
    assert grade_upper(8.1, 5, 8) == Status.FAIL
    assert grade_lower(25, 20, 10) == Status.PASS
    assert grade_lower(15, 20, 10) == Status.WARNING
    assert grade_lower(9, 20, 10) == Status.FAIL


def test_worst_ignores_skipped():
    assert worst([Status.PASS, Status.SKIPPED]) == Status.PASS
    assert worst([Status.PASS, Status.WARNING]) == Status.WARNING
    assert worst([Status.WARNING, Status.FAIL, Status.PASS]) == Status.FAIL
    assert worst([Status.SKIPPED]) == Status.PASS


@pytest.mark.parametrize(
    ("diff", "expected"), [(2.2, Status.PASS), (6.0, Status.WARNING), (9.0, Status.FAIL)]
)
def test_iris_agreement_thresholds(diff, expected):
    assert check_iris_agreement(scale_with(diff), CFG).status == expected


def test_iris_agreement_thresholds_are_configurable():
    cfg = MeasurementConfig(iris_scale_warning_percent=1, iris_scale_fail_percent=2)
    assert check_iris_agreement(scale_with(2.2), cfg).status == Status.FAIL


@pytest.mark.parametrize(
    ("diameter", "expected"), [(30, Status.PASS), (15, Status.WARNING), (8, Status.FAIL)]
)
def test_iris_resolution(diameter, expected):
    assert check_iris_resolution(scale_with(diameter=diameter), CFG).status == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(1.1, Status.PASS), (-2.9, Status.PASS), (-4.0, Status.WARNING), (7.0, Status.FAIL)],
)
def test_roll_thresholds(angle, expected):
    check = check_angle("roll", angle, CFG.max_roll_deg, CFG.fail_roll_deg)
    assert check.status == expected


def test_angle_without_matrix_is_skipped():
    assert check_angle("yaw", None, 5, 10).status == Status.SKIPPED


def test_face_count():
    assert check_face_count(1).status == Status.PASS
    assert check_face_count(0).status == Status.FAIL
    assert check_face_count(2).status == Status.FAIL


def _geom(**kw):
    return FaceGeometry.from_pixels(synthetic_face_px(**kw))


def test_eyes_open_and_closed():
    geom = _geom()
    scale = FaceLandmarkerIrisScaleProvider().estimate(geom, CFG)
    assert check_eyes_open(geom, scale, {}, CFG).status == Status.PASS
    closed = _geom(lid_opening=3.0)
    assert check_eyes_open(closed, scale, {}, CFG).status == Status.FAIL
    blinking = {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1}
    assert check_eyes_open(geom, scale, blinking, CFG).status == Status.FAIL


def test_clipping():
    geom = _geom()
    assert check_clipping(geom, IMAGE_W, IMAGE_H, CFG).status == Status.PASS
    # Image only 640 px wide: the subject's left side (image right, x up to 650) is clipped.
    check = check_clipping(geom, 640, IMAGE_H, CFG)
    assert check.status == Status.FAIL
    assert "subject's left side" in check.message
    assert check_clipping(geom, IMAGE_W, 600, CFG).status == Status.FAIL  # chin


def test_neutral_expression():
    assert check_neutral_expression({"browInnerUp": 0.1}, CFG).status == Status.PASS
    raised = check_neutral_expression({"browOuterUpLeft": 0.7}, CFG)
    assert raised.status == Status.WARNING
    assert "raised eyebrows" in raised.message
    assert check_neutral_expression({}, CFG).status == Status.SKIPPED


def test_sharpness_is_diagnostic_only():
    assert check_sharpness(None, CFG).status == Status.SKIPPED
    assert check_sharpness(5.0, CFG).status == Status.WARNING
    assert check_sharpness(500.0, CFG).status == Status.PASS
