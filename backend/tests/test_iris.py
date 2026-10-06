import pytest

from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.geometry import FaceGeometry
from facial_measurement.measurements.iris import (
    FaceLandmarkerIrisScaleProvider,
    IrisModelScaleProvider,
    ReferenceObjectScaleProvider,
    mm_per_px,
    px_to_mm,
    scale_difference_percent,
)
from tests.conftest import synthetic_face_px, synthetic_iris_model_px


def test_mm_per_px():
    assert mm_per_px(11.7, 42.0) == pytest.approx(0.278571, rel=1e-5)
    with pytest.raises(ValueError):
        mm_per_px(11.7, 0)


def test_scale_difference_percent_example_from_spec():
    assert scale_difference_percent(0.281, 0.287) == pytest.approx(2.11, abs=0.01)
    assert scale_difference_percent(0.3, 0.3) == 0.0


def test_iris_scale_symmetric_face():
    scale = FaceLandmarkerIrisScaleProvider().estimate(
        FaceGeometry.from_pixels(synthetic_face_px()), MeasurementConfig()
    )
    assert scale.left_iris_diameter_px == pytest.approx(20.0)
    assert scale.right_iris_diameter_px == pytest.approx(20.0)
    assert scale.left_mm_per_px == pytest.approx(11.7 / 20)
    assert scale.final_mm_per_px == pytest.approx(0.585)
    assert scale.difference_percent == pytest.approx(0.0)


def test_iris_scale_left_right_mismatch():
    geom = FaceGeometry.from_pixels(synthetic_face_px(left_iris_radius=10.5))
    scale = FaceLandmarkerIrisScaleProvider().estimate(
        geom, MeasurementConfig(assumed_iris_diameter_mm=12.0)
    )
    assert scale.left_iris_diameter_px == pytest.approx(21.0)
    assert scale.left_mm_per_px == pytest.approx(12 / 21)
    assert scale.right_mm_per_px == pytest.approx(12 / 20)
    expected = abs(12 / 21 - 12 / 20) / ((12 / 21 + 12 / 20) / 2) * 100
    assert scale.difference_percent == pytest.approx(expected)
    assert scale.final_mm_per_px == pytest.approx((12 / 21 + 12 / 20) / 2)


def test_face_landmarker_scale_reports_endpoints():
    scale = FaceLandmarkerIrisScaleProvider().estimate(
        FaceGeometry.from_pixels(synthetic_face_px()), MeasurementConfig()
    )
    assert scale.method == "face_landmarker_iris"
    assert scale.endpoints_px["right_horizontal"] == [[465.0, 460.0], [445.0, 460.0]]
    assert scale.iris_centers_px["left"] == [545.0, 460.0]


@pytest.mark.parametrize("roll_deg", [0.0, 20.0, -20.0])
@pytest.mark.parametrize("swap_pairs", [False, True])
def test_iris_model_scale_picks_horizontal_pair(roll_deg, swap_pairs):
    geom = FaceGeometry.from_pixels(synthetic_face_px(roll_deg=roll_deg))
    right, left = synthetic_iris_model_px(
        roll_deg=roll_deg, right_radius=11.0, left_radius=12.0, swap_pairs=swap_pairs
    )
    scale = IrisModelScaleProvider(right, left).estimate(geom, MeasurementConfig())
    assert scale.method == "iris_model"
    assert scale.right_iris_diameter_px == pytest.approx(22.0)
    assert scale.left_iris_diameter_px == pytest.approx(24.0)
    assert scale.right_iris_vertical_diameter_px == pytest.approx(22.0 * 0.9)
    assert scale.right_mm_per_px == pytest.approx(11.7 / 22)
    assert scale.left_mm_per_px == pytest.approx(11.7 / 24)
    assert scale.final_mm_per_px == pytest.approx((11.7 / 22 + 11.7 / 24) / 2)
    assert scale.landmarks == {}
    assert scale.iris_centers_px["right"] == pytest.approx(list(right[0]))


def test_iris_model_provider_rejects_wrong_shape():
    right, left = synthetic_iris_model_px()
    with pytest.raises(ValueError):
        IrisModelScaleProvider(right[:4], left)


def test_px_to_mm():
    assert px_to_mm(100.0, 0.25) == pytest.approx(25.0)
    assert px_to_mm(None, 0.25) is None


def test_reference_object_provider_is_a_stub():
    with pytest.raises(NotImplementedError):
        ReferenceObjectScaleProvider(85.6, 300).estimate(
            FaceGeometry.from_pixels(synthetic_face_px()), MeasurementConfig()
        )
