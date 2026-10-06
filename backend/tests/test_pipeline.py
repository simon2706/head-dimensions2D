import json
import math

import pytest
from fastapi.testclient import TestClient

from facial_measurement.main import app
from facial_measurement.models import MeasureRequest, MeasureResponse
from facial_measurement.pipeline import FEATURE_IDS, measure_frame
from tests.conftest import (
    iris_model_payload,
    make_request,
    synthetic_face_px,
    synthetic_iris_model_px,
)

client = TestClient(app)


def run(**kw) -> MeasureResponse:
    return measure_frame(MeasureRequest.model_validate(make_request(**kw)))


def by_id(resp: MeasureResponse) -> dict:
    return {m.id: m for m in resp.measurements}


def test_direct_measurements_on_synthetic_face():
    resp = run()
    m = by_id(resp)
    assert resp.scale.final_mm_per_px == pytest.approx(0.585)
    assert m["outer_canthal_width"].value_px == pytest.approx(140.0)
    assert m["outer_canthal_width"].value_mm == pytest.approx(140 * 0.585)
    assert m["outer_canthal_width"].landmarks == [33, 263]
    assert m["inner_canthal_width"].value_px == pytest.approx(40.0)
    assert m["iris_center_distance"].value_px == pytest.approx(90.0)
    assert m["outer_brow_span"].value_px == pytest.approx(190.0)
    assert m["outer_brow_span"].details["selected_right"] == 70
    assert m["outer_brow_span"].details["selected_left"] == 300
    assert m["upper_temporal_mesh_width"].landmarks == [162, 389]


def test_contour_widths_are_symmetric_and_ordered():
    m = by_id(run())
    brow = m["upper_face_width_at_brow_level"]
    eye = m["face_width_at_eye_level"]
    mid = m["midface_width_proxy"]
    mx = m["maximum_upper_face_width"]
    for meas in (brow, eye, mid, mx):
        (x0, y0), (x1, y1) = meas.endpoints_px
        assert y0 == pytest.approx(y1)
        assert (x0 + x1) / 2 == pytest.approx(500.0)
        assert meas.value_px == pytest.approx(x1 - x0)
    # Ellipse widens towards y=500 (its centre): brow < eye < midface.
    assert brow.value_px < eye.value_px < mid.value_px
    assert mx.value_px >= mid.value_px - 1e-9
    assert mid.level_y_face_px == pytest.approx(460 + 0.4 * (560 - 460))


def test_widths_are_invariant_to_small_roll():
    flat = by_id(run())
    rolled = by_id(run(px=synthetic_face_px(roll_deg=2.5)))
    for mid in FEATURE_IDS:
        assert rolled[mid].value_px == pytest.approx(flat[mid].value_px, rel=1e-9), mid


def test_good_capture_passes():
    resp = run()
    assert resp.quality.overall == "pass"
    assert all(m.quality == "good" for m in resp.measurements)


def test_failed_check_flags_measurements_as_poor():
    resp = run(face_count=2)
    assert resp.quality.overall == "fail"
    assert all(m.quality == "poor" for m in resp.measurements)
    assert all(m.value_mm is not None for m in resp.measurements)  # flagged, not hidden


def test_lateral_measurements_warn_on_moderate_yaw():
    y = math.radians(4.0)  # passes global yaw (5°) but exceeds the lateral limit (2.5°)
    matrix = [
        math.cos(y),
        0,
        -math.sin(y),
        0,
        0,
        1,
        0,
        0,
        math.sin(y),
        0,
        math.cos(y),
        0,
        0,
        0,
        -50,
        1,
    ]
    m = by_id(run(facial_transformation_matrix=matrix))
    assert m["outer_canthal_width"].quality == "good"
    assert m["face_width_at_eye_level"].quality == "warning"
    assert m["upper_temporal_mesh_width"].quality == "warning"


def test_features_vector():
    resp = run()
    features = resp.features.model_dump()
    assert features["feature_schema_version"] == "1.1"
    assert features["scale_method"] == "iris_model"
    for mid in FEATURE_IDS:
        assert features[f"{mid}_mm"] == pytest.approx(by_id(resp)[mid].value_mm)


def test_measurement_serialization_round_trip():
    resp = run()
    data = json.loads(resp.model_dump_json())
    meas = data["measurements"][0]
    assert set(meas) >= {"id", "name", "value_px", "value_mm", "landmarks", "quality", "type"}
    assert meas["type"] == "direct"
    assert MeasureResponse.model_validate(data) == resp


def test_config_override_changes_scale():
    resp = run(config={"assumed_iris_diameter_mm": 12.0})
    assert resp.scale.final_mm_per_px == pytest.approx(0.6)
    assert resp.config.max_yaw_deg == 5.0  # other defaults preserved


def test_api_endpoints():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/config").json()["assumed_iris_diameter_mm"] == 11.7
    r = client.post("/api/measure", json=make_request())
    assert r.status_code == 200
    assert r.json()["features"]["outer_canthal_width_mm"] == pytest.approx(81.9)


def test_api_rejects_wrong_landmark_count():
    payload = make_request()
    payload["landmarks"] = payload["landmarks"][:468]
    assert client.post("/api/measure", json=payload).status_code == 422


def test_api_rejects_unknown_config_keys():
    assert client.post("/api/measure", json=make_request(config={"bogus": 1})).status_code == 422


def test_iris_model_is_the_default_scale_source():
    resp = run(iris_model=iris_model_payload(*synthetic_iris_model_px(right_radius=11.0)))
    assert resp.scale.method == "iris_model"
    assert resp.scale.fallback_reason is None
    assert resp.scale.right_iris_diameter_px == pytest.approx(22.0)
    assert resp.scale.left_iris_diameter_px == pytest.approx(20.0)
    checks = {c.id: c for c in resp.quality.checks}
    assert checks["scale_source"].status == "pass"


def test_scale_source_face_landmarker_ignores_iris_model():
    resp = run(
        iris_model=iris_model_payload(*synthetic_iris_model_px(right_radius=11.0)),
        config={"scale_source": "face_landmarker"},
    )
    assert resp.scale.method == "face_landmarker_iris"
    assert resp.scale.right_iris_diameter_px == pytest.approx(20.0)
    assert resp.features.scale_method == "face_landmarker_iris"


def test_missing_iris_model_falls_back_with_warning():
    resp = run(iris_model=None)
    assert resp.scale.method == "face_landmarker_iris"
    assert resp.scale.fallback_reason
    checks = {c.id: c for c in resp.quality.checks}
    assert checks["scale_source"].status == "warning"
    assert resp.quality.overall == "warning"
    # Not a core check: measurements are not downgraded by the fallback alone.
    assert all(m.quality == "good" for m in resp.measurements)


def test_api_rejects_wrong_iris_point_count():
    payload = make_request()
    payload["iris_model"]["left"]["iris"] = payload["iris_model"]["left"]["iris"][:4]
    assert client.post("/api/measure", json=payload).status_code == 422


def test_api_rejects_unknown_scale_source():
    payload = make_request(config={"scale_source": "bogus"})
    assert client.post("/api/measure", json=payload).status_code == 422
