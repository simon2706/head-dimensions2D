"""Capture-quality checks.

Each check is a small function returning a ``QualityCheck`` with status
``pass`` / ``warning`` / ``fail`` (or ``skipped`` when its input is not
available). The overall status is the worst individual status; ``skipped``
checks do not affect it. Thresholds come from ``MeasurementConfig`` and are
heuristics, not validated limits.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel

from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.geometry import FaceGeometry
from facial_measurement.measurements.iris import ScaleResult
from facial_measurement.measurements.pose import Pose


class Status(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


_SEVERITY = {Status.SKIPPED: -1, Status.PASS: 0, Status.WARNING: 1, Status.FAIL: 2}


class QualityCheck(BaseModel):
    id: str
    label: str
    status: Status
    value: float | None = None
    unit: str = ""
    message: str = ""
    thresholds: dict[str, float] = {}


class QualityReport(BaseModel):
    overall: Status
    checks: list[QualityCheck]


def worst(statuses: list[Status]) -> Status:
    relevant = [s for s in statuses if s != Status.SKIPPED]
    return max(relevant, key=_SEVERITY.__getitem__) if relevant else Status.PASS


def grade_upper(value: float, warning: float, fail: float) -> Status:
    """Status for a value where larger is worse."""
    if value > fail:
        return Status.FAIL
    if value > warning:
        return Status.WARNING
    return Status.PASS


def grade_lower(value: float, warning: float, fail: float) -> Status:
    """Status for a value where smaller is worse."""
    if value < fail:
        return Status.FAIL
    if value < warning:
        return Status.WARNING
    return Status.PASS


# --- Individual checks ------------------------------------------------------------


def check_face_count(face_count: int) -> QualityCheck:
    ok = face_count == 1
    return QualityCheck(
        id="face_count",
        label="One face detected",
        status=Status.PASS if ok else Status.FAIL,
        value=face_count,
        message="" if ok else f"Expected exactly one face, found {face_count}.",
    )


def check_angle(name: str, value: float | None, warn: float, fail: float) -> QualityCheck:
    if value is None:
        return QualityCheck(
            id=f"head_{name}",
            label=f"Head {name}",
            status=Status.SKIPPED,
            message="Transformation matrix not provided.",
        )
    status = grade_upper(abs(value), warn, fail)
    return QualityCheck(
        id=f"head_{name}",
        label=f"Head {name}",
        status=status,
        value=value,
        unit="°",
        thresholds={"warning": warn, "fail": fail},
        message="" if status == Status.PASS else f"|{name}| {abs(value):.1f}° exceeds {warn:g}°.",
    )


def check_pose(pose: Pose, config: MeasurementConfig) -> list[QualityCheck]:
    return [
        check_angle("roll", pose.roll_deg, config.max_roll_deg, config.fail_roll_deg),
        check_angle("yaw", pose.yaw_deg, config.max_yaw_deg, config.fail_yaw_deg),
        check_angle("pitch", pose.pitch_deg, config.max_pitch_deg, config.fail_pitch_deg),
    ]


def check_iris_agreement(scale: ScaleResult, config: MeasurementConfig) -> QualityCheck:
    if scale.difference_percent is None:
        return QualityCheck(
            id="iris_agreement", label="Iris agreement", status=Status.SKIPPED,
            message="Scale is not iris-based.",
        )  # fmt: skip
    status = grade_upper(
        scale.difference_percent, config.iris_scale_warning_percent, config.iris_scale_fail_percent
    )
    return QualityCheck(
        id="iris_agreement",
        label="Iris agreement",
        status=status,
        value=scale.difference_percent,
        unit="%",
        thresholds={
            "warning": config.iris_scale_warning_percent,
            "fail": config.iris_scale_fail_percent,
        },
        message=""
        if status == Status.PASS
        else "Left and right iris scales disagree (head rotation, gaze or detection error).",
    )


def check_iris_resolution(scale: ScaleResult, config: MeasurementConfig) -> QualityCheck:
    diameters = [d for d in (scale.left_iris_diameter_px, scale.right_iris_diameter_px) if d]
    if not diameters:
        return QualityCheck(
            id="iris_resolution", label="Iris resolution", status=Status.SKIPPED,
            message="Scale is not iris-based.",
        )  # fmt: skip
    smallest = min(diameters)
    status = grade_lower(
        smallest, config.min_iris_diameter_px_warning, config.min_iris_diameter_px_fail
    )
    return QualityCheck(
        id="iris_resolution",
        label="Iris resolution",
        status=status,
        value=smallest,
        unit="px",
        thresholds={
            "warning": config.min_iris_diameter_px_warning,
            "fail": config.min_iris_diameter_px_fail,
        },
        message=""
        if status == Status.PASS
        else "Iris covers few pixels; move closer or use a higher resolution.",
    )


def check_scale_source(scale: ScaleResult) -> QualityCheck:
    if scale.fallback_reason:
        return QualityCheck(
            id="scale_source", label="Scale source", status=Status.WARNING,
            message=f"{scale.fallback_reason} Using {scale.method}.",
        )  # fmt: skip
    return QualityCheck(
        id="scale_source", label="Scale source", status=Status.PASS, message=scale.method
    )


def eye_opening_ratios(geom: FaceGeometry, scale: ScaleResult) -> tuple[float, float]:
    """Lid opening (face-frame vertical) divided by the horizontal iris diameter, (right, left)."""
    a = geom.aligned
    right_open = abs(a[T.RIGHT_EYE_LOWER_LID, 1] - a[T.RIGHT_EYE_UPPER_LID, 1])
    left_open = abs(a[T.LEFT_EYE_LOWER_LID, 1] - a[T.LEFT_EYE_UPPER_LID, 1])
    right_d = scale.right_iris_diameter_px or 1.0
    left_d = scale.left_iris_diameter_px or 1.0
    return float(right_open / right_d), float(left_open / left_d)


def check_eyes_open(
    geom: FaceGeometry, scale: ScaleResult, blendshapes: dict[str, float], config: MeasurementConfig
) -> QualityCheck:
    right_ratio, left_ratio = eye_opening_ratios(geom, scale)
    min_ratio = min(right_ratio, left_ratio)
    statuses = [
        grade_lower(min_ratio, config.eye_opening_ratio_warning, config.eye_opening_ratio_fail)
    ]
    blink = [blendshapes[k] for k in ("eyeBlinkLeft", "eyeBlinkRight") if k in blendshapes]
    max_blink = max(blink) if blink else None
    if max_blink is not None:
        statuses.append(
            grade_upper(
                max_blink, config.eye_blink_blendshape_warning, config.eye_blink_blendshape_fail
            )
        )
    status = worst(statuses)
    blink_txt = f", max blink score {max_blink:.2f}" if max_blink is not None else ""
    return QualityCheck(
        id="eyes_open",
        label="Eyes open",
        status=status,
        value=min_ratio,
        unit="lid/iris",
        thresholds={
            "ratio_warning": config.eye_opening_ratio_warning,
            "ratio_fail": config.eye_opening_ratio_fail,
            "blink_warning": config.eye_blink_blendshape_warning,
            "blink_fail": config.eye_blink_blendshape_fail,
        },
        message=f"Lid opening / iris diameter: R {right_ratio:.2f}, L {left_ratio:.2f}{blink_txt}.",
    )


def check_clipping(
    geom: FaceGeometry, width: int, height: int, config: MeasurementConfig
) -> QualityCheck:
    mx = config.clipping_margin_fraction * width
    my = config.clipping_margin_fraction * height
    oval = geom.image_px[list(T.FACE_OVAL)]
    problems = []
    if geom.image_px[T.FOREHEAD_TOP, 1] < my:
        problems.append("forehead")
    if geom.image_px[T.CHIN, 1] > height - my:
        problems.append("chin")
    # Image-left / image-right borders correspond to the subject's right / left side.
    if float(np.min(oval[:, 0])) < mx:
        problems.append("subject's right side")
    if float(np.max(oval[:, 0])) > width - mx:
        problems.append("subject's left side")
    return QualityCheck(
        id="face_not_clipped",
        label="Face not clipped",
        status=Status.FAIL if problems else Status.PASS,
        message=("Clipped: " + ", ".join(problems)) if problems else "",
        thresholds={"margin_fraction": config.clipping_margin_fraction},
    )


def check_neutral_expression(
    blendshapes: dict[str, float], config: MeasurementConfig
) -> QualityCheck:
    if not blendshapes:
        return QualityCheck(
            id="neutral_expression", label="Neutral expression", status=Status.SKIPPED,
            message="Blendshapes not provided.",
        )  # fmt: skip
    groups = {
        "raised eyebrows": (
            ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
            config.brow_raise_blendshape_warning,
        ),
        "frowning": (("browDownLeft", "browDownRight"), config.brow_down_blendshape_warning),
        "squinting": (("eyeSquintLeft", "eyeSquintRight"), config.eye_squint_blendshape_warning),
    }
    flagged = []
    max_score = 0.0
    for label, (keys, threshold) in groups.items():
        score = max((blendshapes.get(k, 0.0) for k in keys), default=0.0)
        max_score = max(max_score, score)
        if score > threshold:
            flagged.append(f"{label} ({score:.2f})")
    return QualityCheck(
        id="neutral_expression",
        label="Neutral expression",
        status=Status.WARNING if flagged else Status.PASS,
        value=max_score,
        unit="score",
        thresholds={
            "brow_raise": config.brow_raise_blendshape_warning,
            "brow_down": config.brow_down_blendshape_warning,
            "squint": config.eye_squint_blendshape_warning,
        },
        message=("Possible " + ", ".join(flagged) + "; brow measurements may be biased.")
        if flagged
        else "",
    )


def check_sharpness(sharpness: float | None, config: MeasurementConfig) -> QualityCheck:
    if sharpness is None:
        return QualityCheck(
            id="sharpness", label="Image sharpness", status=Status.SKIPPED,
            message="Sharpness not provided.",
        )  # fmt: skip
    low = sharpness < config.sharpness_warning_threshold
    return QualityCheck(
        id="sharpness",
        label="Image sharpness",
        status=Status.WARNING if low else Status.PASS,
        value=sharpness,
        unit="lap. var",
        thresholds={"warning": config.sharpness_warning_threshold},
        message="Diagnostic only; the threshold depends on camera and resolution."
        + (" Image may be blurred." if low else ""),
    )


def run_quality_checks(
    *,
    geom: FaceGeometry,
    scale: ScaleResult,
    pose: Pose,
    face_count: int,
    image_width: int,
    image_height: int,
    blendshapes: dict[str, float],
    sharpness: float | None,
    config: MeasurementConfig,
) -> QualityReport:
    checks = [
        check_face_count(face_count),
        *check_pose(pose, config),
        check_scale_source(scale),
        check_iris_agreement(scale, config),
        check_iris_resolution(scale, config),
        check_eyes_open(geom, scale, blendshapes, config),
        check_clipping(geom, image_width, image_height, config),
        check_neutral_expression(blendshapes, config),
        check_sharpness(sharpness, config),
    ]
    return QualityReport(overall=worst([c.status for c in checks]), checks=checks)
