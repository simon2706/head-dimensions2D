"""Central measurement configuration.

Every threshold used by the measurement engine lives here. None of the default
values are scientifically validated; they are initial engineering heuristics
intended to be tuned against real data.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ScaleSource = Literal["iris_model", "face_landmarker"]


class MeasurementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Metric scale -------------------------------------------------------
    assumed_iris_diameter_mm: float = Field(
        11.7, gt=0, description="Assumed horizontal visible iris diameter (HVID) in mm."
    )
    scale_source: ScaleSource = Field(
        "iris_model",
        description=(
            "Iris landmarks used for the scale: 'iris_model' = dedicated MediaPipe Iris "
            "landmark model run on 64x64 eye crops (falls back to 'face_landmarker' with a "
            "warning when its output is missing); 'face_landmarker' = iris points 468-477 "
            "of the Face Landmarker mesh."
        ),
    )

    # --- Head pose (degrees). |angle| <= max -> pass, <= fail -> warning, else fail.
    max_yaw_deg: float = Field(5.0, gt=0)
    max_pitch_deg: float = Field(5.0, gt=0)
    max_roll_deg: float = Field(3.0, gt=0)
    fail_yaw_deg: float = Field(10.0, gt=0)
    fail_pitch_deg: float = Field(10.0, gt=0)
    fail_roll_deg: float = Field(6.0, gt=0)
    lateral_measurement_max_yaw_deg: float = Field(
        2.5,
        gt=0,
        description=(
            "Measurements reaching the lateral face contour are flagged as 'warning' "
            "above this |yaw| even when the global yaw check passes."
        ),
    )

    # --- Iris agreement / resolution ------------------------------------------
    iris_scale_warning_percent: float = Field(5.0, gt=0)
    iris_scale_fail_percent: float = Field(8.0, gt=0)
    min_iris_diameter_px_warning: float = Field(
        20.0, gt=0, description="Warn when the smaller iris diameter is below this many px."
    )
    min_iris_diameter_px_fail: float = Field(
        10.0, gt=0, description="Fail when the smaller iris diameter is below this many px."
    )

    # --- Eyes open ---------------------------------------------------------------
    # Eye opening ratio = upper/lower lid distance / horizontal iris diameter.
    eye_opening_ratio_warning: float = Field(0.55, gt=0)
    eye_opening_ratio_fail: float = Field(0.35, gt=0)
    eye_blink_blendshape_warning: float = Field(0.4, ge=0, le=1)
    eye_blink_blendshape_fail: float = Field(0.6, ge=0, le=1)

    # --- Neutral expression (blendshape scores, warning only) --------------------
    brow_raise_blendshape_warning: float = Field(0.5, ge=0, le=1)
    brow_down_blendshape_warning: float = Field(0.5, ge=0, le=1)
    eye_squint_blendshape_warning: float = Field(0.6, ge=0, le=1)

    # --- Clipping ----------------------------------------------------------------
    clipping_margin_fraction: float = Field(
        0.01,
        ge=0,
        lt=0.5,
        description="Face contour must stay this fraction of the image size inside the border.",
    )

    # --- Sharpness (diagnostic only) ---------------------------------------------
    sharpness_warning_threshold: float = Field(
        30.0,
        ge=0,
        description=(
            "Variance of the Laplacian on the face crop resampled to a fixed width "
            "(computed in the browser). Below this value -> warning."
        ),
    )

    # --- Measurement geometry ----------------------------------------------------
    midface_level_fraction: float = Field(
        0.4,
        ge=0,
        le=1,
        description=(
            "Midface level = eye level + fraction * (subnasale level - eye level), "
            "in the roll-corrected face frame."
        ),
    )
    max_width_samples: int = Field(
        41,
        ge=2,
        description="Uniform sample count for the maximum-upper-face-width search "
        "(contour vertex levels inside the range are always evaluated as well).",
    )


DEFAULT_CONFIG = MeasurementConfig()
