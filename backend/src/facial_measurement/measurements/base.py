"""Pixel-space measurement result shared by all measurement functions.

Measurement functions only produce pixel values. Conversion to millimetres
happens later in one place (``iris.px_to_mm``) so that the scale source can be
swapped without touching the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class MeasurementType(StrEnum):
    DIRECT = "direct"
    SURFACE_PROXY = "surface_proxy"
    EXPERIMENTAL_PROXY = "experimental_proxy"


@dataclass
class PixelMeasurement:
    id: str
    name: str
    type: MeasurementType
    definition: str
    formula: str
    value_px: float | None
    landmarks: list[int]
    endpoints_px: list[list[float]] = field(default_factory=list)  # image pixel coordinates
    level_y_face_px: float | None = None  # horizontal level in the roll-corrected face frame
    uses_lateral_contour: bool = False
    uses_brows: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def points_to_list(points: np.ndarray) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in np.atleast_2d(points)]
