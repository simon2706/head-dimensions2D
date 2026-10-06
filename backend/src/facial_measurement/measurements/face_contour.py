"""Face-oval contour intersection and contour-based width measurements.

The MediaPipe face oval is treated as two piecewise-linear chains (subject's
right and left side) between the forehead-top and chin landmarks. Widths are
measured between the intersections of a horizontal line (in the roll-corrected
face frame) with each chain, using linear interpolation along contour segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facial_measurement import topology as T
from facial_measurement.config import MeasurementConfig
from facial_measurement.measurements.base import MeasurementType, PixelMeasurement, points_to_list
from facial_measurement.measurements.distances import lateral_brow_points
from facial_measurement.measurements.geometry import FaceGeometry


@dataclass(frozen=True)
class ContourChain:
    ids: tuple[int, ...]
    points: np.ndarray  # (M, 2), face-frame coordinates, ordered along the contour
    side: int  # -1: subject's right (image left, smaller x); +1: subject's left


@dataclass(frozen=True)
class FaceContour:
    right: ContourChain
    left: ContourChain


@dataclass(frozen=True)
class ChainIntersection:
    point: np.ndarray  # (2,)
    segment: tuple[int, int]  # landmark IDs of the segment that was interpolated
    t: float  # interpolation parameter along the segment


@dataclass(frozen=True)
class ContourWidth:
    y: float
    width: float
    right: ChainIntersection
    left: ChainIntersection


def split_oval_chains(
    oval_ids: tuple[int, ...] = T.FACE_OVAL, top: int = T.FOREHEAD_TOP, bottom: int = T.CHIN
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the closed oval loop into (right_chain, left_chain), both ordered top -> bottom."""
    ids = list(oval_ids)
    i_top, i_bottom = ids.index(top), ids.index(bottom)
    rolled = ids[i_top:] + ids[:i_top]
    i_bottom = rolled.index(bottom)
    first = tuple(rolled[: i_bottom + 1])  # top -> bottom along loop order
    second = tuple([bottom] + rolled[i_bottom + 1 :] + [top])[::-1]  # top -> bottom
    # MediaPipe's loop runs down the subject's left side first.
    return second, first


def build_face_contour(geom: FaceGeometry) -> FaceContour:
    right_ids, left_ids = split_oval_chains()
    return FaceContour(
        right=ContourChain(right_ids, geom.aligned[list(right_ids)], side=-1),
        left=ContourChain(left_ids, geom.aligned[list(left_ids)], side=+1),
    )


def intersect_chain_at_y(chain: ContourChain, y: float) -> ChainIntersection | None:
    """Intersect a horizontal line with a contour chain.

    Every segment crossing the line is interpolated linearly; if the chain
    crosses the line several times, the most lateral crossing is returned.
    """
    best: ChainIntersection | None = None
    pts = chain.points
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if (y0 - y) * (y1 - y) > 0:
            continue
        if y0 == y1:
            candidates = [(np.array([x0, y0]), 0.0), (np.array([x1, y1]), 1.0)]
        else:
            t = (y - y0) / (y1 - y0)
            candidates = [(np.array([x0 + t * (x1 - x0), y]), float(t))]
        for point, t in candidates:
            if best is None or chain.side * point[0] > chain.side * best.point[0]:
                best = ChainIntersection(point, (chain.ids[i], chain.ids[i + 1]), t)
    return best


def width_at_y(face_contour: FaceContour, target_y: float) -> ContourWidth | None:
    """Horizontal face-contour width at ``target_y`` (face frame) with both intersection points."""
    right = intersect_chain_at_y(face_contour.right, target_y)
    left = intersect_chain_at_y(face_contour.left, target_y)
    if right is None or left is None:
        return None
    return ContourWidth(
        y=float(target_y), width=float(left.point[0] - right.point[0]), right=right, left=left
    )


def maximum_face_width_in_range(
    face_contour: FaceContour, y_top: float, y_bottom: float, samples: int
) -> ContourWidth | None:
    """Maximum contour width for y in [y_top, y_bottom].

    The width is piecewise linear in y with breakpoints at contour vertex
    levels, so evaluating the uniform samples *plus* every vertex level inside
    the range finds the exact maximum.
    """
    lo, hi = min(y_top, y_bottom), max(y_top, y_bottom)
    vertex_ys = np.concatenate((face_contour.right.points[:, 1], face_contour.left.points[:, 1]))
    levels = np.unique(
        np.concatenate(
            (np.linspace(lo, hi, samples), vertex_ys[(vertex_ys >= lo) & (vertex_ys <= hi)])
        )
    )
    best: ContourWidth | None = None
    for y in levels:
        w = width_at_y(face_contour, float(y))
        if w is not None and (best is None or w.width > best.width):
            best = w
    return best


# --- Measurement definitions ------------------------------------------------------


def _contour_measurement(
    geom: FaceGeometry,
    w: ContourWidth | None,
    *,
    id: str,
    name: str,
    type: MeasurementType,
    definition: str,
    formula: str,
    level_landmarks: list[int],
    details: dict,
) -> PixelMeasurement:
    m = PixelMeasurement(
        id=id,
        name=name,
        type=type,
        definition=definition,
        formula=formula,
        value_px=None,
        landmarks=list(level_landmarks),
        uses_lateral_contour=True,
        details=details,
    )
    if w is None:
        m.error = "Horizontal level does not intersect both sides of the face contour."
        return m
    m.value_px = w.width
    m.level_y_face_px = w.y
    m.landmarks = list(dict.fromkeys([*level_landmarks, *w.right.segment, *w.left.segment]))
    m.endpoints_px = points_to_list(geom.to_image(np.array([w.right.point, w.left.point])))
    m.details |= {
        "level_y_face_px": w.y,
        "right_contour_segment": list(w.right.segment),
        "right_segment_t": w.right.t,
        "left_contour_segment": list(w.left.segment),
        "left_segment_t": w.left.t,
    }
    return m


def brow_level_y(geom: FaceGeometry) -> tuple[float, list[int]]:
    right_id, left_id = lateral_brow_points(geom)
    return geom.mean_aligned_y([right_id, left_id]), [right_id, left_id]


def eye_level_y(geom: FaceGeometry) -> float:
    return geom.mean_aligned_y([T.RIGHT_EYE_OUTER, T.LEFT_EYE_OUTER])


def midface_level_y(geom: FaceGeometry, config: MeasurementConfig) -> float:
    eye_y = eye_level_y(geom)
    sub_y = float(geom.aligned[T.SUBNASALE, 1])
    return eye_y + config.midface_level_fraction * (sub_y - eye_y)


def upper_face_width_at_brow_level(
    geom: FaceGeometry, contour: FaceContour, config: MeasurementConfig
) -> PixelMeasurement:
    y, brow_ids = brow_level_y(geom)
    m = _contour_measurement(
        geom,
        width_at_y(contour, y),
        id="upper_face_width_at_brow_level",
        name="Upper face width @ brow level",
        type=MeasurementType.SURFACE_PROXY,
        definition=(
            "Width of the MediaPipe face-oval contour along a horizontal line (roll-corrected "
            "face frame) at the mean height of the two lateral-most eyebrow landmarks."
        ),
        formula="brow_y = mean(y[R brow tail], y[L brow tail]); width_at_y(face_oval, brow_y)",
        level_landmarks=brow_ids,
        details={"brow_landmarks": brow_ids},
    )
    m.uses_brows = True
    return m


def face_width_at_eye_level(
    geom: FaceGeometry, contour: FaceContour, config: MeasurementConfig
) -> PixelMeasurement:
    ids = [T.RIGHT_EYE_OUTER, T.LEFT_EYE_OUTER]
    return _contour_measurement(
        geom,
        width_at_y(contour, eye_level_y(geom)),
        id="face_width_at_eye_level",
        name="Face width @ eye level",
        type=MeasurementType.SURFACE_PROXY,
        definition=(
            "Width of the face-oval contour along a horizontal line at the mean height of "
            "the outer eye corners."
        ),
        formula="eye_y = mean(y[33], y[263]); width_at_y(face_oval, eye_y)",
        level_landmarks=ids,
        details={},
    )


def midface_width_proxy(
    geom: FaceGeometry, contour: FaceContour, config: MeasurementConfig
) -> PixelMeasurement:
    f = config.midface_level_fraction
    return _contour_measurement(
        geom,
        width_at_y(contour, midface_level_y(geom, config)),
        id="midface_width_proxy",
        name="Midface width proxy",
        type=MeasurementType.EXPERIMENTAL_PROXY,
        definition=(
            f"Face-oval width at the level {f:g} of the way from eye level (mean of 33, 263) "
            "down to the subnasal level (landmark 2). Reproducible cheek-region width; NOT "
            "bizygomatic breadth."
        ),
        formula=f"y = eye_y + {f:g} * (y[2] - eye_y); width_at_y(face_oval, y)",
        level_landmarks=[T.RIGHT_EYE_OUTER, T.LEFT_EYE_OUTER, T.SUBNASALE],
        details={"midface_level_fraction": f},
    )


def maximum_upper_face_width(
    geom: FaceGeometry, contour: FaceContour, config: MeasurementConfig
) -> PixelMeasurement:
    top, brow_ids = brow_level_y(geom)
    bottom = midface_level_y(geom, config)
    w = maximum_face_width_in_range(contour, top, bottom, config.max_width_samples)
    m = _contour_measurement(
        geom,
        w,
        id="maximum_upper_face_width",
        name="Maximum upper-face width",
        type=MeasurementType.SURFACE_PROXY,
        definition=(
            "Maximum horizontal face-oval width over the band from brow level down to the "
            "midface-proxy level. Visible silhouette width, not a skeletal measurement."
        ),
        formula="max over y in [brow_y, midface_y] of width_at_y(face_oval, y)",
        level_landmarks=[*brow_ids, T.RIGHT_EYE_OUTER, T.LEFT_EYE_OUTER, T.SUBNASALE],
        details={"search_top_y_face_px": top, "search_bottom_y_face_px": bottom},
    )
    m.uses_brows = True
    return m
