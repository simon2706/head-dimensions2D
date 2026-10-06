import numpy as np
import pytest

from facial_measurement import topology as T
from facial_measurement.measurements.face_contour import (
    ContourChain,
    FaceContour,
    intersect_chain_at_y,
    maximum_face_width_in_range,
    split_oval_chains,
    width_at_y,
)


def diamond() -> FaceContour:
    """Right chain (0,-10)->(-5,0)->(0,10); left chain mirrored. Width at y is 10 - |y|."""
    right = ContourChain((1, 2, 3), np.array([[0.0, -10.0], [-5.0, 0.0], [0.0, 10.0]]), side=-1)
    left = ContourChain((1, 4, 3), np.array([[0.0, -10.0], [5.0, 0.0], [0.0, 10.0]]), side=+1)
    return FaceContour(right=right, left=left)


def test_split_oval_chains():
    right, left = split_oval_chains()
    assert right[0] == left[0] == T.FOREHEAD_TOP
    assert right[-1] == left[-1] == T.CHIN
    assert 389 in left and 454 in left  # subject's left
    assert 162 in right and 234 in right  # subject's right
    assert len(right) + len(left) == len(T.FACE_OVAL) + 2


def test_contour_intersection_interpolates_segment():
    hit = intersect_chain_at_y(diamond().right, -5.0)
    assert hit is not None
    np.testing.assert_allclose(hit.point, [-2.5, -5.0])
    assert hit.segment == (1, 2)
    assert hit.t == pytest.approx(0.5)


def test_contour_intersection_out_of_range():
    assert intersect_chain_at_y(diamond().left, 11.0) is None


def test_contour_intersection_picks_most_lateral_crossing():
    # A chain that crosses y=0 three times; the most lateral (smallest x for side=-1) wins.
    chain = ContourChain(
        (1, 2, 3, 4), np.array([[0.0, -5.0], [-8.0, 5.0], [-2.0, -5.0], [-3.0, 5.0]]), side=-1
    )
    hit = intersect_chain_at_y(chain, 0.0)
    assert hit is not None
    assert hit.point[0] == pytest.approx(-5.0)


@pytest.mark.parametrize(("y", "expected"), [(0.0, 10.0), (5.0, 5.0), (-7.5, 2.5), (10.0, 0.0)])
def test_width_at_y(y, expected):
    w = width_at_y(diamond(), y)
    assert w is not None
    assert w.width == pytest.approx(expected)
    assert w.right.point[1] == pytest.approx(y)
    assert w.left.point[0] - w.right.point[0] == pytest.approx(expected)


def test_width_at_y_none_outside_contour():
    assert width_at_y(diamond(), -20.0) is None


def test_maximum_width_in_range_finds_vertex_level_exactly():
    # Only 2 uniform samples (the range ends, width 6 and 4) - the vertex level y=0 must still win.
    w = maximum_face_width_in_range(diamond(), -4.0, 6.0, samples=2)
    assert w is not None
    assert w.width == pytest.approx(10.0)
    assert w.y == pytest.approx(0.0)


def test_maximum_width_in_range_without_vertices():
    w = maximum_face_width_in_range(diamond(), 2.0, 8.0, samples=5)
    assert w is not None
    assert w.width == pytest.approx(8.0)
    assert w.y == pytest.approx(2.0)
