"""Geometry primitives — pure NumPy, no heavy deps."""

import numpy as np

from src.common import centroids, iou_matrix, points_in_polygon, segments_intersect


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    assert abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-6
    b = np.array([[20, 20, 30, 30]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_partial():
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[1, 0, 3, 2]])          # inter=2, union=6 -> 1/3
    assert abs(iou_matrix(a, b)[0, 0] - (1 / 3)) < 1e-6


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_point_in_polygon():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    pts = np.array([[5, 5], [15, 5], [-1, -1]])
    assert points_in_polygon(pts, poly).tolist() == [True, False, False]


def test_segments_intersect():
    assert segments_intersect([0, 0], [2, 2], [0, 2], [2, 0]) is True
    assert segments_intersect([0, 0], [1, 0], [0, 1], [1, 1]) is False


def test_centroids():
    assert centroids(np.array([[0, 0, 10, 20]])).tolist() == [[5.0, 10.0]]
