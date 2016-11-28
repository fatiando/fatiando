from __future__ import division, absolute_import, print_function
import numpy.testing as npt
import numpy as np

from ... import gridder


def test_inside():
    "gridder.inside returns correct results for simple input"
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([10, 11, 12, 13, 14, 15])
    area = [2.5, 5.5, 12, 15]
    is_inside = gridder.inside(x, y, area)
    assert np.all(is_inside == [False, False, True, True, True, False])
    # Test 2D arrays as well
    x = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
    y = np.array([[5, 7, 9],
                  [5, 7, 9],
                  [5, 7, 9]])
    area = [0.5, 2.5, 7, 9]
    is_inside = gridder.inside(x, y, area)
    truth = [[False, True, True], [False, True, True], [False, False, False]]
    assert np.all(is_inside == truth)
    # Test some large arrays of random points inside an area
    area = (-1035, 255, 0.2345, 23355)
    x, y = gridder.scatter((-1035, 255, 0.2345, 23355), n=10000, seed=0)
    assert np.all(gridder.inside(x, y, area))


def test_cut():
    "Cutting a grid works for simple grids"
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([10, 11, 12, 13, 14, 15])
    data = np.array([42, 65, 92, 24, 135, 456])
    area = [2.5, 5.5, 12, 15]
    xs, ys, [datas] = gridder.cut(x, y, [data], area)
    npt.assert_allclose(xs, [3, 4, 5])
    npt.assert_allclose(ys, [12, 13, 14])
    npt.assert_allclose(datas, [92, 24, 135])
    # Test on 2D arrays
    x = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
    y = np.array([[5, 7, 9],
                  [5, 7, 9],
                  [5, 7, 9]])
    data = np.array([[12, 84, 53],
                     [43, 79, 29],
                     [45, 27, 10]])
    area = [0.5, 2.5, 7, 9]
    xs, ys, [datas] = gridder.cut(x, y, [data], area)
    npt.assert_allclose(xs, [1, 1, 2, 2])
    npt.assert_allclose(ys, [7, 9, 7, 9])
    npt.assert_allclose(datas, [84, 53, 79, 29])
