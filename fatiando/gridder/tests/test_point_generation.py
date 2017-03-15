from __future__ import division, absolute_import, print_function
from future.builtins import range
import numpy.testing as npt
import numpy as np
from pytest import raises

from ... import gridder


def test_regular():
    "Regular grid generation works in the correct order of points in the array"
    shape = (5, 3)
    x, y = gridder.regular((0, 10, 0, 5), shape)
    x_true = np.array([[0., 0., 0.],
                       [2.5, 2.5, 2.5],
                       [5., 5., 5.],
                       [7.5, 7.5, 7.5],
                       [10., 10., 10.]])
    npt.assert_allclose(x.reshape(shape), x_true)
    y_true = np.array([[0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.]])
    npt.assert_allclose(y.reshape(shape), y_true)
    # Test that the z variable is returned correctly
    x, y, z = gridder.regular((0, 10, 0, 5), shape, z=-10)
    z_true = -10 + np.zeros(shape)
    npt.assert_allclose(z.reshape(shape), z_true)
    # Test a case with a single value in x
    shape = (1, 3)
    x, y = gridder.regular((0, 0, 0, 5), shape)
    x_true = np.array([[0., 0., 0.]])
    npt.assert_allclose(x.reshape(shape), x_true)
    y_true = np.array([[0., 2.5, 5.]])
    npt.assert_allclose(y.reshape(shape), y_true)


def test_regular_fails():
    "gridder.regular should fail for invalid input"
    # If the area parameter is specified in the wrong order
    with raises(AssertionError):
        x, y = gridder.regular((1, -1, 0, 10), (20, 12))
    with raises(AssertionError):
        x, y = gridder.regular((0, 10, 1, -1), (20, 12))


def test_scatter():
    "Scatter point generation returns sane values with simple inputs"
    # Can't test random points for equality. So I'll test that the values are
    # in the correct ranges.
    area = (-1287, 5433, 0.1234, 0.1567)
    xmin, xmax, ymin, ymax = area
    n = 10000000
    x, y = gridder.scatter(area, n=n, seed=0)
    assert x.size == n
    assert y.size == n
    npt.assert_almost_equal(x.min(), xmin, decimal=1)
    npt.assert_almost_equal(x.max(), xmax, decimal=1)
    npt.assert_almost_equal(y.min(), ymin, decimal=1)
    npt.assert_almost_equal(y.max(), ymax, decimal=1)
    npt.assert_almost_equal(x.mean(), (xmax + xmin)/2, decimal=1)
    npt.assert_almost_equal(y.mean(), (ymax + ymin)/2, decimal=1)
    # Test that the z array is correct
    n = 1000
    x, y, z = gridder.scatter(area, n=n, z=-150, seed=0)
    assert z.size == n
    npt.assert_allclose(z, -150 + np.zeros(n))


def test_scatter_fails():
    "gridder.scatter should fail for invalid input"
    # If the area parameter is specified in the wrong order
    with raises(AssertionError):
        x, y = gridder.scatter((1, -1, 0, 10), 20, seed=1)
    with raises(AssertionError):
        x, y = gridder.scatter((0, 10, 1, -1), 20, seed=2)


def test_scatter_noseed():
    "gridder.scatter returns different sequence after using random seed"
    area = [0, 1000, 0, 1000]
    z = 20
    size = 1000
    seed = 1242
    x1, y1, z1 = gridder.scatter(area, size, z, seed=seed)
    x2, y2, z2 = gridder.scatter(area, size, z, seed=seed)
    npt.assert_allclose(x1, x2)
    npt.assert_allclose(y1, y2)
    npt.assert_allclose(z1, z2)
    x3, y3, z3 = gridder.scatter(area, size, z)
    with raises(AssertionError):
        npt.assert_allclose(x1, x3)
    with raises(AssertionError):
        npt.assert_allclose(y1, y3)
    npt.assert_allclose(z1, z3)


def test_circular_scatter():
    """
    gridder.circular_scatter returns equidistant points if random=False.
    """
    area = [0, 1000, 0, 1000]
    size = 1000
    x, y = gridder.circular_scatter(area, size, random=False)
    distances = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    npt.assert_allclose(distances, distances[0]*np.ones(size-1), rtol=1e-09)


def test_circular_scatter_z():
    """
    gridder.circular_scatter returns a filled z array.
    """
    area = [0, 1000, 0, 1000]
    size = 1000
    z_value = 500
    x, y, z = gridder.circular_scatter(area, size, z=z_value, random=False)
    assert z.size == size
    npt.assert_allclose(z, z_value + np.zeros(size))


def test_circular_scatter_num_point():
    "gridder.circular_scatter returns a specified ``n`` number of points."
    area = [0, 1000, 0, 1000]
    size = 1000
    x, y = gridder.circular_scatter(area, size, random=False)
    assert x.size == size and y.size == size
    x, y = gridder.circular_scatter(area, size, random=True)
    assert x.size == size and y.size == size


def test_circular_scatter_random():
    "gridder.circular_scatter return different sequences if random=True"
    area = [-1000, 1200, -40, 200]
    size = 1300
    for i in range(20):
        x1, y1 = gridder.circular_scatter(area, size, random=True)
        x2, y2 = gridder.circular_scatter(area, size, random=True)
        with raises(AssertionError):
            npt.assert_allclose(x1, x2)
        with raises(AssertionError):
            npt.assert_allclose(y1, y2)


def test_circular_scatter_seed():
    "gridder.circular_scatter returns same sequence using same random seed"
    area = [0, 1000, 0, 1000]
    size = 1000
    for seed in range(10):
        x1, y1 = gridder.circular_scatter(area, size, random=True, seed=seed)
        x2, y2 = gridder.circular_scatter(area, size, random=True, seed=seed)
        npt.assert_allclose(x1, x2)
        npt.assert_allclose(y1, y2)
