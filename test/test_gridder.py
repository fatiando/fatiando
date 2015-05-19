from __future__ import division
from fatiando import gridder

import numpy as np
from numpy.testing import assert_almost_equal


def test_spacing():
    "gridder.spacing returns correct results"
    dx, dy = 1.5, 3.3
    ny, nx = 30, 41
    area = [0, dx*(nx - 1), 0, dy*(ny - 1)]
    dy_, dx_ = gridder.spacing(area, (ny, nx))
    assert_almost_equal(dx_, dx, decimal=10, err_msg="Failed dx")
    assert_almost_equal(dy_, dy, decimal=10, err_msg="Failed dy")


def test_regular():
    "gridder.regular generates the correct output"
    shape = (28, 17)
    areas = [[0, 1, 0, 1],
             [-1, 1, -1, 1],
             [-1000, 0, 0, 5000],
             [0, 501, -1, 0.1]]
    for area in areas:
        x1, x2, y1, y2 = area
        ny, nx = shape
        x, y = gridder.regular(area, shape)
        i = 0
        for yp in np.linspace(y1, y2, ny):
            for xp in np.linspace(x1, x2, nx):
                msg = 'area {}: expected {}, {}'.format(area, xp, yp)
                assert_almost_equal(x[i], xp, decimal=10, err_msg=msg)
                assert_almost_equal(y[i], yp, decimal=10, err_msg=msg)
                i += 1


def test_scatter():
    "gridder.scatter returns random results"
    x, y = gridder.scatter([0, 1, 0, 1], 100)
    assert not np.abs(x - y).max() < 1e-15, "x and y are equal"

    x1, y1 = gridder.scatter([0, 5, 0, 1], 100, seed=1)
    x2, y2 = gridder.scatter([0, 5, 0, 1], 100, seed=2)
    assert not np.abs(x1 - x2).max() < 1e-15, "x1 and x2 are equal"
    assert not np.abs(y1 - y2).max() < 1e-15, "y1 and y2 are equal"

    x1, y1 = gridder.scatter([0, 5, 0, 1], 100, seed=1)
    x2, y2 = gridder.scatter([0, 5, 0, 1], 100, seed=1)
    assert np.abs(x1 - x2).max() < 1e-15, "x1 and x2 are different"
    assert np.abs(y1 - y2).max() < 1e-15, "y1 and y2 are different"


def test_grid_regular():
    "gridder.Grid.regular generates the correct output"
    shape = (28, 17)
    areas = [[0, 1, 0, 1],
             [-1, 1, -1, 1],
             [-1000, 0, 0, 5000],
             [0, 501, -1, 0.1]]
    for area in areas:
        x1, x2, y1, y2 = area
        ny, nx = shape
        g = gridder.Grid.regular(area, shape)
        g2 = gridder.Grid.regular(area, shape, latlon=True)
        i = 0
        for yp in np.linspace(y1, y2, ny):
            for xp in np.linspace(x1, x2, nx):
                msg = 'area {}: expected {}, {}'.format(area, xp, yp)
                assert_almost_equal(g.x[i], xp, decimal=10, err_msg=msg)
                assert_almost_equal(g.y[i], yp, decimal=10, err_msg=msg)
                assert_almost_equal(g2.lat[i], xp, decimal=10, err_msg=msg)
                assert_almost_equal(g2.lon[i], yp, decimal=10, err_msg=msg)
                i += 1


def test_grid_scatter():
    "gridder.Grid.scatter returns random results"
    g = gridder.Grid.scatter([0, 1, 0, 1], 100)
    assert not np.abs(g.x - g.y).max() < 1e-15, "x and y are equal"

    g1 = gridder.Grid.scatter([0, 5, 0, 1], 100, seed=1)
    g2 = gridder.Grid.scatter([0, 5, 0, 1], 100, seed=2)
    assert not np.abs(g1.x - g2.x).max() < 1e-15, "x1 and x2 are equal"
    assert not np.abs(g1.y - g2.y).max() < 1e-15, "y1 and y2 are equal"

    g1 = gridder.Grid.scatter([0, 5, 0, 1], 100, seed=1)
    g2 = gridder.Grid.scatter([0, 5, 0, 1], 100, seed=1)
    assert np.abs(g1.x - g2.x).max() < 1e-15, "x1 and x2 are different"
    assert np.abs(g1.y - g2.y).max() < 1e-15, "y1 and y2 are different"
