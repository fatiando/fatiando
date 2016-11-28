from __future__ import division, absolute_import, print_function
import numpy.testing as npt
import numpy as np

from ... import gridder
from ..interpolation import fill_nans


def test_fill_nans():
    "fill_nans recovers values that were removed from a simple grid"
    x, y, z = gridder.regular((0, 20, -10, 2), (10, 10), z=500)
    missing = np.zeros(x.shape, dtype=np.bool)
    missing[[0, 15, 67, 32, 12]] = True
    z_missing = np.copy(z)
    z_missing[missing] = np.nan
    assert np.any(z_missing != z)
    fill_nans(x, y, z, x, y, z_missing)
    npt.assert_allclose(z_missing, z)
    # Test using a masked array as well
    mask = np.zeros_like(z)
    mask[[35, 43, 1, 46, 83, 99, 14, 78]] = 1
    z_missing = np.ma.masked_array(z, mask=mask)
    assert np.any(z_missing.mask)
    fill_nans(x, y, z, x, y, z_missing)
    assert np.all(~z_missing.mask)
    npt.assert_allclose(z_missing, z)


def test_interp_at():
    "Interpolate a few points of smooth data gives correct results"
    # Generate some smooth data
    x, y = gridder.regular([0, 10, -10, -5], (50, 50))
    data = x**2 + y**2
    # Remove some points and interpolate them
    kept = np.ones(data.shape, dtype=np.bool)
    kept[[10, 51, 456, 2000, 1501]] = False
    removed = ~kept
    for algorithm in ['linear', 'cubic']:
        data_interp = gridder.interp_at(x[kept], y[kept], data[kept],
                                        x[removed], y[removed],
                                        algorithm=algorithm,
                                        extrapolate=False)
        npt.assert_allclose(data[removed], data_interp, rtol=0.01)


def test_interp_at_extrapolation():
    "Interpolate and extrapolate edge points that are not included"
    # Generate some smooth data
    x, y = gridder.regular([0, 10, -10, -5], (50, 50))
    data = np.ones_like(x)
    # Remove some points and interpolate them
    kept = np.ones(data.shape, dtype=np.bool)
    # All these points should interpolate to nan using cubic or linear because
    # they are on the edges of the grid and outside the convex hull of the kept
    # data. Using extrapolate=True will calculate these points use nearest
    # neighbors.
    kept[[0, 49, 2449, 2499]] = False
    removed = ~kept
    args = (x[kept], y[kept], data[kept], x[removed], y[removed])
    for algorithm in ['linear', 'cubic']:
        data_interp = gridder.interp_at(*args, algorithm=algorithm,
                                        extrapolate=False)
        # Make sure they are nans when using extrapolate=False
        assert np.all(np.isnan(data_interp))
        # Now test with extrapolation
        data_interp = gridder.interp_at(*args, algorithm=algorithm,
                                        extrapolate=True)
        npt.assert_allclose(data[removed], data_interp, rtol=0.01)


def test_interp():
    "Test interpolating on a regular grid using smooth data"
    # Generate some smooth data on a scatter of points
    area = [0, 10, -10, -5]
    # Use a lot of points to get a good interpolation
    x, y = gridder.scatter(area, n=50000, seed=0)

    def makedata(x, y):
        return x**3 + y**2

    data = makedata(x, y)
    shape = (50, 50)
    for algorithm in ['linear', 'cubic']:
        # Test passing in a specified area equal the true area
        xp, yp, datap = gridder.interp(x, y, data, shape, area=area,
                                       algorithm=algorithm, extrapolate=True)
        xpt, ypt = gridder.regular(area, shape)
        npt.assert_allclose(xp, xpt)
        npt.assert_allclose(yp, ypt)
        npt.assert_allclose(datap, makedata(xp, yp), rtol=0.05)
        # Test letting the function determine the area from the inputs
        xp, yp, datap = gridder.interp(x, y, data, shape, area=None,
                                       algorithm=algorithm, extrapolate=True)
        xpt, ypt = gridder.regular((xp.min(), xp.max(), yp.min(), yp.max()),
                                   shape)
        npt.assert_allclose(xp, xpt)
        npt.assert_allclose(yp, ypt)
        npt.assert_allclose(datap, makedata(xp, yp), rtol=0.05)


def test_profile():
    "Extracting a profile from smooth data works"
    # Generate some smooth data
    area = [0, 50, -50, 0]
    shape = (51, 51)
    x, y = gridder.regular(area, shape)

    def makedata(x, y):
        return x**3 + y**2

    data = makedata(x, y)
    for algorithm in ['linear', 'cubic']:
        # Test with two points along the x axis
        p1 = [20, 0]
        p2 = [20, -50]
        n = shape[0]
        xp, yp, dist, datap = gridder.profile(x, y, data, p1, p2, n,
                                              algorithm=algorithm)
        npt.assert_allclose(dist, np.linspace(0, area[1] - area[0], n))
        npt.assert_almost_equal(xp, np.zeros_like(xp) + 20)
        npt.assert_allclose(yp, np.linspace(area[2], area[3], shape[1])[::-1])
        npt.assert_allclose(datap, makedata(xp, yp))
        # Test with two points along the y axis
        p1 = [0, -25]
        p2 = [50, -25]
        n = shape[1]
        xp, yp, dist, datap = gridder.profile(x, y, data, p1, p2, n,
                                              algorithm=algorithm)
        npt.assert_allclose(dist, np.linspace(0, area[3] - area[2], n))
        npt.assert_almost_equal(yp, np.zeros_like(yp) - 25)
        npt.assert_allclose(xp, np.linspace(area[0], area[1], shape[0]))
        npt.assert_allclose(datap, makedata(xp, yp))
