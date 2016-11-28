from __future__ import division, absolute_import, print_function
import numpy.testing as npt
import numpy as np
from pytest import raises
import scipy.optimize
from numpy.random import RandomState

from ... import gridder


def test_fails_if_bad_pad_operation():
    'gridder.pad_array fails if given a bad padding array operation option'
    p = 'foo'
    shape = (100, 100)
    x, y, z = gridder.regular((-1000., 1000., -1000., 1000.), shape, z=-150)
    g = z.reshape(shape)
    raises(ValueError, gridder.pad_array, g, padtype=p)


def test_pad_and_unpad_equal_2d():
    'gridder.pad_array and subsequent .unpad_array gives original array: 2D'
    shape = (100, 101)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    # rosenbrock: (a-x)^2 + b(y-x^2)^2  a=1 b=100 usually
    X = x.reshape(shape)
    Y = y.reshape(shape)
    gz = scipy.optimize.rosen([Y/100000., X/100000.])
    pads = ['mean', 'edge', 'lintaper', 'reflection', 'oddreflection',
            'oddreflectiontaper', '0']
    for p in pads:
        gpad, nps = gridder.pad_array(gz, padtype=p)
        gunpad = gridder.unpad_array(gpad, nps)
        npt.assert_allclose(gunpad, gz)


def test_pad_and_unpad_equal_1d():
    'gridder.pad_array and subsequent .unpad_array gives original array: 1D'
    x = np.array([3, 4, 4, 5, 6])
    xpad_true = np.array([4.4, 3.2, 3, 4, 4, 5, 6, 4.4])
    xpad, nps = gridder.pad_array(x)
    npt.assert_allclose(xpad_true, xpad)
    assert nps == [(2, 1)]
    xunpad = gridder.unpad_array(xpad, nps)
    npt.assert_allclose(xunpad, x)
    # Using a custom number of padding elements
    xpad, nps = gridder.pad_array(x, npd=(10,))
    assert nps == [(3, 2)]
    npt.assert_allclose(xpad[3:-2], x)
    xunpad = gridder.unpad_array(xpad, nps)
    npt.assert_allclose(xunpad, x)


def test_coordinatevec_padding_1d():
    'gridder.padcoords accurately pads coordinate vector in 1D'
    prng = RandomState(12345)
    f = prng.rand(72) * 10
    x = np.arange(100, 172)
    fpad, nps = gridder.pad_array(f)
    N = gridder.pad_coords(x, f.shape, nps)
    npt.assert_allclose(N[0][nps[0][0]:-nps[0][1]], x)


def test_coordinatevec_padding_2d():
    'gridder.padcoords accurately pads coordinate vector in 2D'
    shape = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    gz = np.zeros(shape)
    xy = []
    xy.append(x)
    xy.append(y)
    gpad, nps = gridder.pad_array(gz)
    N = gridder.pad_coords(xy, gz.shape, nps)
    Yp = N[1].reshape(gpad.shape)
    Xp = N[0].reshape(gpad.shape)
    assert N[0].reshape(gpad.shape).shape == gpad.shape
    npt.assert_allclose(Yp[nps[0][0]:-nps[0][1], nps[1][0]:-nps[1][1]].ravel(),
                        y)
    npt.assert_allclose(Xp[nps[0][0]:-nps[0][1], nps[1][0]:-nps[1][1]].ravel(),
                        x)


def test_fails_if_npd_incorrect_dimension():
    'gridder.pad_array raises error if given improper dimension on npadding'
    s = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), s, z=-150)
    g = z.reshape(s)
    npdt = 128
    raises(ValueError, gridder.pad_array, g, npd=npdt)
    npdt = (128, 256, 142)
    raises(ValueError, gridder.pad_array, g, npd=npdt)
    prng = RandomState(12345)
    g = prng.rand(50)
    raises(ValueError, gridder.pad_array, g, npd=npdt)


def test_fails_if_npd_lessthan_arraydim():
    'gridder.pad_array raises error if given npad is less than array length'
    shape = (101, 172)
    x, y, z = gridder.regular((-5000., 5000., -5000., 5000.), shape, z=-150)
    g = z.reshape(shape)
    npdt = (128, 128)
    raises(ValueError, gridder.pad_array, g, npd=npdt)
    prng = RandomState(12345)
    g = prng.rand(20)
    npdt = 16
    raises(ValueError, gridder.pad_array, g, npd=npdt)
