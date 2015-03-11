from __future__ import division
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_raises

from fatiando.gravmag import tesseroid
from fatiando.mesher import Tesseroid, TesseroidMesh
from fatiando import gridder
from fatiando.constants import SI2MGAL, SI2EOTVOS, G, MEAN_EARTH_RADIUS


def test_numba_vs_python():
    "gravmag.tesseroid numba and pure python implementations give same result"
    model = TesseroidMesh((0, 1, 0, 2, 1000, 0), (5, 5, 5))
    lon, lat, height = gridder.regular((0, 1, 0, 2), (30, 30), z=50e3)
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        func = getattr(tesseroid, f)
        py = func(lon, lat, height, model, engine='numpy')
        nb = func(lon, lat, height, model, engine='numba')
        assert_allclose(np, py, err_msg="Mismatch for {}".format(f))


def test_fails_if_shape_mismatch():
    'gravmag.tesseroid fails if given computation points with different shapes'
    model = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670})]
    area = [-2, 2, -2, 2]
    shape = (51, 51)
    lon, lat, h = gridder.regular(area, shape, z=100000)

    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        func = getattr(tesseroid, f)
        assert_raises(AssertionError, func, lon[:-2], lat, h, model)
        assert_raises(AssertionError, func, lon, lat[:-2], h, model)
        assert_raises(AssertionError, func, lon, lat, h[:-2], model)
        assert_raises(AssertionError, func, lon[:-5], lat, h[:-2], model)


def calc_shell_effect(height, top, bottom, density):
    "Calculate the effects of a homogeneous spherical shell"
    r = height + MEAN_EARTH_RADIUS
    # top and bottom are heights
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    potential = (4/3)*np.pi*G*density*(r2**3 - r1**3)/r
    data = {'potential': potential,
            'gx': 0,
            'gy': 0,
            'gz': SI2MGAL*(potential/r),
            'gxx': SI2EOTVOS*(-potential/r**2),
            'gxy': 0,
            'gxz': 0,
            'gyy': SI2EOTVOS*(-potential/r**2),
            'gyz': 0,
            'gzz': SI2EOTVOS*(2*potential/r**2)}
    return data


def test_tesseroid_vs_spherical_shell():
    "gravmag.tesseroid equal analytical solution of spherical shell to 0.1%"
    density = 1000.
    top = 1000
    bottom = 0
    model = TesseroidMesh((0, 360, -90, 90, top, bottom), (1, 6, 12))
    model.addprop('density', density*np.ones(model.size))
    h = 10000
    lon, lat, height = gridder.regular((0, model.dims[0], 0, model.dims[1]),
                                       (10, 10), z=h)
    funcs = ['potential', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
             'gzz']
    shellvalues = calc_shell_effect(h, top, bottom, density)
    for f in funcs:
        shell = shellvalues[f]
        tess = getattr(tesseroid, f)(lon, lat, height, model)
        diff = np.abs(shell - tess)
        # gz gy and the off-diagonal gradients should be zero so I can't
        # calculate a relative error (in %).
        # To do that, I'll use the gz and gzz shell values to calculate the
        # percentage.
        if f in 'gx gy'.split():
            shell = shellvalues['gz']
        elif f in 'gxy gxz gyz'.split():
            shell = shellvalues['gzz']
        diff = 100*diff/np.abs(shell)
        assert diff.max() < 0.1, "diff > 0.1% for {}: {}".format(
            f, diff.max())


def test_skip_none_and_missing_properties():
    "gravmag.tesseroid ignores Nones and tesseroids without density prop"
    model = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670}),
             None,
             Tesseroid(-1.5, -0.5, -1.5, -1, -1000, -20000),
             Tesseroid(0.1, 0.6, -0.8, -0.3, 10000, -20000,
                       {'magnetization': [1, 2, 3]}),
             None,
             None,
             Tesseroid(-1.5, -0.5, -1.5, -1, 1000, -20000,
                       {'density': 2000, 'magnetization': [1, 2, 3]}),
             None]
    puremodel = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670}),
                 Tesseroid(-1.5, -0.5, -1.5, -1, 1000, -20000,
                           {'density': 2000})]
    area = [-2, 2, -2, 2]
    shape = (51, 51)
    lon, lat, h = gridder.regular(area, shape, z=150000)
    funcs = ['potential', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
             'gzz']
    for f in funcs:
        pure = getattr(tesseroid, f)(lon, lat, h, puremodel)
        dirty = getattr(tesseroid, f)(lon, lat, h, model)
        assert_array_almost_equal(pure, dirty, 9, 'Failed %s' % (f))


def test_overwrite_density():
    "gravmag.tesseroid uses given density instead of tesseroid property"
    model = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670})]
    density = -1000
    other = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': density})]
    area = [-2, 2, -2, 2]
    shape = (51, 51)
    lon, lat, h = gridder.regular(area, shape, z=250000)
    funcs = ['potential', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
             'gzz']
    for f in funcs:
        correct = getattr(tesseroid, f)(lon, lat, h, other)
        effect = getattr(tesseroid, f)(lon, lat, h, model, dens=density)
        assert_array_almost_equal(correct, effect, 9, 'Failed %s' % (f))


def test_laplace_equation():
    "gravmag.tesseroid obeys Laplace equation"
    model = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670}),
             Tesseroid(-1.5, 1.5, -1.5, -1, -1000, -20000, {'density': -1000}),
             Tesseroid(0.1, 0.6, -0.8, -0.3, 10000, -20000, {'density': 2000}),
             ]
    area = [-2, 2, -2, 2]
    shape = (51, 51)
    lon, lat, h = gridder.regular(area, shape, z=50000)
    gxx = tesseroid.gxx(lon, lat, h, model)
    gyy = tesseroid.gyy(lon, lat, h, model)
    gzz = tesseroid.gzz(lon, lat, h, model)
    trace = gxx + gyy + gzz
    assert_array_almost_equal(trace, np.zeros_like(lon), 9,
                              'Failed whole model. Max diff %.15g'
                              % (np.abs(trace).max()))
    for tess in model:
        gxx = tesseroid.gxx(lon, lat, h, [tess])
        gyy = tesseroid.gyy(lon, lat, h, [tess])
        gzz = tesseroid.gzz(lon, lat, h, [tess])
        trace = gxx + gyy + gzz
        assert_array_almost_equal(trace, np.zeros_like(lon), 9,
                                  'Failed tesseroid %s. Max diff %.15g'
                                  % (str(tess), np.abs(trace).max()))


def test_stack_overflow():
    "gravmag.tesseroid raises exceptions on stack overflow"
    model = [Tesseroid(0, 1, 0, 1, 0, -20e4, {'density': 2600})]
    area = [0, 1, 0, 1]
    shape = [20, 20]
    lon, lat, h = gridder.regular(area, shape, z=1000)
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    backup = tesseroid.STACK_SIZE
    tesseroid.STACK_SIZE = 5
    for f in fields:
        assert_raises(OverflowError, getattr(tesseroid, f), lon, lat, h, model)
    # Check if overflows on normal queue size when trying to calculated on top
    # of the tesseroid
    tesseroid.STACK_SIZE = 20
    lon, lat, h = np.array([0.5]), np.array([0.5]), np.array([0])
    for f in fields:
        assert_raises(OverflowError, getattr(tesseroid, f), lon, lat, h, model)
    # Restore the module default queue size
    tesseroid.STACK_SIZE = backup
