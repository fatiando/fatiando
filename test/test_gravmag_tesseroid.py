from __future__ import division
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_raises
import multiprocessing
import warnings

from fatiando.gravmag import tesseroid
from fatiando.mesher import Tesseroid, TesseroidMesh
from fatiando import gridder
from fatiando.constants import SI2MGAL, SI2EOTVOS, G, MEAN_EARTH_RADIUS


def test_warn_if_division_makes_too_small():
    "gravmag.tesseroid warn if not dividing further bc tesseroid got too small"
    # When tesseroids get below a threshold, they should not divide further and
    # compute as is instead. Otherwise results in ZeroDivisionError involving
    # some trigonometric functions.
    ds = 1e-6
    models = [
        [Tesseroid(-ds, ds, -ds, ds, 0, -1000, {'density': 100})],
        [Tesseroid(-1e-3, 1e-3, -1e-3, 1e-3, 0, -1e-2, {'density': 100})]]
    lat, lon = np.zeros((2, 1))
    h = np.array([0.1])
    warning_msg = (
        "Stopped dividing a tesseroid because it's dimensions would be below "
        + "the minimum numerical threshold (1e-6 degrees or 1e-3 m). "
        + "Will compute without division. Cannot guarantee the accuracy of "
        + "the solution.")
    for i, model in enumerate(models):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            tesseroid.gz(lon, lat, h, model)
            msg = ("Failed model {}. Got {} warnings.\n\n".format(i, len(w))
                   + "\n\n".join([str(j.message) for j in w]))
            assert len(w) >= 1, msg
            assert any(issubclass(j.category, RuntimeWarning) for j in w), \
                "No RuntimeWarning found. " + msg
            assert any(warning_msg in str(j.message) for j in w), \
                "Message mismatch. " + msg


def test_warn_if_too_small():
    "gravmag.tesseroid warns if ignoring tesseroid that is too small"
    ds = 1e-6/2
    models = [
        [Tesseroid(-ds, ds, -ds, ds, 0, -1000, {'density': 100})],
        [Tesseroid(-1e-2, 1e-2, -1e-2, 1e-2, 0, -1e-4, {'density': 100})]]
    lat, lon = np.zeros((2, 1))
    h = np.array([10])
    warning_msg = (
        "Encountered tesseroid with dimensions smaller than the "
        + "numerical threshold (1e-6 degrees or 1e-3 m). "
        + "Ignoring this tesseroid.")
    for i, model in enumerate(models):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            tesseroid.gz(lon, lat, h, model)
            msg = ("Failed model {}. Got {} warnings.\n\n".format(i, len(w))
                   + "\n\n".join([str(j.message) for j in w]))
            assert len(w) >= 1, msg
            assert any(issubclass(j.category, RuntimeWarning) for j in w), \
                "No RuntimeWarning found. " + msg
            assert any(warning_msg in str(j.message) for j in w), \
                "Message mismatch. " + msg


def test_pool_as_argument():
    "gravmag.tesseroid takes an open Pool as argument and uses it"
    class MockPool(object):
        "Record if the map method of pool was used."
        def __init__(self, pool):
            self.pool = pool
            self.used = False

        def map(self, *args):
            res = self.pool.map(*args)
            self.used = True
            return res
    njobs = 2
    pool = MockPool(multiprocessing.Pool(njobs))
    model = [Tesseroid(0, 1, 0, 1, 2000, 0, {'density': 600})]
    lon, lat, height = gridder.regular((-1, 2, -1, 2), (20, 20), z=250e3)
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        func = getattr(tesseroid, f)
        f1 = func(lon, lat, height, model)
        f2 = func(lon, lat, height, model, njobs=njobs, pool=pool)
        assert_allclose(f1, f2, err_msg="Mismatch for {}".format(f))
        assert pool.used, "The given pool was not used in {}".format(f)
        with assert_raises(AssertionError):
            func(lon, lat, height, model, njobs=1, pool=pool)


def test_ignore_zero_volume():
    "gravmag.tesseroid ignores tesseroids with 0 volume"
    props = dict(density=2000)
    model = [Tesseroid(-10, 0, 4, 5, 1000.1, 1000.1, props),
             Tesseroid(-10, 0, 4, 5, 1000.001, 1000, props),
             Tesseroid(-10, 0, 3.999999999, 4, 1000, 0, props),
             Tesseroid(-10, -9.9999999999, 4, 5, 1000, 0, props),
             Tesseroid(5, 10, -10, -5, 2000.5, 0, props)]
    lon, lat, height = gridder.regular((-20, 20, -20, 20), (50, 50), z=250e3)
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        with warnings.catch_warnings(record=True) as w:
            func = getattr(tesseroid, f)
            f1 = func(lon, lat, height, model)
            f2 = func(lon, lat, height, [model[-1]])
        assert_allclose(f1, f2, err_msg="Mismatch for {}".format(f))


def test_detect_invalid_tesseroid_dimensions():
    "gravmag.tesseroid raises error when tesseroids with bad dimensions"
    props = dict(density=2000)
    model = [Tesseroid(0, -10, 4, 5, 1000, 0, props),
             Tesseroid(-10, 0, 5, 4, 1000, 0, props),
             Tesseroid(-10, 0, 5, 4, 0, 1000, props)]
    lon, lat, height = gridder.regular((-20, 20, -20, 20), (50, 50), z=250e3)
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        func = getattr(tesseroid, f)
        for t in model:
            assert_raises(AssertionError, func, lon, lat, height, [t])


def test_serial_vs_parallel():
    "gravmag.tesseroid serial and parallel execution give same result"
    model = TesseroidMesh((-1, 1.5, -2, 2, 0, -10e3), (3, 2, 1))
    model.addprop('density', 500*np.ones(model.size))
    lon, lat, height = gridder.regular((-1, 1.5, -2, 2), (15, 21), z=150e3)
    njobs = 3
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        func = getattr(tesseroid, f)
        serial = func(lon, lat, height, model, njobs=1)
        parallel = func(lon, lat, height, model, njobs=njobs)
        assert_allclose(serial, parallel, err_msg="Mismatch for {}".format(f))


def test_numba_vs_python():
    "gravmag.tesseroid numba and pure python implementations give same result"
    model = TesseroidMesh((0.3, 0.6, 0.2, 0.8, 1000, 0), (2, 2, 1))
    model.addprop('density', -200*np.ones(model.size))
    lon, lat, height = gridder.regular((0, 1, 0, 2), (20, 20), z=250e3)
    for f in 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split():
        with warnings.catch_warnings(record=True) as w:
            func = getattr(tesseroid, f)
            py = func(lon, lat, height, model, engine='numpy')
            nb = func(lon, lat, height, model, engine='numba')
        assert_allclose(nb, py, err_msg="Mismatch for {}".format(f))


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
