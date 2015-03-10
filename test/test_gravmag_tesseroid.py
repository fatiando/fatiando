import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises

from fatiando.gravmag import tesseroid, half_sph_shell
from fatiando.mesher import Tesseroid
from fatiando import gridder


def test_queue_overflow():
    "gravmag.tesseroid raises exceptions on queue overflow"
    model = [Tesseroid(0, 1, 0, 1, 0, -20e4, {'density': 2600})]
    area = [0, 1, 0, 1]
    shape = [20, 20]
    lon, lat, h = gridder.regular(area, shape, z=1000)
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    backup = tesseroid.QUEUE_SIZE
    tesseroid.QUEUE_SIZE = 5
    for f in fields:
        assert_raises(ValueError, getattr(tesseroid, f), lon, lat, h, model)
    # Check if overflows on normal queue size when trying to calculated on top
    # of the tesseroid
    tesseroid.QUEUE_SIZE = 20
    lon, lat, h = np.array([0.5]), np.array([0.5]), np.array([0])
    for f in fields:
        assert_raises(ValueError, getattr(tesseroid, f), lon, lat, h, model)
    # Restore the module default queue size
    tesseroid.QUEUE_SIZE = backup


def test_fails_if_shape_mismatch():
    'gravmag.tesseroid fails if given computation points with different shapes'
    model = [Tesseroid(0, 1, 0, 1, 1000, -20000, {'density': 2670})]
    area = [-2, 2, -2, 2]
    shape = (51, 51)
    lon, lat, h = gridder.regular(area, shape, z=100000)

    assert_raises(ValueError, tesseroid.potential, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.potential, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.potential, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.potential, lon[:-5], lat, h[:-2],
                  model)

    assert_raises(ValueError, tesseroid.gx, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gx, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gx, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gx, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gy, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gy, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gy, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gy, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gz, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gz, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gz, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gz, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gxx, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gxx, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gxx, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gxx, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gxy, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gxy, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gxy, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gxy, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gxz, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gxz, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gxz, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gxz, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gyy, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gyy, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gyy, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gyy, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gyz, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gyz, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gyz, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gyz, lon[:-5], lat, h[:-2], model)

    assert_raises(ValueError, tesseroid.gzz, lon[:-2], lat, h, model)
    assert_raises(ValueError, tesseroid.gzz, lon, lat[:-2], h, model)
    assert_raises(ValueError, tesseroid.gzz, lon, lat, h[:-2], model)
    assert_raises(ValueError, tesseroid.gzz, lon[:-5], lat, h[:-2], model)


def test_tesseroid_vs_spherical_shell():
    "gravmag.tesseroid equal analytical solution of spherical half-shell to 1%"
    tlons = np.linspace(-90, 90, 45, endpoint=False)
    tlats = np.linspace(-90, 90, 45, endpoint=False)
    wsize = tlons[1] - tlons[0]
    ssize = tlats[1] - tlats[0]
    density = 1000.
    props = {'density': density}
    top = 0
    bottom = -100000
    shellmodel = [Tesseroid(w, w + wsize, s, s + ssize, top, bottom, props)
                  for w in tlons for s in tlats]
    heights = np.linspace(10000, 1000000, 10)
    lons = np.zeros_like(heights)
    lats = lons
    funcs = ['potential', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
             'gzz']
    for f in funcs:
        shell = getattr(half_sph_shell, f)(heights, top, bottom, density)
        tess = getattr(tesseroid, f)(lons, lats, heights, shellmodel)
        diff = np.abs(shell - tess)
        factor = np.abs(shell).max()
        if factor > 1e-10:
            shell /= factor
            tess /= factor
            precision = 2  # 1% of the maximum
        else:
            precision = 10  # For the components that are zero
        assert_array_almost_equal(shell, tess, precision,
                                  'Failed %s: max diff %.15g'
                                  % (f, diff.max()))


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
    lon, lat, h = gridder.regular(area, shape, z=50000)
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
    lon, lat, h = gridder.regular(area, shape, z=50000)
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
