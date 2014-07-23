import numpy as np
from numpy.testing import assert_array_almost_equal

from fatiando.gravmag import tesseroid, half_sph_shell
from fatiando.mesher import Tesseroid
from fatiando import gridder


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
