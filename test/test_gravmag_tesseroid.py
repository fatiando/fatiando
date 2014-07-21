import numpy as np
from numpy.testing import assert_array_almost_equal

from fatiando.gravmag import tesseroid, half_sph_shell
from fatiando.mesher import Tesseroid


def test_tesseroid_vs_spherical_shell():
    "gravmag.tesseroid equals analytical solution of spherical half-shell"
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
