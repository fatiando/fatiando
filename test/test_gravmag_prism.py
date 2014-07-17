import numpy as np
from numpy.testing import assert_array_almost_equal

from fatiando.mesher import Prism
from fatiando.gravmag import _prism_numpy, prism
from fatiando import utils


def test_cython_agains_numpy():
    "gravmag.prism numpy and cython implementations give same result"
    inc, dec = -30, 50
    model = [
        Prism(100, 300, -100, 100, 0, 400,
              {'density': -1., 'magnetization': utils.ang2vec(-2, inc, dec)}),
        Prism(-300, -100, -100, 100, 0, 200,
              {'density': 2., 'magnetization': utils.ang2vec(5, 25, -10)})]
    tmp = np.linspace(-500, 500, 101)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1 * np.ones_like(xp)
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz',
             'bx', 'by', 'bz', 'tf']
    for f in funcs:
        if f == 'tf':
            py = getattr(_prism_numpy, f)(xp, yp, zp, model, inc, dec)
            cy = getattr(prism, f)(xp, yp, zp, model, inc, dec)
        else:
            py = getattr(_prism_numpy, f)(xp, yp, zp, model)
            cy = getattr(prism, f)(xp, yp, zp, model)
        assert_array_almost_equal(py, cy, 8, 'Field = %s' % (f))
    kernels = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    for comp in kernels:
        for p in model:
            py = getattr(_prism_numpy, 'kernel' + comp)(xp, yp, zp, p)
            cy = getattr(prism, 'kernel' + comp)(xp, yp, zp, p)
            assert_array_almost_equal(py, cy, 8, 'Field = %s; Prism = %s'
                                      % (f, str(prism)))

