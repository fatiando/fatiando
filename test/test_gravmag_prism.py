import numpy as np

from fatiando.mesher import Prism
from fatiando.gravmag import _prism, _cprism, _neprism
from fatiando import utils

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10**(-15)

def setup():
    global model, xp, yp, zp, inc, dec
    inc, dec = -30, 50
    reg_field = np.array(utils.dircos(inc, dec))
    model = [
        Prism(100, 300, -100, 100, 0, 400,
              {'density':1., 'magnetization':2*reg_field}),
        Prism(-300, -100, -100, 100, 0, 200,
            {'density':2., 'magnetization':utils.dircos(25, -10)})]
    tmp = np.linspace(-500, 500, 50)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1*np.ones_like(xp)

def test_potential():
    "gravmag.prism.potential python vs cython implementation"
    py = _prism.potential(xp, yp, zp, model)
    cy = _cprism.potential(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gx():
    "gravmag.prism.gx python vs cython implementation"
    py = _prism.gx(xp, yp, zp, model)
    cy = _cprism.gx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gy():
    "gravmag.prism.gy python vs cython implementation"
    py = _prism.gy(xp, yp, zp, model)
    cy = _cprism.gy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gz():
    "gravmag.prism.gz python vs cython implementation"
    py = _prism.gz(xp, yp, zp, model)
    cy = _cprism.gz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxx():
    "gravmag.prism.gxx python vs cython implementation"
    py = _prism.gxx(xp, yp, zp, model)
    cy = _cprism.gxx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxy():
    "gravmag.prism.gxy python vs cython implementation"
    py = _prism.gxy(xp, yp, zp, model)
    cy = _cprism.gxy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxz():
    "gravmag.prism.gxx python vs cython implementation"
    py = _prism.gxz(xp, yp, zp, model)
    cy = _cprism.gxz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyy():
    "gravmag.prism.gyy python vs cython implementation"
    py = _prism.gyy(xp, yp, zp, model)
    cy = _cprism.gyy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyz():
    "gravmag.prism.gyz python vs cython implementation"
    py = _prism.gyz(xp, yp, zp, model)
    cy = _cprism.gyz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gzz():
    "gravmag.prism.gzz python vs cython implementation"
    py = _prism.gzz(xp, yp, zp, model)
    cy = _cprism.gzz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_tf():
    "gravmag.prism.tf python vs cython implementation"
    py = _prism.tf(xp, yp, zp, model, inc, dec)
    cy = _cprism.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_potential_ne():
    "gravmag.prism.potential python vs numexpr implementation"
    py = _prism.potential(xp, yp, zp, model)
    ne = _neprism.potential(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gx_ne():
    "gravmag.prism.gx python vs numexpr implementation"
    py = _prism.gx(xp, yp, zp, model)
    ne = _neprism.gx(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gy_ne():
    "gravmag.prism.gy python vs numexpr implementation"
    py = _prism.gy(xp, yp, zp, model)
    ne = _neprism.gy(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gz_ne():
    "gravmag.prism.gz python vs numexpr implementation"
    py = _prism.gz(xp, yp, zp, model)
    ne = _neprism.gz(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxx_ne():
    "gravmag.prism.gxx python vs numexpr implementation"
    py = _prism.gxx(xp, yp, zp, model)
    ne = _neprism.gxx(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxy_ne():
    "gravmag.prism.gxy python vs numexpr implementation"
    py = _prism.gxy(xp, yp, zp, model)
    ne = _neprism.gxy(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxz_ne():
    "gravmag.prism.gxz python vs numexpr implementation"
    py = _prism.gxz(xp, yp, zp, model)
    ne = _neprism.gxz(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyy_ne():
    "gravmag.prism.gyy python vs numexpr implementation"
    py = _prism.gyy(xp, yp, zp, model)
    ne = _neprism.gyy(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyz_ne():
    "gravmag.prism.gyz python vs numexpr implementation"
    py = _prism.gyz(xp, yp, zp, model)
    ne = _neprism.gyz(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gzz_ne():
    "gravmag.prism.gzz python vs numexpr implementation"
    py = _prism.gzz(xp, yp, zp, model)
    ne = _neprism.gzz(xp, yp, zp, model)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_tf_ne():
    "gravmag.prism.tf python vs numexpr implementation"
    py = _prism.tf(xp, yp, zp, model, inc, dec)
    ne = _neprism.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(py - ne)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))
