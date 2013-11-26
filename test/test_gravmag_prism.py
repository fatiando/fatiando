import numpy as np

from fatiando.mesher import Prism
from fatiando.gravmag import _prism_numpy, prism
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
              {'density':1., 'magnetization':2}),
        Prism(-300, -100, -100, 100, 0, 200,
            {'density':2., 'magnetization':utils.dircos(25, -10)})]
    tmp = np.linspace(-500, 500, 50)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1*np.ones_like(xp)

def test_potential():
    "gravmag.prism.potential python vs cython implementation"
    py = _prism_numpy.potential(xp, yp, zp, model)
    cy = prism.potential(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gx():
    "gravmag.prism.gx python vs cython implementation"
    py = _prism_numpy.gx(xp, yp, zp, model)
    cy = prism.gx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gy():
    "gravmag.prism.gy python vs cython implementation"
    py = _prism_numpy.gy(xp, yp, zp, model)
    cy = prism.gy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gz():
    "gravmag.prism.gz python vs cython implementation"
    py = _prism_numpy.gz(xp, yp, zp, model)
    cy = prism.gz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxx():
    "gravmag.prism.gxx python vs cython implementation"
    py = _prism_numpy.gxx(xp, yp, zp, model)
    cy = prism.gxx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxy():
    "gravmag.prism.gxy python vs cython implementation"
    py = _prism_numpy.gxy(xp, yp, zp, model)
    cy = prism.gxy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gxz():
    "gravmag.prism.gxx python vs cython implementation"
    py = _prism_numpy.gxz(xp, yp, zp, model)
    cy = prism.gxz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyy():
    "gravmag.prism.gyy python vs cython implementation"
    py = _prism_numpy.gyy(xp, yp, zp, model)
    cy = prism.gyy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gyz():
    "gravmag.prism.gyz python vs cython implementation"
    py = _prism_numpy.gyz(xp, yp, zp, model)
    cy = prism.gyz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_gzz():
    "gravmag.prism.gzz python vs cython implementation"
    py = _prism_numpy.gzz(xp, yp, zp, model)
    cy = prism.gzz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))

def test_tf():
    "gravmag.prism.tf python vs cython implementation"
    py = _prism_numpy.tf(xp, yp, zp, model, inc, dec)
    cy = prism.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))
