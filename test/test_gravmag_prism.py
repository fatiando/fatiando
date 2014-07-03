import numpy as np

from fatiando.mesher import Prism
from fatiando.gravmag import _prism_numpy, prism
from fatiando import utils, gridder

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10**(-10)

def setup():
    global model, xp, yp, zp, inc, dec
    inc, dec = -30, 50
    reg_field = np.array(utils.dircos(inc, dec))
    model = [
        Prism(100, 300, -100, 100, 0, 400,
              {'density':1000., 'magnetization':2}),
        Prism(-300, -100, -100, 100, 0, 200,
            {'density':2000, 'magnetization':utils.dircos(25, -10)})]
    tmp = np.linspace(-500, 500, 50)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1*np.ones_like(xp)

def test_potential_around():
    "gravmag.prism.potential is the same around the prism"
    model = [Prism(-3000, 3000, -3000, 3000, -3000, 3000, {'density':1000})]
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], (101, 101),
                              z=5000)
    field = dict(above=prism.potential(x, y, -z, model),
                 below=prism.potential(x, y, z, model),
                 north=prism.potential(z, x, y, model),
                 south=prism.potential(-z, x, y, model),
                 east=prism.potential(x, z, y, model),
                 west=prism.potential(x, -z, y, model))
    done = dict(field)
    for i in field:
        done.pop(i)
        for j in done:
            diff = np.abs(field[i] - done[j])
            assert np.all(diff <= precision), \
                'faces %s and %s max diff: %g' % (i, j, max(diff))

def test_gx_around():
    "gravmag.prism.gx is the coherent around the prism"
    model = [Prism(-3000, 3000, -3000, 3000, -3000, 3000, {'density':1000})]
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], (101, 101),
                              z=5000)
    above = prism.gx(x, y, -z, model)
    below = prism.gx(x, y, z, model)
    north = prism.gx(z, x, y, model)
    south = prism.gx(-z, x, y, model)
    east = prism.gx(x, z, y, model)
    west = prism.gx(x, -z, y, model)
    assert np.allclose(above, below, atol=precision, rtol=0), \
        "different above and below"
    assert np.allclose(east, west, atol=precision, rtol=0), \
            "different east and west"
    assert np.allclose(north, -south, atol=precision, rtol=0), \
        "different north and south"
    gz = prism.gz(x, y, -z, model)
    assert np.allclose(north, -gz, atol=precision, rtol=0), \
        "different north and gz"
    assert np.allclose(south, gz, atol=precision, rtol=0), \
        "different south and gz"

def test_gy_around():
    "gravmag.prism.gy is the coherent around the prism"
    model = [Prism(-3000, 3000, -3000, 3000, -3000, 3000, {'density':1000})]
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], (101, 101),
                              z=5000)
    above = prism.gy(x, y, -z, model)
    below = prism.gy(x, y, z, model)
    north = prism.gy(z, x, y, model)
    south = prism.gy(-z, x, y, model)
    east = prism.gy(x, z, y, model)
    west = prism.gy(x, -z, y, model)
    assert np.allclose(above, below, atol=precision, rtol=0), \
        "different above and below"
    assert np.allclose(east, -west, atol=precision, rtol=0), \
            "different east and west"
    assert np.allclose(north, south, atol=precision, rtol=0), \
        "different north and south"
    gz = prism.gz(x, y, -z, model)
    assert np.allclose(east, -gz, atol=precision, rtol=0), \
        "different east and gz"
    assert np.allclose(west, gz, atol=precision, rtol=0), \
        "different west and gz"

def test_gz_around():
    "gravmag.prism.gz is the coherent around the prism"
    model = [Prism(-3000, 3000, -3000, 3000, -3000, 3000, {'density':1000})]
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], (101, 101),
                              z=5000)
    above = prism.gz(x, y, -z, model)
    below = prism.gz(x, y, z, model)
    north = prism.gz(z, x, y, model)
    south = prism.gz(-z, x, y, model)
    east = prism.gz(x, z, y, model)
    west = prism.gz(x, -z, y, model)
    assert np.allclose(above, -below, atol=precision, rtol=0), \
        "different above and below"
    assert np.allclose(east, west, atol=precision, rtol=0), \
            "different east and west"
    assert np.allclose(north, south, atol=precision, rtol=0), \
        "different north and south"
    assert np.allclose(north, east, atol=precision, rtol=0), \
        "different north and east"
    gy = prism.gy(x, y, -z, model)
    assert np.allclose(north, gy, atol=precision, rtol=0), \
        "different north and gy"
    assert np.allclose(east, gy, atol=precision, rtol=0), \
        "different east and gy"

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
