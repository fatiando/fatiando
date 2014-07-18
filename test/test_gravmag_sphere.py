import numpy as np

from fatiando.mesher import Sphere
from fatiando.gravmag import _sphere_numpy, sphere
from fatiando import utils, gridder

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10 ** (-15)
lower_precision = 10 ** (-12)


def setup():
    global model, xp, yp, zp, inc, dec
    inc, dec = -30, 50
    reg_field = np.array(utils.dircos(inc, dec))
    model = [
        Sphere(500, 0, 1000, 1000,
               {'density': -1., 'magnetization': utils.ang2vec(-2, inc, dec)}),
        Sphere(-1000, 0, 700, 700,
               {'density': 2., 'magnetization': utils.ang2vec(5, 25, -10)})]
    xp, yp, zp = gridder.regular([-2000, 2000, -2000, 2000], (50, 50), z=-1)


def test_gz():
    "gravmag.sphere.gz python vs cython implementation"
    py = _sphere_numpy.gz(xp, yp, zp, model)
    cy = sphere.gz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gxx():
    "gravmag.sphere.gxx python vs cython implementation"
    py = _sphere_numpy.gxx(xp, yp, zp, model)
    cy = sphere.gxx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gxy():
    "gravmag.sphere.gxy python vs cython implementation"
    py = _sphere_numpy.gxy(xp, yp, zp, model)
    cy = sphere.gxy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gxz():
    "gravmag.sphere.gxx python vs cython implementation"
    py = _sphere_numpy.gxz(xp, yp, zp, model)
    cy = sphere.gxz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gyy():
    "gravmag.sphere.gyy python vs cython implementation"
    py = _sphere_numpy.gyy(xp, yp, zp, model)
    cy = sphere.gyy(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gyz():
    "gravmag.sphere.gyz python vs cython implementation"
    py = _sphere_numpy.gyz(xp, yp, zp, model)
    cy = sphere.gyz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gzz():
    "gravmag.sphere.gzz python vs cython implementation"
    py = _sphere_numpy.gzz(xp, yp, zp, model)
    cy = sphere.gzz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_tf():
    "gravmag.sphere.tf python vs cython implementation"
    py = _sphere_numpy.tf(xp, yp, zp, model, inc, dec)
    cy = sphere.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(py - cy)
    # Lower precison because python calculates using Blakely and cython using
    # the gravity kernels
    assert np.all(diff <= 10 ** -9), 'max diff: %g' % (max(diff))


def test_bx():
    "gravmag.sphere.bx python vs cython implementation"
    py = _sphere_numpy.bx(xp, yp, zp, model)
    cy = sphere.bx(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= lower_precision), \
        'max diff: %g python: %g cython %g' \
        % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_by():
    "gravmag.sphere.by python vs cython implementation"
    py = _sphere_numpy.by(xp, yp, zp, model)
    cy = sphere.by(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= lower_precision), \
        'max diff: %g python: %g cython %g' \
        % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_bz():
    "gravmag.sphere.bz python vs cython implementation"
    py = _sphere_numpy.bz(xp, yp, zp, model)
    cy = sphere.bz(xp, yp, zp, model)
    diff = np.abs(py - cy)
    assert np.all(diff <= lower_precision), \
        'max diff: %g python: %g cython %g' \
        % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelxx():
    "gravmag.sphere.kernelxx python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelxx(xp, yp, zp, p)
        cy = sphere.kernelxx(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelxy():
    "gravmag.sphere.kernelxy python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelxy(xp, yp, zp, p)
        cy = sphere.kernelxy(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelxz():
    "gravmag.sphere.kernelxz python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelxz(xp, yp, zp, p)
        cy = sphere.kernelxz(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelyy():
    "gravmag.sphere.kernelyy python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelyy(xp, yp, zp, p)
        cy = sphere.kernelyy(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelyz():
    "gravmag.sphere.kernelyz python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelyz(xp, yp, zp, p)
        cy = sphere.kernelyz(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])


def test_kernelzz():
    "gravmag.sphere.kernelzz python vs cython implementation"
    for p in model:
        py = _sphere_numpy.kernelzz(xp, yp, zp, p)
        cy = sphere.kernelzz(xp, yp, zp, p)
        diff = np.abs(py - cy)
        assert np.all(diff <= precision), \
            'max diff: %g python: %g cython %g' \
            % (max(diff), py[diff == max(diff)][0], cy[diff == max(diff)][0])
