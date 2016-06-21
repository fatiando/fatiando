import numpy as np

from fatiando.mesher import PolygonalPrism, Prism
from fatiando.gravmag import polyprism, prism, _polyprism_numpy
from fatiando import utils

model = None
prismmodel = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10e-9
precision_mag = 10e-5


def setup():
    global model, xp, yp, zp, inc, dec, prismmodel
    inc, dec = -30, 50
    props1 = {'density': 2., 'magnetization': utils.ang2vec(-20, 25, -10)}
    props2 = {'density': -3., 'magnetization': utils.ang2vec(10, inc, dec)}
    model = [PolygonalPrism([
        [100, -100],
        [100, 100],
        [-100, 100],
        [-100, -100]], 100, 300, props1),
        PolygonalPrism([
            [400, -100],
            [600, -100],
            [600, 100],
            [400, 100]], 100, 300, props2)]
    prismmodel = [Prism(-100, 100, -100, 100, 100, 300, props1),
                  Prism(400, 600, -100, 100, 100, 300, props2)]
    tmp = np.linspace(-500, 1000, 50)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1 * np.ones_like(xp)


def test_gz():
    "polyprism.gz against prism"
    resprism = prism.gz(xp, yp, zp, prismmodel)
    respoly = polyprism.gz(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    errormsg = 'max diff: %g | max polyprism: %g | max prism: %g' % (
        max(diff), max(respoly), max(resprism))
    assert np.all(diff <= precision), errormsg


def test_gxx():
    "polyprism.gxx against prism"
    resprism = prism.gxx(xp, yp, zp, prismmodel)
    respoly = polyprism.gxx(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gxy():
    "polyprism.gxy against prism"
    resprism = prism.gxy(xp, yp, zp, prismmodel)
    respoly = polyprism.gxy(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gxz():
    "polyprism.gxx against prism"
    resprism = prism.gxz(xp, yp, zp, prismmodel)
    respoly = polyprism.gxz(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gyy():
    "polyprism.gyy against prism"
    resprism = prism.gyy(xp, yp, zp, prismmodel)
    respoly = polyprism.gyy(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gyz():
    "polyprism.gyz against prism"
    resprism = prism.gyz(xp, yp, zp, prismmodel)
    respoly = polyprism.gyz(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_gzz():
    "polyprism.gzz against prism"
    resprism = prism.gzz(xp, yp, zp, prismmodel)
    respoly = polyprism.gzz(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision), 'max diff: %g' % (max(diff))


def test_tf():
    "polyprism.tf against prism"
    resprism = prism.tf(xp, yp, zp, prismmodel, inc, dec)
    respoly = polyprism.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(resprism - respoly)
    errormsg = 'max diff: %g | max polyprism: %g | max prism: %g' % (
        max(diff), max(respoly), max(resprism))
    assert np.all(diff <= precision_mag), errormsg


def test_bx():
    "polyprism.bx against prism"
    resprism = prism.bx(xp, yp, zp, prismmodel)
    respoly = polyprism.bx(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision_mag), 'max diff: %g' % (max(diff))


def test_by():
    "polyprism.by against prism"
    resprism = prism.by(xp, yp, zp, prismmodel)
    respoly = polyprism.by(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision_mag), 'max diff: %g' % (max(diff))


def test_bz():
    "polyprism.bz against prism"
    resprism = prism.bz(xp, yp, zp, prismmodel)
    respoly = polyprism.bz(xp, yp, zp, model)
    diff = np.abs(resprism - respoly)
    assert np.all(diff <= precision_mag), 'max diff: %g' % (max(diff))


def test_kernelxx():
    "polyprism.kernelxx against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelxx(xp, yp, zp, pris)
        respoly = polyprism.kernelxx(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))


def test_kernelxy():
    "polyprism.kernelxy against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelxy(xp, yp, zp, pris)
        respoly = polyprism.kernelxy(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))


def test_kernelxz():
    "polyprism.kernelxz against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelxz(xp, yp, zp, pris)
        respoly = polyprism.kernelxz(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))


def test_kernelyy():
    "polyprism.kernelyy against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelyy(xp, yp, zp, pris)
        respoly = polyprism.kernelyy(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))


def test_kernelyz():
    "polyprism.kernelyz against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelyz(xp, yp, zp, pris)
        respoly = polyprism.kernelyz(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))


def test_kernelzz():
    "polyprism.kernelzz against prism"
    for pris, poly in zip(prismmodel, model):
        resprism = prism.kernelzz(xp, yp, zp, pris)
        respoly = polyprism.kernelzz(xp, yp, zp, poly)
        diff = np.abs(resprism - respoly)
        assert np.all(diff <= precision), \
            'max diff: %g' % (max(diff))
