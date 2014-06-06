import numpy as np

from fatiando.mesher import PolygonalPrism, Prism
from fatiando import gravmag, utils

model = None
prismmodel = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10**(-6)

def setup():
    global model, xp, yp, zp, inc, dec, prismmodel
    inc, dec = -30, 50
    props1 = {'density':2., 'magnetization':utils.ang2vec(-20, 25, -10)}
    props2 = {'density':-3., 'magnetization':utils.ang2vec(10, inc, dec)}
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
    zp = -1*np.ones_like(xp)

#def test_potential():
    #"gravmag.polyprism.potential against gravmag.prism"
    #prism = gravmag.prism.potential(xp, yp, zp, prismmodel)
    #polyprism = gravmag.polyprism.potential(xp, yp, zp, model)
    #diff = np.abs(prism - polyprism)
    #assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

#def test_gx():
    #"gravmag.polyprism.gx against gravmag.prism"
    #prism = gravmag.prism.gx(xp, yp, zp, prismmodel)
    #polyprism = gravmag.polyprism.gx(xp, yp, zp, model)
    #diff = np.abs(prism - polyprism)
    #assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

#def test_gy():
    #"gravmag.polyprism.gy against gravmag.prism"
    #prism = gravmag.prism.gy(xp, yp, zp, prismmodel)
    #polyprism = gravmag.polyprism.gy(xp, yp, zp, model)
    #diff = np.abs(prism - polyprism)
    #assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gz():
    "gravmag.polyprism.gz against gravmag.prism"
    prism = gravmag.prism.gz(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gz(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    errormsg = 'max diff: %g | max polyprism: %g | max prism: %g' % (
        max(diff), max(polyprism), max(prism))
    assert np.all(diff <= max(prism)*precision), errormsg

def test_gxx():
    "gravmag.polyprism.gxx against gravmag.prism"
    prism = gravmag.prism.gxx(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gxx(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gxy():
    "gravmag.polyprism.gxy against gravmag.prism"
    prism = gravmag.prism.gxy(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gxy(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gxz():
    "gravmag.polyprism.gxx against gravmag.prism"
    prism = gravmag.prism.gxz(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gxz(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gyy():
    "gravmag.polyprism.gyy against gravmag.prism"
    prism = gravmag.prism.gyy(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gyy(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gyz():
    "gravmag.polyprism.gyz against gravmag.prism"
    prism = gravmag.prism.gyz(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gyz(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_gzz():
    "gravmag.polyprism.gzz against gravmag.prism"
    prism = gravmag.prism.gzz(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.gzz(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_tf():
    "gravmag.polyprism.tf against gravmag.prism"
    prism = gravmag.prism.tf(xp, yp, zp, prismmodel, inc, dec)
    polyprism = gravmag.polyprism.tf(xp, yp, zp, model, inc, dec)
    diff = np.abs(prism - polyprism)
    errormsg = 'max diff: %g | max polyprism: %g | max prism: %g' % (
        max(diff), max(polyprism), max(prism))
    assert np.all(diff <= max(prism)*precision), errormsg

def test_bx():
    "gravmag.polyprism.bx against gravmag.prism"
    prism = gravmag.prism.bx(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.bx(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_by():
    "gravmag.polyprism.by against gravmag.prism"
    prism = gravmag.prism.by(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.by(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_bz():
    "gravmag.polyprism.bz against gravmag.prism"
    prism = gravmag.prism.bz(xp, yp, zp, prismmodel)
    polyprism = gravmag.polyprism.bz(xp, yp, zp, model)
    diff = np.abs(prism - polyprism)
    assert np.all(diff <= max(prism)*precision), 'max diff: %g' % (max(diff))

def test_kernelxx():
    "gravmag.polyprism.kernelxx against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelxx(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelxx(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

def test_kernelxy():
    "gravmag.polyprism.kernelxy against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelxy(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelxy(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

def test_kernelxz():
    "gravmag.polyprism.kernelxz against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelxz(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelxz(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

def test_kernelyy():
    "gravmag.polyprism.kernelyy against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelyy(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelyy(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

def test_kernelyz():
    "gravmag.polyprism.kernelyz against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelyz(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelyz(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

def test_kernelzz():
    "gravmag.polyprism.kernelzz against gravmag.prism"
    for pris, poly in zip(prismmodel, model):
        prism = gravmag.prism.kernelzz(xp, yp, zp, pris)
        polyprism = gravmag.polyprism.kernelzz(xp, yp, zp, poly)
        diff = np.abs(prism - polyprism)
        assert np.all(diff <= max(prism)*precision), \
            'max diff: %g' % (max(diff))

