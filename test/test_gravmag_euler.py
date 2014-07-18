from __future__ import division
import numpy as np
from fatiando.gravmag.euler import Classic, ExpandingWindow, MovingWindow
from fatiando.gravmag import sphere, fourier
from fatiando.mesher import Sphere
from fatiando import utils, gridder

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
struct_ind = None
base = None
pos = None
field, xderiv, yderiv, zderiv = None, None, None, None
precision = 0.01


def setup():
    global model, x, y, z, inc, dec, struct_ind, field, xderiv, yderiv, \
        zderiv, base, pos
    inc, dec = -30, 50
    pos = np.array([1000, 1000, 200])
    model = Sphere(pos[0], pos[1], pos[2], 1,
                   {'magnetization': utils.ang2vec(10000, inc, dec)})
    struct_ind = 3
    shape = (128, 128)
    x, y, z = gridder.regular((0, 3000, 0, 3000), shape, z=-1)
    base = 10
    field = utils.nt2si(sphere.tf(x, y, z, [model], inc, dec)) + base
    xderiv = fourier.derivx(x, y, field, shape)
    yderiv = fourier.derivy(x, y, field, shape)
    zderiv = fourier.derivz(x, y, field, shape)


def test_euler_classic_sphere_mag():
    "gravmag.euler.Classic for sphere model and magnetic data"
    euler = Classic(x, y, z, field, xderiv, yderiv, zderiv, struct_ind).fit()
    assert (base - euler.baselevel_) / base <= precision, \
        'baselevel: %g estimated: %g' % (base, euler.baselevel_)
    assert np.all((pos - euler.estimate_) / pos <= precision), \
        'position: %s estimated: %s' % (str(pos), str(euler.estimate_))


def test_euler_classic_expandingwindow_sphere_mag():
    "gravmag.euler.ExpandingWindow w Classic for sphere model + magnetic data"
    euler = ExpandingWindow(
        Classic(x, y, z, field, xderiv, yderiv, zderiv, struct_ind),
        center=[1000, 1000], sizes=np.linspace(100, 2000, 20)).fit()
    assert (base - euler.baselevel_) / base <= precision, \
        'baselevel: %g estimated: %g' % (base, euler.baselevel_)
    assert np.all((pos - euler.estimate_) / pos <= precision), \
        'position: %s estimated: %s' % (str(pos), str(euler.estimate_))


def test_euler_classic_movingwindow_sphere_mag():
    "gravmag.euler.MovingWindow w Classic for sphere model + magnetic data"
    euler = MovingWindow(
        Classic(x, y, z, field, xderiv, yderiv, zderiv, struct_ind),
        windows=[10, 10], size=(1000, 1000), keep=0.2).fit()
    for b in euler.baselevel_:
        assert (base - b) / base <= precision, \
            'baselevel: %g estimated: %g' % (base, b)
    for c in euler.estimate_:
        assert np.all((pos - c) / pos <= precision), \
            'position: %s estimated: %s' % (str(pos), str(c))
