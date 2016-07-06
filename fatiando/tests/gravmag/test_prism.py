import numpy as np
import numpy.testing as npt
from pytest import raises

from fatiando.mesher import Prism
from fatiando.gravmag.forward import prism
from fatiando import utils, gridder


# TODO:
#   * test around consistency for any sizes of prism


def test_potential_around():
    'gravmag.prism potential is the same all around the prism'
    # Prism must be a cube for this to work
    model = Prism(-6000, -2000, 1000, 5000, -1000, 3000, {'density': 1000})
    n = 100
    xc, yc, zc = model.center()
    x = xc*np.ones(n)
    y = yc*np.ones(n)
    z = zc*np.ones(n)
    dist = np.linspace(1, 5000, n)
    sides = [
        prism.potential(model.x1 - dist, y, z, model),
        prism.potential(model.x2 + dist, y, z, model),
        prism.potential(x, model.y1 - dist, z, model),
        prism.potential(x, model.y2 + dist, z, model),
        prism.potential(x, y, model.z1 - dist, model),
        prism.potential(x, y, model.z2 + dist, model),
        ]
    for s1 in sides:
        for s2 in sides:
            npt.assert_allclose(s1, s2)


def test_g_around():
    'gravmag.prism amplitude of g is the same all around the prism'
    # Prism must be a cube for this to work
    model = Prism(-6000, -2000, 1000, 5000, -1000, 3000, {'density': 1000})
    n = 100
    xc, yc, zc = model.center()
    x = xc*np.ones(n)
    y = yc*np.ones(n)
    z = zc*np.ones(n)
    dist = np.linspace(1, 5000, n)
    def amp(x, y, z, model):
        gx = prism.gx(x, y, z, model)
        gy = prism.gy(x, y, z, model)
        gz = prism.gz(x, y, z, model)
        return np.sqrt(gx**2 + gy**2 + gz**2)
    sides = [
        amp(model.x1 - dist, y, z, model),
        amp(model.x2 + dist, y, z, model),
        amp(x, model.y1 - dist, z, model),
        amp(x, model.y2 + dist, z, model),
        amp(x, y, model.z1 - dist, model),
        amp(x, y, model.z2 + dist, model),
        ]
    for s1 in sides:
        for s2 in sides:
            npt.assert_allclose(s1, s2)


def test_fails_if_shape_mismatch():
    'gravmag.prism fails if given computation points with different shapes'
    inc, dec = 10, 0
    model = Prism(-6000, -2000, 2000, 4000, 0, 3000,
                  {'density': 1000,
                   'magnetization': utils.ang2vec(10, inc, dec)})
    area = [-5000, 5000, -10000, 10000]
    x, y, z = gridder.regular(area, (101, 51), z=-1)

    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz',
             'bx', 'by', 'bz',
             'kernelxx', 'kernelxy', 'kernelxz', 'kernelyy', 'kernelyz',
             'kernelzz', 'tf']
    for f in funcs:
        if f == 'tf':
            kwargs = dict(inc=inc, dec=dec)
        else:
            kwargs = {}
        func = getattr(prism, f)
        raises(AssertionError, func, x[:-2], y, z, model, **kwargs)
        raises(AssertionError, func, x, y[:-2], z, model, **kwargs)
        raises(AssertionError, func, x, y, z[:-2], model, **kwargs)
        raises(AssertionError, func, x[:-5], y, z[:-2], model, **kwargs)


def test_force_physical_property():
    'gravmag.prism gives correct results when passed a property value as arg'
    inc, dec = 10, 0
    model = Prism(-6000, -2000, 2000, 4000, 0, 3000,
                  {'density': 1000,
                   'magnetization': utils.ang2vec(10, inc, dec)})
    density = -500
    mag = utils.ang2vec(-5, -30, 15)
    reference = Prism(-6000, -2000, 2000, 4000, 0, 3000,
                      {'density': density, 'magnetization': mag})
    area = [-10000, 10000, -5000, 5000]
    x, y, z = gridder.regular(area, (51, 101), z=-1)
    # Test gravity functions
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
    for f in funcs:
        forced = getattr(prism, f)(x, y, z, model, dens=density)
        ref = getattr(prism, f)(x, y, z, reference)
        npt.assert_allclose(forced, ref)
    # Test magnetic functions
    funcs = ['tf', 'bx', 'by', 'bz']
    for f in funcs:
        if f == 'tf':
            forced = getattr(prism, f)(x, y, z, model, inc, dec, pmag=mag)
            ref = getattr(prism, f)(x, y, z, reference, inc, dec)
        else:
            forced = getattr(prism, f)(x, y, z, model, pmag=mag)
            ref = getattr(prism, f)(x, y, z, reference)
        npt.assert_allclose(forced, ref)


def test_around():
    "gravmag.prism gravitational results are consistent around the prism"
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
    model = Prism(-300, 300, -300, 300, -300, 300, {'density': 1000})
    # Make the computation points surround the prism
    shape = (101, 101)
    area = [-600, 600, -600, 600]
    distance = 310
    grids = [gridder.regular(area, shape, z=-distance),
             gridder.regular(area, shape, z=distance),
             gridder.regular(area, shape, z=distance)[::-1],
             gridder.regular(area, shape, z=-distance)[::-1],
             np.array(gridder.regular(area, shape, z=distance))[[0, 2, 1]],
             np.array(gridder.regular(area, shape, z=-distance))[[0, 2, 1]]]
    xp, yp, zp = grids[0]

    # Test if each component is consistent
    # POTENTIAL
    face = [prism.potential(x, y, z, model) for x, y, z in grids]
    for i in range(6):
        for j in range(i + 1, 6):
            npt.assert_almost_equal(face[i], face[j])

    # GX
    top, bottom, north, south, east, west = [prism.gx(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, bottom)
    npt.assert_almost_equal(north, -south)
    npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(east, top)
    npt.assert_almost_equal(north, -prism.gz(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gz(xp, yp, zp, model))

    # GY
    top, bottom, north, south, east, west = [prism.gy(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, bottom)
    npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, -west)
    npt.assert_almost_equal(north, top)
    npt.assert_almost_equal(east, -prism.gz(xp, yp, zp, model))
    npt.assert_almost_equal(west, prism.gz(xp, yp, zp, model))

    # GZ
    top, bottom, north, south, east, west = [prism.gz(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, -bottom)
    npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(north, prism.gx(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gx(xp, yp, zp, model))
    npt.assert_almost_equal(east, prism.gy(xp, yp, zp, model))
    npt.assert_almost_equal(west, prism.gy(xp, yp, zp, model))

    # GXX
    top, bottom, north, south, east, west = [prism.gxx(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, bottom)
    npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(east, top)
    npt.assert_almost_equal(north, prism.gzz(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gzz(xp, yp, zp, model))

    # GXY
    top, bottom, north, south, east, west = [prism.gxy(x, y, z, model)
                                             for x, y, z in grids]
    # npt.assert_almost_equal(bottom, top, decimal=1)
    npt.assert_almost_equal(north, -south)
    npt.assert_almost_equal(east, -west)
    npt.assert_almost_equal(north, -prism.gyz(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gyz(xp, yp, zp, model))

    # GXZ
    top, bottom, north, south, east, west = [prism.gxz(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, -bottom)
    npt.assert_almost_equal(north, -south)
    # npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(bottom, north)
    npt.assert_almost_equal(top, south)
    # npt.assert_almost_equal(east, prism.gxy(xp, yp, zp, model))
    npt.assert_almost_equal(west, prism.gxy(xp, yp, zp, model))

    # GYY
    top, bottom, north, south, east, west = [prism.gyy(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, bottom)
    npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(top, north)
    npt.assert_almost_equal(east, prism.gzz(xp, yp, zp, model))
    npt.assert_almost_equal(west, prism.gzz(xp, yp, zp, model))

    # GYZ
    top, bottom, north, south, east, west = [prism.gyz(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, -bottom)
    # npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, -west)
    npt.assert_almost_equal(top, west)
    npt.assert_almost_equal(bottom, east)
    # npt.assert_almost_equal(north, prism.gxy(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gxy(xp, yp, zp, model))

    # GZZ
    top, bottom, north, south, east, west = [prism.gzz(x, y, z, model)
                                             for x, y, z in grids]
    npt.assert_almost_equal(top, bottom)
    npt.assert_almost_equal(north, south)
    npt.assert_almost_equal(east, west)
    npt.assert_almost_equal(north, prism.gxx(xp, yp, zp, model))
    npt.assert_almost_equal(south, prism.gxx(xp, yp, zp, model))
    npt.assert_almost_equal(east, prism.gyy(xp, yp, zp, model))
    npt.assert_almost_equal(west, prism.gyy(xp, yp, zp, model))
