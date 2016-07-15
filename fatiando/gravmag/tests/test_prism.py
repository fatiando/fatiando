import numpy as np
from numpy.testing import assert_array_almost_equal as assert_almost
from pytest import raises

from fatiando.mesher import Prism
from fatiando.gravmag import _prism_numpy, prism
from fatiando import utils, gridder


def test_fails_if_shape_mismatch():
    'gravmag.prism fails if given computation points with different shapes'
    inc, dec = 10, 0
    model = [Prism(-6000, -2000, 2000, 4000, 0, 3000,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(10, inc, dec)})]
    area = [-5000, 5000, -10000, 10000]
    x, y, z = gridder.regular(area, (101, 51), z=-1)

    raises(ValueError, prism.potential, x[:-2], y, z, model)
    raises(ValueError, prism.potential, x, y[:-2], z, model)
    raises(ValueError, prism.potential, x, y, z[:-2], model)
    raises(ValueError, prism.potential, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gx, x[:-2], y, z, model)
    raises(ValueError, prism.gx, x, y[:-2], z, model)
    raises(ValueError, prism.gx, x, y, z[:-2], model)
    raises(ValueError, prism.gx, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gy, x[:-2], y, z, model)
    raises(ValueError, prism.gy, x, y[:-2], z, model)
    raises(ValueError, prism.gy, x, y, z[:-2], model)
    raises(ValueError, prism.gy, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gz, x[:-2], y, z, model)
    raises(ValueError, prism.gz, x, y[:-2], z, model)
    raises(ValueError, prism.gz, x, y, z[:-2], model)
    raises(ValueError, prism.gz, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gxx, x[:-2], y, z, model)
    raises(ValueError, prism.gxx, x, y[:-2], z, model)
    raises(ValueError, prism.gxx, x, y, z[:-2], model)
    raises(ValueError, prism.gxx, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gxy, x[:-2], y, z, model)
    raises(ValueError, prism.gxy, x, y[:-2], z, model)
    raises(ValueError, prism.gxy, x, y, z[:-2], model)
    raises(ValueError, prism.gxy, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gxz, x[:-2], y, z, model)
    raises(ValueError, prism.gxz, x, y[:-2], z, model)
    raises(ValueError, prism.gxz, x, y, z[:-2], model)
    raises(ValueError, prism.gxz, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gyy, x[:-2], y, z, model)
    raises(ValueError, prism.gyy, x, y[:-2], z, model)
    raises(ValueError, prism.gyy, x, y, z[:-2], model)
    raises(ValueError, prism.gyy, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gyz, x[:-2], y, z, model)
    raises(ValueError, prism.gyz, x, y[:-2], z, model)
    raises(ValueError, prism.gyz, x, y, z[:-2], model)
    raises(ValueError, prism.gyz, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.gzz, x[:-2], y, z, model)
    raises(ValueError, prism.gzz, x, y[:-2], z, model)
    raises(ValueError, prism.gzz, x, y, z[:-2], model)
    raises(ValueError, prism.gzz, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.bx, x[:-2], y, z, model)
    raises(ValueError, prism.bx, x, y[:-2], z, model)
    raises(ValueError, prism.bx, x, y, z[:-2], model)
    raises(ValueError, prism.bx, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.by, x[:-2], y, z, model)
    raises(ValueError, prism.by, x, y[:-2], z, model)
    raises(ValueError, prism.by, x, y, z[:-2], model)
    raises(ValueError, prism.by, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.bz, x[:-2], y, z, model)
    raises(ValueError, prism.bz, x, y[:-2], z, model)
    raises(ValueError, prism.bz, x, y, z[:-2], model)
    raises(ValueError, prism.bz, x[:-5], y, z[:-2], model)

    raises(ValueError, prism.tf, x[:-2], y, z, model, inc, dec)
    raises(ValueError, prism.tf, x, y[:-2], z, model, inc, dec)
    raises(ValueError, prism.tf, x, y, z[:-2], model, inc, dec)
    raises(ValueError, prism.tf, x[:-5], y, z[:-2], model, inc, dec)

    raises(ValueError, prism.kernelxx, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelxx, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelxx, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelxx, x[:-5], y, z[:-2], model[0])

    raises(ValueError, prism.kernelxy, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelxy, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelxy, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelxy, x[:-5], y, z[:-2], model[0])

    raises(ValueError, prism.kernelxz, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelxz, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelxz, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelxz, x[:-5], y, z[:-2], model[0])

    raises(ValueError, prism.kernelyy, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelyy, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelyy, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelyy, x[:-5], y, z[:-2], model[0])

    raises(ValueError, prism.kernelyz, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelyz, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelyz, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelyz, x[:-5], y, z[:-2], model[0])

    raises(ValueError, prism.kernelzz, x[:-2], y, z, model[0])
    raises(ValueError, prism.kernelzz, x, y[:-2], z, model[0])
    raises(ValueError, prism.kernelzz, x, y, z[:-2], model[0])
    raises(ValueError, prism.kernelzz, x[:-5], y, z[:-2], model[0])


def test_force_physical_property():
    'gravmag.prism gives correct results when passed a property value as arg'
    inc, dec = 10, 0
    model = [Prism(-6000, -2000, 2000, 4000, 0, 3000,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(10, inc, dec)}),
             Prism(2000, 6000, 2000, 4000, 0, 1000,
                   {'density': -1000,
                    'magnetization': utils.ang2vec(15, inc, dec)})]
    density = -500
    mag = utils.ang2vec(-5, -30, 15)
    reference = [
        Prism(-6000, -2000, 2000, 4000, 0, 3000,
              {'density': density, 'magnetization': mag}),
        Prism(2000, 6000, 2000, 4000, 0, 1000,
              {'density': density, 'magnetization': mag})]
    area = [-10000, 10000, -5000, 5000]
    x, y, z = gridder.regular(area, (51, 101), z=-1)
    for mod in [prism, _prism_numpy]:
        # Test gravity functions
        funcs = ['potential', 'gx', 'gy', 'gz',
                 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
        for f in funcs:
            forced = getattr(mod, f)(x, y, z, model, dens=density)
            ref = getattr(mod, f)(x, y, z, reference)
            precision = 10
            assert_almost(forced, ref, precision, 'Field = %s' % (f))
        # Test magnetic functions
        funcs = ['tf', 'bx', 'by', 'bz']
        for f in funcs:
            if f == 'tf':
                forced = getattr(mod, f)(x, y, z, model, inc, dec, pmag=mag)
                ref = getattr(mod, f)(x, y, z, reference, inc, dec)
            else:
                forced = getattr(mod, f)(x, y, z, model, pmag=mag)
                ref = getattr(mod, f)(x, y, z, reference)
            precision = 10
            assert_almost(forced, ref, precision, 'Field = %s' % (f))


def test_ignore_none_and_missing_properties():
    'gravmag.prism ignores None and prisms without the required property'
    inc, dec = 50, -30
    model = [None,
             Prism(-6000, -2000, 2000, 4000, 0, 3000,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(10, inc, dec)}),
             Prism(2000, 6000, 2000, 4000, 0, 1000,
                   {'magnetization': utils.ang2vec(15, inc, dec)}),
             None,
             Prism(-6000, -2000, -4000, -2000, 500, 2000,
                   {'density': -1000})]
    area = [-10000, 10000, -5000, 5000]
    x, y, z = gridder.regular(area, (101, 51), z=-1)
    for mod in [prism, _prism_numpy]:
        # Test gravity functions
        funcs = ['potential', 'gx', 'gy', 'gz',
                 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
        for f in funcs:
            combined = getattr(mod, f)(x, y, z, model)
            separate = getattr(mod, f)(x, y, z, [model[1], model[4]])
            precision = 10
            assert_almost(separate, combined, precision, 'Field = %s' % (f))
        # Test magnetic functions
        funcs = ['tf', 'bx', 'by', 'bz']
        for f in funcs:
            mag_only = [model[1], model[2]]
            if f == 'tf':
                combined = getattr(mod, f)(x, y, z, model, inc, dec)
                separate = getattr(mod, f)(x, y, z, mag_only, inc, dec)
            else:
                combined = getattr(mod, f)(x, y, z, model)
                separate = getattr(mod, f)(x, y, z, mag_only)
            precision = 10
            assert_almost(separate, combined, precision, 'Field = %s' % (f))


def test_cython_agains_numpy():
    "gravmag.prism numpy and cython implementations give same result"
    inc, dec = -30, 50
    model = [
        Prism(100, 300, -100, 100, 0, 400,
              {'density': -1000,
               'magnetization': utils.ang2vec(-2, inc, dec)}),
        Prism(-300, -100, -100, 100, 0, 200,
              {'density': 2000, 'magnetization': utils.ang2vec(5, 25, -10)})]
    tmp = np.linspace(-500, 500, 101)
    xp, yp = [i.ravel() for i in np.meshgrid(tmp, tmp)]
    zp = -1 * np.ones_like(xp)
    kernels = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    for comp in kernels:
        for p in model:
            py = getattr(_prism_numpy, 'kernel' + comp)(xp, yp, zp, p)
            cy = getattr(prism, 'kernel' + comp)(xp, yp, zp, p)
            assert_almost(py, cy, 10,
                          'Kernel = %s, max field %.15g max diff %.15g'
                          % (comp, np.abs(cy).max(), np.abs(py - cy).max()))
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
        if f in ['bx', 'by', 'bz', 'tf']:
            precision = 8
        else:
            precision = 10
        assert_almost(py, cy, precision,
                      'Field = %s, max field %.15g max diff %.15g'
                      % (f, np.abs(cy).max(), np.abs(py - cy).max()))


def test_around():
    "gravmag.prism gravitational results are consistent around the prism"
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
    model = [Prism(-300, 300, -300, 300, -300, 300, {'density': 1000})]
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
            assert_almost(face[i], face[j], 10,
                          'Failed potential, faces %d and %d' % (i, j))
    # GX
    top, bottom, north, south, east, west = [prism.gx(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 10, 'Failed gx, top and bottom')
    assert_almost(north, -south, 10, 'Failed gx, north and south')
    assert_almost(east, west, 10, 'Failed gx, east and west')
    assert_almost(east, top, 10, 'Failed gx, east and top')
    assert_almost(north, -prism.gz(xp, yp, zp, model), 10,
                  'Failed gx, north and gz')
    assert_almost(south, prism.gz(xp, yp, zp, model), 10,
                  'Failed gx, south and gz')
    # GY
    top, bottom, north, south, east, west = [prism.gy(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 10, 'Failed gy, top and bottom')
    assert_almost(north, south, 10, 'Failed gy, north and south')
    assert_almost(east, -west, 10, 'Failed gy, east and west')
    assert_almost(north, top, 10, 'Failed gy, north and top')
    assert_almost(east, -prism.gz(xp, yp, zp, model), 10,
                  'Failed gy, east and gz')
    assert_almost(west, prism.gz(xp, yp, zp, model), 10,
                  'Failed gy, west and gz')
    # GZ
    top, bottom, north, south, east, west = [prism.gz(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, -bottom, 10, 'Failed gz, top and bottom')
    assert_almost(north, south, 10, 'Failed gz, north and south')
    assert_almost(east, west, 10, 'Failed gz, east and west')
    assert_almost(north, prism.gx(xp, yp, zp, model), 10,
                  'Failed gz, north and gx')
    assert_almost(south, prism.gx(xp, yp, zp, model), 10,
                  'Failed gz, south and gx')
    assert_almost(east, prism.gy(xp, yp, zp, model), 10,
                  'Failed gz, east and gy')
    assert_almost(west, prism.gy(xp, yp, zp, model), 10,
                  'Failed gz, west and gy')
    # GXX
    top, bottom, north, south, east, west = [prism.gxx(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 10, 'Failed gxx, top and bottom')
    assert_almost(north, south, 10, 'Failed gxx, north and south')
    assert_almost(east, west, 10, 'Failed gxx, east and west')
    assert_almost(east, top, 10, 'Failed gxx, east and top')
    assert_almost(north, prism.gzz(xp, yp, zp, model), 10,
                  'Failed gxx, north and gzz')
    assert_almost(south, prism.gzz(xp, yp, zp, model), 10,
                  'Failed gxx, south and gzz')
    # GXY
    top, bottom, north, south, east, west = [prism.gxy(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 4, 'Failed gxy, top and bottom')
    assert_almost(north, -south, 10, 'Failed gxy, north and south')
    assert_almost(east, -west, 10, 'Failed gxy, east and west')
    assert_almost(north, -prism.gyz(xp, yp, zp, model), 10,
                  'Failed gxy, north and gyz')
    assert_almost(south, prism.gyz(xp, yp, zp, model), 10,
                  'Failed gxy, south and gyz')
    # GXZ
    top, bottom, north, south, east, west = [prism.gxz(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, -bottom, 10, 'Failed gxz, top and bottom')
    assert_almost(north, -south, 10, 'Failed gxz, north and south')
    assert_almost(east, west, 4, 'Failed gxz, east and west')
    assert_almost(bottom, north, 10, 'Failed gxz, bottom and north')
    assert_almost(top, south, 10, 'Failed gxz, top and south')
    assert_almost(east, prism.gxy(xp, yp, zp, model), 4,
                  'Failed gxz, east and gxy')
    assert_almost(west, prism.gxy(xp, yp, zp, model), 10,
                  'Failed gxz, west and gxy')
    # GYY
    top, bottom, north, south, east, west = [prism.gyy(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 10, 'Failed gyy, top and bottom')
    assert_almost(north, south, 10, 'Failed gyy, north and south')
    assert_almost(east, west, 10, 'Failed gyy, east and west')
    assert_almost(top, north, 10, 'Failed gyy, top and north')
    assert_almost(east, prism.gzz(xp, yp, zp, model), 10,
                  'Failed gyy, east and gzz')
    assert_almost(west, prism.gzz(xp, yp, zp, model), 10,
                  'Failed gyy, west and gzz')
    # GYZ
    top, bottom, north, south, east, west = [prism.gyz(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, -bottom, 10, 'Failed gyz, top and bottom')
    assert_almost(north, south, 4, 'Failed gyz, north and south')
    assert_almost(east, -west, 10, 'Failed gyz, east and west')
    assert_almost(top, west, 10, 'Failed gyz, top and west')
    assert_almost(bottom, east, 10, 'Failed gyz, bottom and east')
    assert_almost(north, prism.gxy(xp, yp, zp, model), 4,
                  'Failed gyz, north and gxy')
    assert_almost(south, prism.gxy(xp, yp, zp, model), 10,
                  'Failed gyz, south and gxy')
    # GZZ
    top, bottom, north, south, east, west = [prism.gzz(x, y, z, model)
                                             for x, y, z in grids]
    assert_almost(top, bottom, 10, 'Failed gzz, top and bottom')
    assert_almost(north, south, 10, 'Failed gzz, north and south')
    assert_almost(east, west, 10, 'Failed gzz, east and west')
    assert_almost(north, prism.gxx(xp, yp, zp, model), 10,
                  'Failed gzz, north and gxx')
    assert_almost(south, prism.gxx(xp, yp, zp, model), 10,
                  'Failed gzz, south and gxx')
    assert_almost(east, prism.gyy(xp, yp, zp, model), 10,
                  'Failed gzz, east and gyy')
    assert_almost(west, prism.gyy(xp, yp, zp, model), 10,
                  'Failed gzz, west and gyy')
