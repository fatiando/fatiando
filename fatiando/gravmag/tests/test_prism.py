"""
Test the prism forward modeling code.
Verify if functions fail the way they are expected.
Check that field values are consistent around the prism.
Test against saved results to avoid regressions.
"""
from __future__ import absolute_import
import os
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_almost
import numpy.testing as npt
from pytest import raises
import pytest

from ...mesher import Prism
from .. import prism
from ... import utils, gridder, constants
from ...datasets import check_hash


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_DATA_FILE = os.path.join(TEST_DATA_DIR, 'prism.npz')
# Hash obtained using openssl
TEST_DATA_SHA256 = \
    '690accd7fabb665d3dc45ff1cb3fe973a827fe6f3eb15f5f13382bfa379fe532'
FIELDS = 'potential gx gy gz gxx gxy gxz gyy gyz gzz bx by bz tf'.split()
KERNELS = ['kernel' + k for k in 'xx xy xz yy yz zz'.split()]


@pytest.fixture(scope='module')
def data():
    "Load test data for regression tests"
    check_hash(TEST_DATA_FILE, TEST_DATA_SHA256, hash_type='sha256')
    return np.load(TEST_DATA_FILE)


@pytest.fixture(scope='module')
def model():
    "Make the model for the regression tests."
    return make_model()


def make_model():
    """
    Generate the model used to make the saved test data.
    """
    props = {'density': 10, 'magnetization': utils.ang2vec(10, -20, 30)}
    return [Prism(x1=-200, x2=100, y1=500, y2=800, z1=100, z2=600,
                  props=props)]


def generate_test_data():
    """
    Create the test data file for regression tests.
    This function is not used when testing!
    It is here to document how the data file was generated.
    """
    model = make_model()
    # The geomagnetic field direction
    inc, dec = 30, -20
    shape = (97, 121)
    area = [-500, 500, 0, 1200]
    x, y, z = gridder.regular(area, shape, z=-10)
    data = dict(x=x, y=y, z=z, inc=inc, dec=dec, shape=shape)
    for field in FIELDS:
        if field == 'tf':
            data[field] = getattr(prism, field)(x, y, z, model, inc, dec)
        else:
            data[field] = getattr(prism, field)(x, y, z, model)
    np.savez_compressed(TEST_DATA_FILE, **data)


def test_prism_regression(data, model):
    "Test the prism code against recorded results to check for regressions"
    inc, dec = data['inc'], data['dec']
    x, y, z = data['x'], data['y'], data['z']
    for field in FIELDS:
        if field == 'tf':
            result = getattr(prism, field)(x, y, z, model, inc, dec)
        else:
            result = getattr(prism, field)(x, y, z, model)
        if field == 'tf':
            tolerance = 1e-10
        elif field in 'bx by bz'.split():
            tolerance = 1e-10
        elif field == 'gz':
            tolerance = 1e-10
        elif field == 'gxx':
            tolerance = 1e-10
        elif field == 'gxy':
            tolerance = 1e-10
        elif field == 'gyy':
            tolerance = 1e-10
        elif field == 'gzz':
            tolerance = 1e-10
        else:
            tolerance = 1e-10
        npt.assert_allclose(result, data[field], atol=tolerance, rtol=0,
                            err_msg='field: {}'.format(field))
    density = model[0].props['density']
    for kernel in KERNELS:
        result = getattr(prism, kernel)(x, y, z, model[0])
        true = data['g' + kernel[-2:]]/constants.G/constants.SI2EOTVOS/density
        tolerance = 1e-10
        npt.assert_allclose(result, true, rtol=0, atol=tolerance,
                            err_msg='kernel: {}'.format(kernel))


def test_fails_if_shape_mismatch():
    'gravmag.prism fails if given computation points with different shapes'
    inc, dec = 10, 0
    model = [Prism(-6000, -2000, 2000, 4000, 0, 3000,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(10, inc, dec)})]
    area = [-5000, 5000, -10000, 10000]
    x, y, z = gridder.regular(area, (101, 51), z=-1)

    for field in FIELDS:
        func = getattr(prism, field)
        kwargs = {}
        if field == 'tf':
            kwargs['inc'] = inc
            kwargs['dec'] = dec
        raises(ValueError, func, x[:-2], y, z, model, **kwargs)
        raises(ValueError, func, x, y[:-2], z, model, **kwargs)
        raises(ValueError, func, x, y, z[:-2], model, **kwargs)
        raises(ValueError, func, x[:-5], y, z[:-2], model, **kwargs)

    for kernel in KERNELS:
        func = getattr(prism, kernel)
        raises(ValueError, func, x[:-2], y, z, model[0])
        raises(ValueError, func, x, y[:-2], z, model[0])
        raises(ValueError, func, x, y, z[:-2], model[0])
        raises(ValueError, func, x[:-5], y, z[:-2], model[0])


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
    # Test gravity functions
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
    for f in funcs:
        forced = getattr(prism, f)(x, y, z, model, dens=density)
        ref = getattr(prism, f)(x, y, z, reference)
        precision = 10
        assert_almost(forced, ref, precision, 'Field = %s' % (f))
    # Test magnetic functions
    funcs = ['tf', 'bx', 'by', 'bz']
    for f in funcs:
        if f == 'tf':
            forced = getattr(prism, f)(x, y, z, model, inc, dec, pmag=mag)
            ref = getattr(prism, f)(x, y, z, reference, inc, dec)
        else:
            forced = getattr(prism, f)(x, y, z, model, pmag=mag)
            ref = getattr(prism, f)(x, y, z, reference)
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
    # Test gravity functions
    funcs = ['potential', 'gx', 'gy', 'gz',
             'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
    for f in funcs:
        combined = getattr(prism, f)(x, y, z, model)
        separate = getattr(prism, f)(x, y, z, [model[1], model[4]])
        precision = 10
        assert_almost(separate, combined, precision, 'Field = %s' % (f))
    # Test magnetic functions
    funcs = ['tf', 'bx', 'by', 'bz']
    for f in funcs:
        mag_only = [model[1], model[2]]
        if f == 'tf':
            combined = getattr(prism, f)(x, y, z, model, inc, dec)
            separate = getattr(prism, f)(x, y, z, mag_only, inc, dec)
        else:
            combined = getattr(prism, f)(x, y, z, model)
            separate = getattr(prism, f)(x, y, z, mag_only)
        precision = 10
        assert_almost(separate, combined, precision, 'Field = %s' % (f))


def test_around():
    "gravmag.prism gravitational results are consistent around the prism"
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
