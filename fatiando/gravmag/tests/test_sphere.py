import os
import numpy as np
import numpy.testing as npt
from pytest import raises

from ... import utils, gridder, constants
from ...mesher import Sphere
from ...datasets import check_hash
from .. import sphere


# Load the test data to check against a regression
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATA_SHA256 = \
    'cbc2e8d64b705325812dd4587bdb52452dbd2cbec3634c06070d1ff71962cd0f'
TEST_DATA = os.path.join(TEST_DATA_DIR, 'sphere.npz')
check_hash(TEST_DATA, DATA_SHA256, hash_type='sha256')
data = np.load(TEST_DATA)

model = eval(str(data['model']))
inc, dec = data['inc'], data['dec']
sinc, sdec = data['sinc'], data['sdec']
area = data['area']
shape = data['shape']
height = data['z']
x, y, z = gridder.regular(area, shape, z=height)


def test_sphere_regression():
    "Test the sphere code against recorded results to check for regressions"
    for field in 'gz gxx gxy gxz gyy gyz gzz bx by bz'.split():
        result = getattr(sphere, field)(x, y, z, model)
        npt.assert_allclose(result, data[field], rtol=1e-10)
    result = sphere.tf(x, y, z, model, inc, dec)
    npt.assert_allclose(result, data['tf'], rtol=1e-10)
    kernels = ['kernel' + k for k in 'xx xy xz yy yz yz zz'.split()]
    for kernel in kernels:
        for s in model:
            result = getattr(sphere, kernel)(x, y, z, s)
            true = data['g' + kernel[-2:]]/constants.G/constants.SI2EOTVOS
            npt.assert_allclose(result, true, rtol=1e-10)


def test_sphere_regression_force_prop():
    "Test the sphere code with forcing a physical property value"
    for field in 'gz gxx gxy gxz gyy gyz gzz'.split():
        result = getattr(sphere, field)(x, y, z, model, dens=-10)
        npt.assert_allclose(result, -10*data[field], rtol=1e-10)
    pmag = utils.ang2vec(-10, sinc, sdec)
    for field in 'bx by bz'.split():
        result = getattr(sphere, field)(x, y, z, model, pmag=pmag)
        npt.assert_allclose(result, -10*data[field], rtol=1e-10)
    result = sphere.tf(x, y, z, model, inc, dec, pmag=pmag)
    npt.assert_allclose(result, -10*data['tf'], rtol=1e-10)


def test_sphere_ignore_none():
    "Sphere ignores model elements that are None"
    model_none = [None]*10
    model_none.extend(model)
    for field in 'gz gxx gxy gxz gyy gyz gzz bx by bz'.split():
        result = getattr(sphere, field)(x, y, z, model_none)
        npt.assert_allclose(result, data[field], rtol=1e-10)
    result = sphere.tf(x, y, z, model_none, inc, dec)
    npt.assert_allclose(result, data['tf'], rtol=1e-10)


def test_sphere_ignore_missing_prop():
    "Sphere ignores model elements that don't have the needed property"
    model2 = model + [Sphere(0, 0, 200, 200)]
    for field in 'gz gxx gxy gxz gyy gyz gzz bx by bz'.split():
        result = getattr(sphere, field)(x, y, z, model2)
        npt.assert_allclose(result, data[field], rtol=1e-10)
    result = sphere.tf(x, y, z, model2, inc, dec)
    npt.assert_allclose(result, data['tf'], rtol=1e-10)
