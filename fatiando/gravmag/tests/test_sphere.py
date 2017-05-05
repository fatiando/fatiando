"""
Tests for sphere forward modeling.
Includes regression tests comparing results against saved output.
"""
from __future__ import absolute_import, division
import os
import numpy as np
import numpy.testing as npt
import pytest

from ... import utils, gridder, constants
from ...mesher import Sphere
from ...datasets import check_hash
from .. import sphere


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
# Hash obtained using openssl
TEST_DATA_SHA256 = \
    'bd76969800fb269679f43b36a734167c45fc146c3d8e169eda40bdc382d77609'
TEST_DATA_FILE = os.path.join(TEST_DATA_DIR, 'sphere.npz')
FIELDS = 'gz gxx gxy gxz gyy gyz gzz bx by bz tf'.split()
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
    props = {'density': 1, 'magnetization': utils.ang2vec(1, 25, -10)}
    return [Sphere(x=-100, y=200, z=500, radius=400, props=props)]


def generate_test_data():
    """
    Create the test data file for regression tests.
    This function is not used when testing!
    It is here to document how the data file was generated.
    """
    model = make_model()
    # The geomagnetic field direction
    inc, dec = -30, 50
    shape = (101, 96)
    area = [-2000, 2000, -1900, 1900]
    x, y, z = gridder.regular(area, shape, z=-1)
    data = dict(x=x, y=y, z=z, inc=inc, dec=dec, shape=shape)
    for field in FIELDS:
        if field == 'tf':
            data[field] = getattr(sphere, field)(x, y, z, model, inc, dec)
        else:
            data[field] = getattr(sphere, field)(x, y, z, model)
    np.savez_compressed(TEST_DATA_FILE, **data)


def test_sphere_regression(data, model):
    "Test the sphere code against recorded results to check for regressions"
    x, y, z = data['x'], data['y'], data['z']
    inc, dec = data['inc'], data['dec']
    for field in FIELDS:
        if field == 'tf':
            result = getattr(sphere, field)(x, y, z, model, inc, dec)
        else:
            result = getattr(sphere, field)(x, y, z, model)
        if field == 'tf':
            tolerance = 1e-10
        elif field in 'bx by bz'.split():
            tolerance = 1e-10
        else:
            tolerance = 1e-10
        npt.assert_allclose(result, data[field], rtol=0, atol=tolerance)
    density = model[0].props['density']
    for kernel in KERNELS:
        result = getattr(sphere, kernel)(x, y, z, model[0])
        true = data['g' + kernel[-2:]]/constants.G/constants.SI2EOTVOS/density
        npt.assert_allclose(result, true, rtol=0, atol=1e-10)


def test_sphere_regression_force_prop(data, model):
    "Test the sphere code with forcing a physical property value"
    x, y, z = data['x'], data['y'], data['z']
    inc, dec = data['inc'], data['dec']
    pmag = -10*model[0].props['magnetization']
    dens = -10
    for field in FIELDS:
        if field == 'tf':
            result = getattr(sphere, field)(x, y, z, model, inc, dec,
                                            pmag=pmag)
        elif field in 'bx by bz'.split():
            result = getattr(sphere, field)(x, y, z, model, pmag=pmag)
        else:
            result = getattr(sphere, field)(x, y, z, model, dens=dens)
        npt.assert_allclose(result, -10*data[field], atol=1e-10, rtol=0)


def test_sphere_ignore_none(data, model):
    "Sphere ignores model elements that are None"
    x, y, z = data['x'], data['y'], data['z']
    inc, dec = data['inc'], data['dec']
    model_none = [None]*10
    model_none.extend(model)
    for field in FIELDS:
        if field == 'tf':
            result = getattr(sphere, field)(x, y, z, model_none, inc, dec)
        else:
            result = getattr(sphere, field)(x, y, z, model_none)
        npt.assert_allclose(result, data[field], atol=1e-10, rtol=0)


def test_sphere_ignore_missing_prop(data, model):
    "Sphere ignores model elements that don't have the needed property"
    x, y, z = data['x'], data['y'], data['z']
    inc, dec = data['inc'], data['dec']
    model2 = model + [Sphere(0, 0, 200, 200)]
    for field in FIELDS:
        if field == 'tf':
            result = getattr(sphere, field)(x, y, z, model2, inc, dec)
        else:
            result = getattr(sphere, field)(x, y, z, model2)
        npt.assert_allclose(result, data[field], atol=1e-10, rtol=0)
