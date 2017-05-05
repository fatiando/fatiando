"""
Tests for polyprism.
Includes regression tests comparing results against saved output.
"""
from __future__ import absolute_import, division
import os
import numpy as np
import numpy.testing as npt
import pytest

from ... import utils, gridder, constants
from ...mesher import PolygonalPrism
from ...datasets import check_hash
from .. import polyprism


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_DATA_FILE = os.path.join(TEST_DATA_DIR, 'polyprism.npz')
# Hash obtained using openssl
TEST_DATA_SHA256 = \
    '6b7f749692ff3698f5197d30ce26c2ec567421b482086627004457bd0fa8999d'
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
    # Make vertices scattered on an ellipse
    vertices = np.transpose(gridder.circular_scatter([-1000, 1000]*2, 30))
    vertices[:, 1] *= 2.5
    props = {'density': 1000, 'magnetization': utils.ang2vec(2, 25, -10)}
    return [PolygonalPrism(vertices, 100, 500, props)]


def generate_test_data():
    """
    Create the test data file for regression tests.
    This function is not used when testing!
    It is here to document how the data file was generated.
    """
    model = make_model()
    # The geomagnetic field direction
    inc, dec = -30, 20
    shape = (100, 91)
    area = [-4500, 4500, -5000, 5000]
    x, y, z = gridder.regular(area, shape, z=-10)
    data = dict(x=x, y=y, z=z, inc=inc, dec=dec, shape=shape)
    for field in FIELDS:
        if field == 'tf':
            data[field] = getattr(polyprism, field)(x, y, z, model, inc, dec)
        else:
            data[field] = getattr(polyprism, field)(x, y, z, model)
    np.savez_compressed(TEST_DATA_FILE, **data)


def test_polyprism_regression(data, model):
    "Test the polyprism code against recorded results to check for regressions"
    inc, dec = data['inc'], data['dec']
    x, y, z = data['x'], data['y'], data['z']
    for field in FIELDS:
        if field == 'tf':
            result = getattr(polyprism, field)(x, y, z, model, inc, dec)
        else:
            result = getattr(polyprism, field)(x, y, z, model)
        if field == 'tf':
            tolerance = 1e-7
        elif field in 'bx by bz'.split():
            tolerance = 1e-6
        elif field == 'gz':
            tolerance = 1e-6
        elif field == 'gxx':
            tolerance = 1e-8
        elif field == 'gxy':
            tolerance = 1e-6
        elif field == 'gyy':
            tolerance = 1e-8
        elif field == 'gzz':
            tolerance = 1e-9
        else:
            tolerance = 1e-10
        npt.assert_allclose(result, data[field], atol=tolerance, rtol=0,
                            err_msg='field: {}'.format(field))
    density = model[0].props['density']
    for kernel in KERNELS:
        result = getattr(polyprism, kernel)(x, y, z, model[0])
        true = data['g' + kernel[-2:]]/constants.G/constants.SI2EOTVOS/density
        tolerance = 1e-8
        npt.assert_allclose(result, true, rtol=0, atol=tolerance,
                            err_msg='kernel: {}'.format(kernel))
