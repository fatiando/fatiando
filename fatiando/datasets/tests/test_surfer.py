from __future__ import division, absolute_import
import os
import numpy as np
import numpy.testing as npt

from .. import load_surfer

MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, 'data')


def test_load_surfer():
    "Check if load_surfer reads test data correctly (no masked values)"
    fname = os.path.join(TEST_DATA_DIR, 'simple_surfer.grd')
    for dtype in ['float64', 'float32']:
        data = load_surfer(fname, dtype=dtype)
        shape = (3, 11)

        assert set(data.keys()) == set('shape area file data x y'.split())
        assert data['shape'] == shape
        assert data['area'] == (0, 20, 0, 10)
        assert not np.ma.is_masked(data['data'])
        assert data['data'].dtype == dtype
        assert data['data'].size == 33

        true_data = np.empty(shape, dtype=dtype)
        true_data[0, :] = -1
        true_data[1, :] = 1
        true_data[2, :] = 2
        npt.assert_allclose(true_data, data['data'].reshape(shape))

        true_x = np.empty(shape, dtype=dtype)
        true_x[0, :] = 0
        true_x[1, :] = 10
        true_x[2, :] = 20
        npt.assert_allclose(true_x, data['x'].reshape(shape))

        true_y = np.empty(shape, dtype=dtype)
        for j in range(11):
            true_y[:, j] = j
        npt.assert_allclose(true_y, data['y'].reshape(shape))


def test_load_surfer_masked():
    "Check if load_surfer reads test data correctly (with masked values)"
    fname = os.path.join(TEST_DATA_DIR, 'simple_surfer_masked.grd')
    for dtype in ['float64', 'float32']:
        data = load_surfer(fname, dtype=dtype)
        shape = (3, 11)

        assert set(data.keys()) == set('shape area file data x y'.split())
        assert data['shape'] == shape
        assert data['area'] == (0, 20, 0, 10)
        assert np.ma.is_masked(data['data'])
        assert data['data'].dtype == dtype
        assert data['data'].size == 33

        # Check is the mask is in the right places
        npt.assert_equal(np.arange(data['data'].size)[data['data'].mask],
                         [2, 32])

        true_data = np.empty(shape, dtype=dtype)
        true_data[0, :] = -1
        true_data[1, :] = 1
        true_data[2, :] = 2
        mask = np.empty(shape, dtype='bool')
        mask[:, :] = False
        mask[0, 2] = True
        mask[2, 10] = True
        true_data = np.ma.array(true_data, mask=mask)
        npt.assert_allclose(true_data, data['data'].reshape(shape))

        true_x = np.empty(shape, dtype=dtype)
        true_x[0, :] = 0
        true_x[1, :] = 10
        true_x[2, :] = 20
        npt.assert_allclose(true_x, data['x'].reshape(shape))

        true_y = np.empty(shape, dtype=dtype)
        for j in range(11):
            true_y[:, j] = j
        npt.assert_allclose(true_y, data['y'].reshape(shape))
