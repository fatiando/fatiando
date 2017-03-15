from __future__ import division, absolute_import
import os
import numpy as np
import numpy.testing as npt

from .. import load_surfer

MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, 'data')


def test_load_surfer():
    "Check if load_surfer reads test data correctly"
    fname = os.path.join(TEST_DATA_DIR, 'simple_surfer.grd')
    for dtype in ['float64', 'float32']:
        data = load_surfer(fname, dtype=dtype)
        shape = (3, 11)

        assert set(data.keys()) == set('shape area file data x y'.split())
        assert data['shape'] == shape
        assert data['area'] == (0, 20, 0, 10)

        true_data = np.empty(shape)
        true_data[0, :] = -1
        true_data[1, :] = 1
        true_data[2, :] = 2
        npt.assert_allclose(true_data, data['data'].reshape(shape))

        true_x = np.empty(shape)
        true_x[0, :] = 0
        true_x[1, :] = 10
        true_x[2, :] = 20
        npt.assert_allclose(true_x, data['x'].reshape(shape))

        true_y = np.empty(shape)
        for j in range(11):
            true_y[:, j] = j
        npt.assert_allclose(true_y, data['y'].reshape(shape))
