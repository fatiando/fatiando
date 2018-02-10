from __future__ import division, absolute_import
import os
import numpy as np
import numpy.testing as npt

from .. import load_icgem_gdf

MODULE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MODULE_DIR, 'data')


def test_load_icgem_gdf():
    "Check if load_icgem_gdf reads ICGEM test data correctly"
    fname = os.path.join(TEST_DATA_DIR, "icgem-sample.gdf")
    data = load_icgem_gdf(fname)

    area = [14.0, 28.0, 150.0, 164.0]
    lon = np.arange(area[2], area[3] + 1, 2.0, dtype="float64")
    lat = np.arange(area[0], area[1] + 1, 2.0, dtype="float64")
    shape = (lat.size, lon.size)
    lon, lat = np.meshgrid(lon, lat)
    true_data = np.zeros_like(lat, dtype="float64")
    for i in range(true_data.shape[1]):
        true_data[:, i] = i
    lon, lat, true_data = lon.ravel(), lat.ravel(), true_data.ravel()

    assert data['shape'] == shape
    assert data['area'] == area
    npt.assert_equal(data['longitude'], lon)
    npt.assert_equal(data['latitude'], lat)
    assert data['sample_data'].size == true_data.size
    npt.assert_allclose(true_data, data['sample_data'])
