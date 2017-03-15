from __future__ import absolute_import
import numpy as np
import numpy.testing as npt

from .. import fetch_hawaii_gravity


def test_hawaii_data():
    "Fetch the data and perform some sanity checks"
    data = fetch_hawaii_gravity()
    keys = ['metadata', 'shape', 'area', 'height', 'lat', 'lon', 'topography',
            'gravity', 'x', 'y', 'disturbance', 'topo-free',
            'topo-free-bouguer']
    assert set(data.keys()) == set(keys), "Different keys in data dict"
    npt.assert_allclose(data['shape'], (76, 76))
    size = data['shape'][0]*data['shape'][0]
    for k in keys[4:]:
        assert data[k].size == size, 'Array size and shape mismatch'
    npt.assert_allclose(data['area'], (13, 28, 195, 210))
    npt.assert_allclose(data['lat'].min(), 13)
    npt.assert_allclose(data['lat'].max(), 28)
    npt.assert_allclose(data['lon'].min(), 195)
    npt.assert_allclose(data['lon'].max(), 210)
    npt.assert_allclose(data['height'], 5000)
    npt.assert_allclose(data['gravity'].min(), 9.7673969e+05)
    npt.assert_allclose(data['gravity'].max(), 9.7766755e+05)
    npt.assert_allclose(data['topography'].min(), -6.2880000e+03)
    npt.assert_allclose(data['topography'].max(), 3.5530000e+03)
