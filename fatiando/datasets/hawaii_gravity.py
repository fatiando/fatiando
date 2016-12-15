"""
Load gravity data from the eigen-6c4 model for Hawaii.
"""
import os
import hashlib

import numpy as np

from . import check_hash

# The sha256 hash of the data file to make sure it is not corrupted.
SHA256 = "aed67ac8f8787e4c0365b65a5bc04322ae549842553243f74618c827ebf4aefa"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def fetch_hawaii_gravity():
    """
    Load raw gravity and topography data for Hawaii.

    The data were generated from the eigen-6c4 spherical harmonic model and
    downloaded from `ICGEM <http://icgem.gfz-potsdam.de/>`__).

    The topography is in meters and gravity in mGal.

    Returns:

    * data : dict
        A dictionary with the data fields:

        * ``'metadata'`` : string
            The headers from the original ICGEM files.
        * ``'shape'`` : tuple
            (nlat, nlon), the number of points in each dimension of the data
            grid.
        * ``'area'`` : tuple
            (south, north, west, east), the bounding dimensions of the data
            grid.
        * ``'height'`` : float
            The observation height of the gravity data in meters.
        * ``'lat'`` : 1d array
            The latitude coordinates of each point.
        * ``'lon'`` : 1d array
            The longitude coordinates of each point.
        * ``'topography'`` : 1d array
            The topographic (geometric) height of each point in meters.
        * ``'gravity'`` : 1d array
            The raw gravity value of each point in mGal.

    """
    fname = os.path.join(DATA_DIR, 'hawaii-gravity.npz')
    # Check the data file for corruption.
    check_hash(fname, known_hash=SHA256, hash_type='sha256')
    data = dict(np.load(fname))
    data['area'] = (data['lat'].min(), data['lat'].max(),
                    data['lon'].min(), data['lon'].max())
    # Some sanity checks
    keys = 'metadata shape area height lat lon topography gravity'.split()
    assert set(data.keys()) == set(keys), "Hawaii data might be corrupted"
    size = data['shape'][0]*data['shape'][0]
    for k in 'lat lon topography gravity'.split():
        assert data[k].size == size, 'Hawaii data might be corrupted'
    return data
