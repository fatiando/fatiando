# coding: utf-8
"""
Load gravity data from the eigen-6c4 model for Hawaii.
"""
from __future__ import unicode_literals, absolute_import
import os
import numpy as np

from . import check_hash

# The sha256 hash of the data file to make sure it is not corrupted.
SHA256 = "abc0e3c71e026c334d8452ed6d47af67f3962c6a955e26644bcf399e0513f5ba"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def fetch_hawaii_gravity():
    """
    Load gravity and topography data for Hawaii.

    The raw gravity and topography data were generated from the eigen-6c4
    spherical harmonic model and downloaded from `ICGEM
    <http://icgem.gfz-potsdam.de/>`__).

    The topography is in meters and gravity in mGal.

    x (north-south) and y (east-west) coordinates are UTM zone 4 (WGS84) in
    meters.

    The gravity disturbance was calculated using the closed-form formula and
    the WGS84 ellipsoid.

    The topography-free disturbances (both Bouguer and full) used densities
    2670 kg/m³ for the crust and 1040 kg/m³ for the ocean water.

    The full topography-free disturbance was calculated using a tesseroid model
    of the topography.

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
            grid in degrees.
        * ``'height'`` : float
            The observation height of the gravity data in meters.
        * ``'lat'``, ``'lon'`` : 1d arrays
            The latitude and longitude coordinates of each point.
        * ``'x'``, ``'y'`` : 1d arrays
            The UTM zone 4 coordinates of each point in meters. x is
            north-south and y is east-west.
        * ``'topography'`` : 1d array
            The topographic (geometric) height of each point in meters.
        * ``'gravity'`` : 1d array
            The raw gravity value of each point in mGal.
        * ``'disturbance'`` : 1d array
            The gravity disturbance value of each point in mGal.
        * ``'topo-free'`` : 1d array
            The topography-free gravity disturbance value of each point in
            mGal calculated using tesseroids.
        * ``'topo-free-bouguer'`` : 1d array
            The topography-free gravity disturbance value of each point in
            mGal calculated using the Bouguer plate approximation.

    """
    fname = os.path.join(DATA_DIR, 'hawaii-gravity.npz')
    # Check the data file for corruption.
    check_hash(fname, known_hash=SHA256, hash_type='sha256')
    data = dict(np.load(fname))
    data['area'] = (data['lat'].min(), data['lat'].max(),
                    data['lon'].min(), data['lon'].max())
    # Some sanity checks
    keys = ['metadata', 'shape', 'area', 'height', 'lat', 'lon', 'topography',
            'gravity', 'x', 'y', 'disturbance', 'topo-free',
            'topo-free-bouguer']
    assert set(data.keys()) == set(keys), "Hawaii data might be corrupted"
    size = data['shape'][0]*data['shape'][0]
    for k in keys[4:]:
        assert data[k].size == size, 'Hawaii data might be corrupted'
    return data
