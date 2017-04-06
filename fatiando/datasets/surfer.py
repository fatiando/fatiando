"""
Functions for dealing with Surfer data grids.
"""
from __future__ import division, absolute_import

import numpy as np

from .. import gridder


def load_surfer(fname, dtype='float64'):
    """
    Read data from a Surfer ASCII grid file.

    Surfer is a contouring, griding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * dtype : numpy dtype object or string
        The type of variable used for the data. Default is numpy.float64. Use
        numpy.float32 if the data are large and precision is not an issue.

    Returns:

    * data : dict
        The data in a dictionary with some metadata:

        * ``'file'`` : string
            The name of the original data file
        * ``'shape'`` : tuple
            (nx, ny), the number of grid points in x (North) and y (East)
        * ``'area'`` : tuple
            (x1, x2, y1, y2), the spacial limits of the grid.
        * ``'x'`` : 1d-array
            Value of the North-South coordinate of each grid point.
        * ``'y'`` : 1d-array
            Value of the East-West coordinate of each grid point.
        * ``'data'`` : 1d-array
            Values of the field in each grid point. Field can be for example
            topography, gravity anomaly, etc. If any field values are >=
            1.70141e+38 (Surfers way of marking NaN values), the array will be
            masked at those values (i.e., ``'data'`` will be a numpy masked
            array).

    """
    # Surfer ASCII grid structure
    # DSAA            Surfer ASCII GRD ID
    # nCols nRows     number of columns and rows
    # xMin xMax       X min max
    # yMin yMax       Y min max
    # zMin zMax       Z min max
    # z11 z21 z31 ... List of Z values
    with open(fname) as input_file:
        # DSAA is a Surfer ASCII GRD ID (discard it for now)
        input_file.readline()
        # Read the number of columns (ny) and rows (nx)
        ny, nx = [int(s) for s in input_file.readline().split()]
        shape = (nx, ny)
        # Our x points North, so the first thing we read is y, not x.
        ymin, ymax = [float(s) for s in input_file.readline().split()]
        xmin, xmax = [float(s) for s in input_file.readline().split()]
        area = (xmin, xmax, ymin, ymax)
        dmin, dmax = [float(s) for s in input_file.readline().split()]
        field = np.fromiter((float(s)
                             for line in input_file
                             for s in line.split()),
                            dtype=dtype)
        nans = field >= 1.70141e+38
        if np.any(nans):
            field = np.ma.masked_where(nans, field)
        err_msg = "{} of data ({}) doesn't match one from file ({})."
        assert np.allclose(dmin, field.min()), err_msg.format('Min', dmin,
                                                              field.min())
        assert np.allclose(dmax, field.max()), err_msg.format('Max', dmax,
                                                              field.max())
    x, y = gridder.regular(area, shape)
    data = dict(file=fname, shape=shape, area=area, data=field, x=x, y=y)
    return data
