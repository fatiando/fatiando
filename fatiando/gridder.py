# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Create and operate on grids and profiles.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'


import logging

import numpy
import matplotlib.mlab

def regular(x1, x2, y1, y2, nx, ny, z=None):
    """
    Create an regular grid. Order of the output grid is x varies first, then y.

    Parameters:
    * x1, x2, y1, y2
        Borders of the grid
    * nx, ny
        Number of points in the x and y directions, respectively
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:
    * [xcoords, ycoords]
        Numpy arrays with the x and y coordinates of the grid points
    * [xcoords, ycoords,zcoords]
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points
    """
    dx = float(x2 - x1)/float(nx - 1)
    dy = float(y2 - y1)/float(ny - 1)
    x_range = numpy.arange(x1, x2, dx)
    y_range = numpy.arange(y1, y2, dy)
    # Need to make sure that the number of points in the grid is correct because
    # of rounding errors in arange. Sometimes x2 and y2 are included, sometimes
    # not
    if len(x_range) < nx:
        x_range = numpy.append(x_range, x2)
    if len(y_range) < ny:
        y_range = numpy.append(y_range, y2)
    assert len(x_range) == nx, "Failed! x_range doesn't have nx points"
    assert len(y_range) == ny, "Failed! y_range doesn't have ny points"

    xcoords, ycoords = [mat.ravel() for mat in numpy.meshgrid(x_range, y_range)]
    if z is not None:
        zcoords = z*numpy.ones_like(xcoords)
        return [xcoords, ycoords, zcoords]
    else:
        return [xcoords, ycoords]


def interpolate(x, y, v, shape, algorithm='nn'):
    """
    Interpolate data onto a regular grid.

    Parameters:
    * x, y
        Arrays with the x and y coordinates of the data points.
    * v
        Array with the scalar value assigned to the data points.
    * shape
        Shape of the interpolated regular grid, ie (nx, ny).
    * algorithm
        Interpolation algorithm. Either ``'nn'`` for natural neighbor interpolation
        or ``'linear'`` for linear interpolation. (see numpy.griddata)
    Returns:
    * [X, Y, V]
        Three 2D arrays with the interpolated x, y, and v
    """
    if algorithm != 'nn' and algorithm != 'linear':
        raise ValueError, "Invalid interpolation: %s" % (str(algorithm))
    ny, nx = shape
    dx = float(x.max() - x.min())/nx
    dy = float(y.max() - y.min())/ny
    xs = numpy.arange(x.min(), x.max(), dx, 'f')
    ys = numpy.arange(y.min(), y.max(), dy, 'f')
    X, Y = numpy.meshgrid(xs, ys)
    V = matplotlib.mlab.griddata(x, y, v, X, Y, algorithm)
    return [X, Y, V]


def fill(values, grid, key='value'):
    """
    Fill a regular grid with the values in a 2D array **IN PLACE**.

    Parameters:

    * values
        2D array with the value to be put into each grid point.

    * grid
        A regular grid store in a dictionary (see :func:`fatiando.grid.regular`)

    * key
        The key in the grid dictionary the values will be put into.

    """

    varray = numpy.array(values)

    assert varray.shape == (grid['ny'], grid['nx']), \
        ("Woops, 'values' doesn't have the right shape. " +
        "Should be %d rows X %d columns (ny X nx)." % (grid['ny'], grid['nx']))

    grid[key] = varray.ravel()


def copy(grid):
    """
    Return a copy of *grid*.

    Parameters:

    * grid
        Grid stored in a dictionary.

    Returns:

    * copied
        Copy of *grid*

    """

    copied = {}

    for key in grid:

        copied[key] = numpy.copy(grid[key])

        if copied[key].shape == ():

            copied[key] = copied[key].tolist()

    return copied


def subtract(grid1, grid2, key1='value', key2='value', percent=False):
    """
    Subtract grid2 from grid1.

    Parameters:

    * grid1, grid2
        Grids stored in dictionaries. Do not need to be regular, but points must
        coincide. (see :mod:`fatiando.grid`)

    * key1, key2
        Keys in grid1 and grid2, respectively, with the values to subtract.

    * percent
        If ``True``, the difference will be calculated in a percentage of the
        value in *grid1*.

    Returns:

    * subgrid
        Grid with the subtraction results stored in key ``'value'``

    """

    assert key1 in grid1, "%s does not exist in grid1." % (key1)

    assert key2 in grid2, "%s does not exist in grid2." % (key2)

    assert len(grid1[key1]) == len(grid2[key2]), \
        "Grids must have same number of points!"

    # TODO: Check is 'grid' is the same in both

    value = []

    vappend = value.append

    for i, tmp in enumerate(grid1[key1]):

        assert grid1['x'][i] == grid2['x'][i] and \
                grid1['y'][i] == grid2['y'][i], \
                "Grid points must coincide! (point %d is off)" % (i)

        if not percent:

            vappend(grid1[key1][i] - grid2[key2][i])

        else:

            vappend(100*abs((grid1[key1][i] - grid2[key2][i])/grid1[key1][i]))

    subgrid = {'x':grid1['x'], 'y':grid1['y'], 'value':numpy.array(value)}

    if 'grid' in grid1 or 'grid' in grid2:

        subgrid['grid'] = grid1['grid']

        if grid1['grid'] and grid2['grid']:
            subgrid['nx'] = grid1['nx']
            subgrid['ny'] = grid1['ny']

    return subgrid


def cut(grid, xmin, xmax, ymin, ymax):
    """
    Remove a subsection of the grid.

    Parameters:

    * grid
        Grid stored in a dictionary.

    * xmin, xmax
        Limits of the subsection in the x direction.

    * ymin, ymax
        Limits of the subsection in the y direction.

    Returns:

    * subgrid
        Grid stored in a dictionary.

    """

    subgrid = {}

    for key in grid:

        if isinstance(grid[key], numpy.ndarray):

            subgrid[key] = []

    for i, pos in enumerate(zip(grid['x'], grid['y'])):

        x, y = pos

        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:

            for key in subgrid:

                subgrid[key].append(grid[key][i])

    for key in subgrid:

        subgrid[key] = numpy.array(subgrid[key])

    subgrid['grid'] = False

    return subgrid
