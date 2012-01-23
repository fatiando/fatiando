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

**Grid generation**

* :func:`fatiando.gridder.regular`
* :func:`fatiando.gridder.scatter`

**Grid I/O**

**Grid operations**

* :func:`fatiando.gridder.cut`
* :func:`fatiando.gridder.interpolate`

**Misc**

* :func:`fatiando.gridder.spacing`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'

from fatiando import logger

import numpy
import matplotlib.mlab

log = logger.dummy()


def regular(area, shape, z=None):
    """
    Create a regular grid. Order of the output grid is x varies first, then y.

    Parameters:
    
    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(ny, nx)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:
    
    * ``[xcoords, ycoords]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[xcoords, ycoords, zcoords]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    """
    log.info("Generating regular grid:")
    ny, nx = shape
    x1, x2, y1, y2 = area
    dy, dx = spacing(area, shape)
    log.info("  area = (x1, x2, y1, y2) = %s" % (str((x1,x2,y1,y2))))
    log.info("  shape = (ny, nx) = %s" % (str(shape)))
    log.info("  spacing = (dy, dx) = %s" % (str((dy, dx))))
    log.info("  points = nx*ny = %d" % (nx*ny))
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
        log.info("  z = %s" % (str(z)))
        zcoords = z*numpy.ones_like(xcoords)
        return [xcoords, ycoords, zcoords]
    else:
        return [xcoords, ycoords]

def scatter(area, n, z=None):
    """
    Create an irregular grid with a random scattering of points.

    Parameters:
    
    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * n
        Number of points
    * z
        Optional. z coordinate of the points. If given, will return an
        array with the value *z*.

    Returns:
    
    * ``[xcoords, ycoords]``
        Numpy arrays with the x and y coordinates of the points
    * ``[xcoords, ycoords, zcoords]``
        If *z* given. Arrays with the x, y, and z coordinates of the points

    """
    x1, x2, y1, y2 = area
    log.info("Generating irregular grid (scatter):")
    log.info("  area = (x1, x2, y1, y2) = %s" % (str((x1,x2,y1,y2))))
    log.info("  number of points = n = %s" % (str(n)))
    xcoords = numpy.random.uniform(x1, x2, n)
    ycoords = numpy.random.uniform(y1, y2, n)
    if z is not None:
        log.info("  z = %s" % (str(z)))
        zcoords = z*numpy.ones(n)
        return [xcoords, ycoords, zcoords]
    else:
        return [xcoords, ycoords]

def spacing(area, shape):
    """
    Returns the spacing between grid nodes

    Parameters:
    
    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(ny, nx)``.

    Returns:
    
    * ``[dy, dx]``
        Spacing the y and x directions

    """
    x1, x2, y1, y2 = area
    ny, nx = shape
    dx = float(x2 - x1)/float(nx - 1)
    dy = float(y2 - y1)/float(ny - 1)
    return [dy, dx]

def interpolate(x, y, v, shape, algorithm='nn'):
    """
    Interpolate data onto a regular grid.

    Parameters:
    
    * x, y
        Arrays with the x and y coordinates of the data points.
    * v
        Array with the scalar value assigned to the data points.
    * shape
        Shape of the interpolated regular grid, ie (ny, nx).
    * algorithm
        Interpolation algorithm. Either ``'nn'`` for natural neighbor interpolation
        or ``'linear'`` for linear interpolation. (see numpy.griddata)
        
    Returns:
    
    * ``[X, Y, V]``
        Three 2D arrays with the interpolated x, y, and v

    """
    if algorithm != 'nn' and algorithm != 'linear':
        raise ValueError, "Invalid interpolation: %s" % (str(algorithm))
    ny, nx = shape
    dx = float(x.max() - x.min())/(nx - 1)
    dy = float(y.max() - y.min())/(ny - 1)
    xs = numpy.arange(x.min(), x.max() + dx, dx, 'f')
    if len(xs) > nx:
        xs = xs[0:-1]
    ys = numpy.arange(y.min(), y.max() + dy, dy, 'f')
    if len(ys) > ny:
        ys = ys[0:-1]
    X, Y = numpy.meshgrid(xs, ys)
    V = matplotlib.mlab.griddata(x, y, v, X, Y, algorithm)
    return [X, Y, V]

def cut(x, y, scalars, area):
    """
    Remove a subsection of the grid.

    Parameters:
    
    * x, y
        Arrays with the x and y coordinates of the data points.
    * scalars
        List of arrays with the scalar values assigned to the grid points.
    * area
        ``(x1, x2, y1, y2)``: Borders of the subsection
        
    Returns:
    
    * ``[subx, suby, subscalars]``
        Arrays with x and y coordinates and scalar values of the subsection.

    """
    xmin, xmax, ymin, ymax = area
    log.info("Cutting grid:")
    log.info("  area = xmin, xmax, ymin, ymax = %s" % (str(area)))
    inside = []
    for i, coords in enumerate(zip(x, y)):
        xp, yp = coords
        if xp >= xmin and xp <= xmax and yp >= ymin and yp <= ymax:
            inside.append(i)
    subx = numpy.array([x[i] for i in inside])
    suby = numpy.array([y[i] for i in inside])
    subscalars = [numpy.array([scl[i] for i in inside]) for scl in scalars]
    return [subx, suby, subscalars]
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
