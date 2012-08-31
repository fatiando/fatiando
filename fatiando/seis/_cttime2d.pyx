"""
Cython extension to speed up fatiando.seis.ttime2d
"""
import numpy

from libc.math cimport sqrt
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T


def straight(
    numpy.ndarray[DTYPE_T, ndim=1] x_src,
    numpy.ndarray[DTYPE_T, ndim=1] y_src,
    numpy.ndarray[DTYPE_T, ndim=1] x_rec,
    numpy.ndarray[DTYPE_T, ndim=1] y_rec,
    int size, cells, velocity, prop):
    """
    Calculate the travel time of a straight ray.
    """
    cdef int l, i
    cdef double maxx, maxy, minx, miny, x1, x2, y1, y2, vel, a_ray, b_ray, \
                 distance
    cdef numpy.ndarray[DTYPE_T, ndim=1] times = numpy.zeros(size, dtype=DTYPE)

    for l in xrange(size):
        maxx = max(x_src[l], x_rec[l])
        maxy = max(y_src[l], y_rec[l])
        minx = min(x_src[l], x_rec[l])
        miny = min(y_src[l], y_rec[l])
        for cell in cells:
            if cell is None or (prop not in cell.props and velocity is None):
                continue
            x1, x2, y1, y2 = cell.x1, cell.x2, cell.y1, cell.y2
            if velocity is None:
                vel = cell.props[prop]
            else:
                vel = velocity
            # Check if the cell is in the rectangle with the ray path as a
            # diagonal. If not, then the ray doesn't go through the cell.
            if x2 < minx or x1 > maxx or y2 < miny or y1 > maxy:
                continue
            # Now need to find the places where the ray intersects the cell
            # If the ray is vertical
            if (x_rec[l] - x_src[l]) == 0:
                xps = [x_rec[l]]*4
                yps = [y_rec[l], y_src[l], y1, y2]
            # If the ray is horizontal
            elif (y_rec[l] - y_src[l]) == 0:
                xps = [x_rec[l], x_src[l], x1, x2]
                yps = [y_rec[l]]*4
            else:
                # Angular and linear coefficients of the ray
                a_ray = float(y_rec[l] - y_src[l])/(x_rec[l] - x_src[l])
                b_ray = y_src[l] - a_ray*(x_src[l])
                # Add the src and rec locations so that the travel time of a src
                # or rec inside a cell is accounted for
                xps = [x1,  x2, (y1 - b_ray)/a_ray, (y2 - b_ray)/a_ray, x_src[l],
                       x_rec[l]]
                yps = [a_ray*x1 + b_ray, a_ray*x2 + b_ray, y1, y2, y_src[l],
                       y_rec[l]]
            # Find out how many points are inside both the cell and the
            # rectangle with the ray path as a diagonal
            incell = lambda x, y: x <= x2 and x >= x1 and y <= y2 and y >= y1
            inray = \
                lambda x, y: (x <= maxx and x >= minx and y <= maxy and
                               y >= miny)
            cross = [[x, y] for x, y in zip(xps, yps)
                     if incell(x, y) and inray(x, y)]
            # Remove the duplicates
            cross = [p for i, p in enumerate(cross) if p not in cross[0:i]]
            if len(cross) > 2:
                raise ValueError('More than 2 crossings ' +
                    'for cell %s and ray src:%s rec:%s'
                    % (str(cell), str([x_src[l], y_src[l]]),
                       str([x_rec[l], y_rec[l]])))
            if len(cross) == 2:
                p1, p2 = cross
                distance = sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                times[l] += distance/float(vel)
    return times
