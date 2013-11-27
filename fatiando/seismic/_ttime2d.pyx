"""
Cython extension to speed up fatiando.seismic.ttime2d
"""
import numpy

from libc.math cimport sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython


@cython.boundscheck(False)
#@cython.wraparound(False)
def straight(
    numpy.ndarray[double, ndim=1] x_src,
    numpy.ndarray[double, ndim=1] y_src,
    numpy.ndarray[double, ndim=1] x_rec,
    numpy.ndarray[double, ndim=1] y_rec,
    int size, cells, velocity, prop):
    """
    Calculate the travel time of a straight ray.
    """
    cdef:
        unsigned int l, crossings, intercept
        double maxx, maxy, minx, miny, x1, x2, y1, y2, vel, a_ray, b_ray,
        double distance, xs, ys, xr, yr
        numpy.ndarray[double, ndim=1] times
        double[::1] xps, yps
        double[:,::1] cross
    xps = numpy.empty(6, dtype=numpy.float)
    yps = numpy.empty(6, dtype=numpy.float)
    cross = numpy.empty((6, 2), dtype=numpy.float)
    times = numpy.zeros(size, dtype=numpy.float)
    for cell in cells:
        if cell is None or (prop not in cell.props and velocity is None):
            continue
        x1, x2, y1, y2 = cell.x1, cell.x2, cell.y1, cell.y2
        if velocity is None:
            vel = cell.props[prop]
        else:
            vel = velocity
        for l in xrange(size):
            xs, ys = x_src[l], y_src[l]
            xr, yr = x_rec[l], y_rec[l]
            maxx = max(xs, xr)
            maxy = max(ys, yr)
            minx = min(xs, xr)
            miny = min(ys, yr)
            # Check if the cell is in the rectangle with the ray path as a
            # diagonal. If not, then the ray doesn't go through the cell.
            if x2 < minx or x1 > maxx or y2 < miny or y1 > maxy:
                continue
            # Now need to find the places where the ray intersects the cell
            # If the ray is vertical
            if (xr - xs) == 0:
                xps[:] = xr
                yps[0], yps[1], yps[2], yps[3] = yr, ys, y1, y2
                intercept = 4
            # If the ray is horizontal
            elif (yr - ys) == 0:
                xps[0], xps[1], xps[2], xps[3] = xr, xs, x1, x2
                yps[:] = yr
                intercept = 4
            else:
                # Angular and linear coefficients of the ray
                a_ray = float(yr - ys)/(xr - xs)
                b_ray = ys - a_ray*(xs)
                # Add the src and rec locations so that the travel time of a
                # src or rec inside a cell is accounted for
                xps[0] = x1
                xps[1] = x2
                xps[2] = (y1 - b_ray)/a_ray
                xps[3] = (y2 - b_ray)/a_ray
                xps[4] = xs
                xps[5] = xr
                yps[0] = a_ray*x1 + b_ray
                yps[1] = a_ray*x2 + b_ray
                yps[2] = y1
                yps[3] = y2
                yps[4] = ys
                yps[5] = yr
                intercept = 6
            # Find out how many points are inside both the cell and the
            # rectangle with the ray path as a diagonal. Also remove the
            # duplicates
            crossings = _cross(xps, yps, intercept, x1, x2, y1, y2, minx, maxx,
                               miny, maxy, cross)
            if crossings > 2:
                raise ValueError('More than 2 crossings ' +
                    'for cell %s and ray src:%s rec:%s'
                    % (str(cell), str([xs, ys]),
                       str([x_rec[l], yr])))
            if crossings == 2:
                distance = sqrt((cross[1, 0] - cross[0, 0])**2 +
                                (cross[1, 1] - cross[0, 1])**2)
                times[l] += distance/float(vel)
    return times

cdef inline unsigned int _cross(double[::1] xps, double[::1] yps,
    unsigned int intercept,
    double x1, double x2, double y1, double y2, double minx, double maxx,
    double miny, double maxy, double[:,::1] cross):
    cdef:
        unsigned int i, j, k, duplicate
    k = 0
    for i in range(intercept):
        if (xps[i] <= x2 and xps[i] >= x1 and yps[i] <= y2 and yps[i] >= y1 and
            xps[i] <= maxx and xps[i] >= minx and yps[i] <= maxy and
            yps[i] >= miny):
            duplicate = 0
            for j in range(k):
                if cross[j, 0] == xps[i] and cross[j, 1] == yps[i]:
                    duplicate = 1
                    break
            if duplicate:
                continue
            cross[k, 0] = xps[i]
            cross[k, 1] = yps[i]
            k += 1
    return k
