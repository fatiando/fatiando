"""
2D interpolation, griding, and profile extraction.
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.interpolate

from .point_generation import regular


def fill_nans(x, y, v, xp, yp, vp):
    """"
    Fill in the NaNs or masked values on interpolated points using nearest
    neighbors.

    .. warning::

        Operation is performed in place. Replaces the NaN or masked values of
        the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the original data points (not
        interpolated).
    * v : 1D array
        Array with the scalar value assigned to the data points (not
        interpolated).
    * xp, yp : 1D arrays
        Points where the data values were interpolated.
    * vp : 1D array
        Interpolated data values (the one that has NaNs or masked values to
        replace).

    """
    if np.ma.is_masked(vp):
        nans = vp.mask
    else:
        nans = np.isnan(vp)
    vp[nans] = scipy.interpolate.griddata((x, y), v, (xp[nans], yp[nans]),
                                          method='nearest').ravel()


def interp_at(x, y, v, xp, yp, algorithm='cubic', extrapolate=False):
    """
    Interpolate spacial data onto specified points.

    Wraps ``scipy.interpolate.griddata``.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * xp, yp : 1D arrays
        Points where the data values will be interpolated
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata)
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * v : 1D array
        1D array with the interpolated v values.

    """
    vp = scipy.interpolate.griddata((x, y), v, (xp, yp),
                                    method=algorithm).ravel()
    if extrapolate and algorithm != 'nearest' and np.any(np.isnan(vp)):
        fill_nans(x, y, v, xp, yp, vp)
    return vp


def interp(x, y, v, shape, area=None, algorithm='cubic', extrapolate=False):
    """
    Interpolate spacial data onto a regular grid.

    Utility function that generates a regular grid with
    :func:`~fatiando.gridder.regular` and calls
    :func:`~fatiando.gridder.interp_at` on the generated points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * shape : tuple = (nx, ny)
        Shape of the interpolated regular grid, ie (nx, ny).
    * area : tuple = (x1, x2, y1, y2)
        The are where the data will be interpolated. If None, then will get the
        area from *x* and *y*.
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata).
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * ``[x, y, v]``
        Three 1D arrays with the interpolated x, y, and v

    """
    if area is None:
        area = (x.min(), x.max(), y.min(), y.max())
    x1, x2, y1, y2 = area
    xp, yp = regular(area, shape)
    vp = interp_at(x, y, v, xp, yp, algorithm=algorithm,
                   extrapolate=extrapolate)
    return xp, yp, vp


def profile(x, y, v, point1, point2, size, algorithm='cubic'):
    """
    Extract a profile between 2 points from spacial data.

    Uses interpolation to calculate the data values at the profile points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * point1, point2 : lists = [x, y]
        Lists the x, y coordinates of the 2 points between which the profile
        will be extracted.
    * size : int
        Number of points along the profile.
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata).

    Returns:

    * [xp, yp, distances, vp] : 1d arrays
        ``xp`` and ``yp`` are the x, y coordinates of the points along the
        profile. ``distances`` are the distances of the profile points from
        ``point1``. ``vp`` are the data points along the profile.

    """
    x1, y1 = point1
    x2, y2 = point2
    maxdist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    distances = np.linspace(0, maxdist, size)
    angle = np.arctan2(y2 - y1, x2 - x1)
    xp = x1 + distances*np.cos(angle)
    yp = y1 + distances*np.sin(angle)
    vp = interp_at(x, y, v, xp, yp, algorithm=algorithm, extrapolate=True)
    return xp, yp, distances, vp
