"""
Generate points on a map as regular grids or points scatters.
"""
from __future__ import division, absolute_import, print_function
import numpy as np


def _check_area(area):
    """
    Check that the area argument is valid.
    For example, the west limit should not be greater than the east limit.
    """
    x1, x2, y1, y2 = area
    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)


def regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples:

    >>> x, y = regular((0, 10, 0, 5), (5, 3))
    >>> print(x)
    [  0.    0.    0.    2.5   2.5   2.5   5.    5.    5.    7.5   7.5   7.5
      10.   10.   10. ]
    >>> print(x.reshape((5, 3)))
    [[  0.    0.    0. ]
     [  2.5   2.5   2.5]
     [  5.    5.    5. ]
     [  7.5   7.5   7.5]
     [ 10.   10.   10. ]]
    >>> print(y.reshape((5, 3)))
    [[ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]]
    >>> x, y = regular((0, 0, 0, 5), (1, 3))
    >>> print(x.reshape((1, 3)))
    [[ 0.  0.  0.]]
    >>> print(y.reshape((1, 3)))
    [[ 0.   2.5  5. ]]
    >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
    >>> print(z.reshape((5, 3)))
    [[-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]]


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    _check_area(area)
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = np.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float))
    return [i.ravel() for i in arrays]


def scatter(area, n, z=None, seed=None):
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
    * seed : None or int
        Seed used to generate the pseudo-random numbers. If `None`, will use a
        different seed every time. Use the same seed to generate the same
        random points.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the points
    * ``[x, y, z]``
        If *z* given. Arrays with the x, y, and z coordinates of the points

    Examples:

    >>> # Passing in a seed value will ensure that scatter will return the same
    >>> # values given the same input. Use seed=None if you don't want this.
    >>> x, y = scatter((0, 10, 0, 2), 4, seed=0)
    >>> # Small function to print the array values with 4 decimal places
    >>> pprint = lambda arr: print(', '.join('{:.4f}'.format(i) for i in arr))
    >>> pprint(x)
    5.4881, 7.1519, 6.0276, 5.4488
    >>> pprint(y)
    0.8473, 1.2918, 0.8752, 1.7835

    >>> # scatter can take the z argument as well
    >>> x2, y2, z2 = scatter((-10, 1, 1245, 3556), 6, z=-150, seed=2)
    >>> pprint(x2)
    -5.2041, -9.7148, -3.9537, -5.2115, -5.3760, -6.3663
    >>> pprint(y2)
    1717.9430, 2676.1352, 1937.5020, 1861.6378, 2680.4403, 2467.8474
    >>> pprint(z2)
    -150.0000, -150.0000, -150.0000, -150.0000, -150.0000, -150.0000

    """
    x1, x2, y1, y2 = area
    _check_area(area)
    np.random.seed(seed)
    arrays = [np.random.uniform(x1, x2, n), np.random.uniform(y1, y2, n)]
    if z is not None:
        arrays.append(z*np.ones(n))
    return arrays


def circular_scatter(area, n, z=None, random=False, seed=None):
    """
    Generate a set of n points positioned in a circular array.

    The diameter of the circle is equal to the smallest dimension of the area

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Area inside of which the points are contained
    * n : int
        Number of points
    * z : float or 1d-array
        Optional. z coordinate of the points. If given, will return an
        array with the value *z*.
    * random : True or False
        If True, positions of the points on the circle will be chosen at random
    * seed : None or int
        Seed used to generate the pseudo-random numbers if `random==True`.
        If `None`, will use a different seed every time.
        Use the same seed to generate the same random sequence.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the points
    * ``[x, y, z]``
        If *z* given. Arrays with the x, y, and z coordinates of the points

    """
    x1, x2, y1, y2 = area
    radius = 0.5 * min(x2 - x1, y2 - y1)
    if random:
        np.random.seed(seed)
        angles = np.random.uniform(0, 2*np.pi, n)
        np.random.seed()
    else:
        # The last point is the same as the first, so discard it
        angles = np.linspace(0, 2*np.pi, n + 1)[:-1]
    xs = 0.5*(x1 + x2) + radius*np.cos(angles)
    ys = 0.5*(y1 + y2) + radius*np.sin(angles)
    arrays = [xs, ys]
    if z is not None:
        arrays.append(z*np.ones(n))
    return arrays
