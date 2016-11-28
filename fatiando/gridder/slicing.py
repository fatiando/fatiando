"""
Functions for segmenting spacial data (windowing, cutting, etc).
"""
from __future__ import division, absolute_import, print_function


def inside(x, y, area):
    """
    Tell which indices of an array fall inside an area.

    Parameters:

    * x, y : ndarrays
        The x and y coordinate vectors.
    * area : list = [xmin, xmax, ymin, ymax]
        x and y limits of the area.

    Returns:

    * is_inside : ndarray of booleans
        An array of booleans. Will be ``True`` if the respective coordinates
        fall inside the area, ``False`` otherwise.

    Examples:

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([10, 11, 12, 13, 14, 15])
    >>> area = [2.5, 5.5, 12, 15]
    >>> is_inside = inside(x, y, area)
    >>> print(is_inside)
    [False False  True  True  True False]
    >>> # This also works for 2D-arrays
    >>> x = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> y = np.array([[5, 7, 9],
    ...               [5, 7, 9],
    ...               [5, 7, 9]])
    >>> area = [0.5, 2.5, 7, 9]
    >>> is_inside = inside(x, y, area)
    >>> print(is_inside)
    [[False  True  True]
     [False  True  True]
     [False False False]]

    """
    x1, x2, y1, y2 = area
    return ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))


def cut(x, y, scalars, area):
    """
    Return a subsection of a grid.

    The returned subsection is not a copy! In technical terms, returns a slice
    of the numpy arrays. So changes made to the subsection reflect on the
    original grid. Use numpy.copy to make copies of the subsections and avoid
    this.

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

    Examples:

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([10, 11, 12, 13, 14, 15])
    >>> data = np.array([42, 65, 92, 24, 135, 456])
    >>> area = [2.5, 5.5, 12, 15]
    >>> xs, ys, [datas] = cut(x, y, [data], area)
    >>> print(xs)
    [3 4 5]
    >>> print(ys)
    [12 13 14]
    >>> print(datas)
    [ 92  24 135]
    >>> # This also works for 2D-arrays
    >>> x = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> y = np.array([[5, 7, 9],
    ...               [5, 7, 9],
    ...               [5, 7, 9]])
    >>> data = np.array([[12, 84, 53],
    ...                  [43, 79, 29],
    ...                  [45, 27, 10]])
    >>> area = [0.5, 2.5, 7, 9]
    >>> xs, ys, [datas] = cut(x, y, [data], area)
    >>> print(xs)
    [1 1 2 2]
    >>> print(ys)
    [7 9 7 9]
    >>> print(datas)
    [84 53 79 29]

    """
    is_inside = inside(x, y, area)
    return x[is_inside], y[is_inside], [s[is_inside] for s in scalars]
