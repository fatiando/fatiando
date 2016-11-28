"""
Misc utility functions for grid manipulation.
"""
from __future__ import division, absolute_import, print_function


def spacing(area, shape):
    """
    Returns the spacing between grid nodes

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.

    Returns:

    * ``[dx, dy]``
        Spacing the y and x directions

    Examples:

    >>> print(spacing((0, 10, 0, 20), (11, 11)))
    [1.0, 2.0]
    >>> print(spacing((0, 10, 0, 20), (11, 21)))
    [1.0, 1.0]
    >>> print(spacing((0, 10, 0, 20), (5, 21)))
    [2.5, 1.0]
    >>> print(spacing((0, 10, 0, 20), (21, 21)))
    [0.5, 1.0]

    """
    x1, x2, y1, y2 = area
    nx, ny = shape
    dx = (x2 - x1)/(nx - 1)
    dy = (y2 - y1)/(ny - 1)
    return [dx, dy]
