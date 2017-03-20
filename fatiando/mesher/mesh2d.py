"""
2D meshes (can iterate over like a list of geometric objects).
"""
from __future__ import division, print_function, absolute_import
from builtins import object, super
import numpy as np
import copy

from .base import RegularMesh
from .geometry import Square


class SquareMesh(RegularMesh):
    """
    A 2D regular mesh of squares.

    For all purposes, :class:`~fatiando.mesher.SquareMesh` can be used as a
    list of :class:`~fatiando.mesher.Square`. The order of the squares in the
    list is: y directions varies first, then x. This is similar to the grid
    generation in :func:`fatiando.gridder.regular`.

    Parameters:

    * bounds :  list = [x1, x2, y1, y2]
        Boundaries of the mesh
    * shape : tuple = (nx, ny)
        Number of squares in the x and y dimension, respectively
    * props : dict
        Physical properties of each square in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each square of the mesh.

    Examples:

        >>> mesh = SquareMesh(bounds=(0, 2, 0, 6), shape=(2, 3))
        >>> # You can iterate on the mesh to get back each square
        >>> for square in mesh:
        ...     print(square.bounds)
        [0.0, 1.0, 0.0, 2.0]
        [0.0, 1.0, 2.0, 4.0]
        [0.0, 1.0, 4.0, 6.0]
        [1.0, 2.0, 0.0, 2.0]
        [1.0, 2.0, 2.0, 4.0]
        [1.0, 2.0, 4.0, 6.0]
        >>> # You can also index the mesh like a list
        >>> mesh[1].bounds
        [0.0, 1.0, 2.0, 4.0]
        >>> mesh[-2].bounds
        [1.0, 2.0, 2.0, 4.0]
        >>> len(mesh)
        6
        >>> mesh.size
        6
        >>> mesh.shape
        (2, 3)
        >>> # The dimensions of each cell of the mesh (dx, dy)
        >>> mesh.dims
        (1.0, 2.0)
        >>> # Physical properties are added as a list or array in the order of
        >>> # the iteration above
        >>> mesh.addprop('density', [1, 2, 3, 4, 5, 6])
        >>> for square in mesh:
        ...     print(square.props['density'])
        1
        2
        3
        4
        5
        6
        >>> # You can get a list (array) of the x and y coordinates of each
        >>> # division in the mesh.
        >>> mesh.xs
        array([ 0.,  1.,  2.])
        >>> mesh.ys
        array([ 0.,  2.,  4.,  6.])

    """

    def __init__(self, bounds, shape, props=None):
        super().__init__(bounds, shape, props)
        x1, x2, y1, y2 = bounds
        nx, ny = shape
        dx = (x2 - x1)/nx
        dy = (y2 - y1)/ny
        self.dims = (dx, dy)

    def _get_element(self, index):
        """
        Make a Square that corresponds to the given index of the mesh.
        """
        nx, ny = self.shape
        i = index//ny
        j = index - i*ny
        x1 = self.bounds[0] + self.dims[0]*i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1]*j
        y2 = y1 + self.dims[1]
        return Square((x1, x2, y1, y2))

    @property
    def xs(self):
        "A list of the x coordinates of the corners of the cells in the mesh."
        return np.linspace(self.bounds[0], self.bounds[1], self.shape[0] + 1)

    @property
    def ys(self):
        "A list of the y coordinates of the corners of the cells in the mesh."
        return np.linspace(self.bounds[2], self.bounds[3], self.shape[1] + 1)
