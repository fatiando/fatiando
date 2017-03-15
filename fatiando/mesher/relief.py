"""
Mesh classes that define a relief (topography) between two interfaces.
"""
from __future__ import division, print_function, absolute_import
from future.builtins import super
import numpy as np

from .base import RegularMesh
from .geometry import Prism


class PrismRelief(RegularMesh):
    """
    A 3D model of a relief (topography) between two interfaces using prisms.

    Use to generate:

    * topographic models
    * basin models
    * Moho models

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    ``PrismRelief`` can used as list of prisms. It acts as an iterator (so you
    can loop over prisms). You can also index it like a list to access
    individual elements in the mesh.  In practice, PrismRelief should be able
    to be passed to any function that asks for a list of prisms, like
    :func:`fatiando.gravmag.prism.gz`.

    Prisms are order in the mesh with the y direction varying first, then x.
    This is similar to the grid generation in :func:`fatiando.gridder.regular`.

    Parameters:

    * ref : float
        Reference level. Prisms will have:
            * bottom on zref and top on z if z > zref;
            * bottom on z and top on zref otherwise.
    * dims :  tuple = (dy, dx)
        Dimensions of the prisms in the y and x directions
    * nodes : list of lists = [x, y, z]
        Coordinates of the center of the top face of each prism. x, y, and z
        are lists with the x, y and z coordinates on a regular grid.
    * props : dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.

    Example::

        >>> # Create a relief from a regular grid of points
        >>> from fatiando import gridder
        >>> x, y = gridder.regular(area=(0, 1, -5, -1), shape=(2, 3))
        >>> x
        array([ 0.,  0.,  0.,  1.,  1.,  1.])
        >>> y
        array([-5., -3., -1., -5., -3., -1.])
        >>> z = [10, 9, -10, -9, 8, 7]
        >>> # Make the prism relief with 0 as the reference. Notice that the
        >>> # z coordinate falls below the reference at some points.
        >>> relief = PrismRelief(ref=0, dims=(1, 2), nodes=[x, y, z])
        >>> relief.size
        6
        >>> relief.shape
        (2, 3)
        >>> relief.bounds
        [-0.5, 1.5, -6.0, 0.0]
        >>> # You can iterate on the relief like a list of Prism objects
        >>> # Notice that the two negative z values are reversed with respect
        >>> # to the reference (zero)
        >>> for prism in relief:
        ...     print(prism.bounds)
        [-0.5, 0.5, -6.0, -4.0, 0, 10]
        [-0.5, 0.5, -4.0, -2.0, 0, 9]
        [-0.5, 0.5, -2.0, 0.0, -10, 0]
        [0.5, 1.5, -6.0, -4.0, -9, 0]
        [0.5, 1.5, -4.0, -2.0, 0, 8]
        [0.5, 1.5, -2.0, 0.0, 0, 7]
        >>> # You can also index the relief like a list
        >>> relief[3].bounds
        [0.5, 1.5, -6.0, -4.0, -9, 0]
        >>> relief[-4].bounds
        [-0.5, 0.5, -2.0, 0.0, -10, 0]
        >>> # And add physical properties to it
        >>> relief.addprop('density', [500, 600, 700, 800, 900, -500, -600])



    """

    def __init__(self, ref, dims, nodes, props=None):
        x, y, z = map(np.array, nodes)
        dx, dy = dims
        assert x.shape == y.shape == z.shape, \
            "nodes has x, y, z coordinate arrays of different shapes"
        bounds = [x.min() - 0.5*dx, x.max() + 0.5*dx,
                  y.min() - 0.5*dy, y.max() + 0.5*dy]
        nx = int((bounds[1] - bounds[0])/dx)
        ny = int((bounds[3] - bounds[2])/dy)
        shape = (nx, ny)
        super().__init__(bounds, shape, props)
        self.x, self.y, self.z = x, y, z
        self.ref = ref
        self.dims = dims
        self.dx, self.dy = dx, dy

    def _get_element(self, index):
        xc, yc, zc = self.x[index], self.y[index], self.z[index]
        x1 = xc - 0.5*self.dx
        x2 = xc + 0.5*self.dx
        y1 = yc - 0.5*self.dy
        y2 = yc + 0.5*self.dy
        if zc <= self.ref:
            z1 = zc
            z2 = self.ref
        else:
            z1 = self.ref
            z2 = zc
        return Prism(x1, x2, y1, y2, z1, z2)
