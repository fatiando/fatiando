"""
3D mesh classes that behave as lists of their elements.
"""
from __future__ import division, print_function
from builtins import super, range
import numpy as np

from .base import RegularMesh
from .geometry import Prism, Tesseroid, Sphere
from .. import gridder


class PrismMesh(RegularMesh):
    """
    A 3D regular mesh of right rectangular prisms.

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    :class:`~fatiando.mesher.PrismMesh` can used as list of prisms. It acts
    as an iterator (so you can loop over prisms). You can also index it like a
    list to access individual elements in the mesh.  In practice,
    :class:`~fatiando.mesher.PrismMesh` should be able to be passed to any
    function that asks for a list of prisms, like
    :func:`fatiando.gravmag.prism.gz`.

    Prisms are ordered as follows: first layers (z coordinate),
    then x coordinate (NS) and finally EW rows (y).

    To make the mesh incorporate a topography, use
    :meth:`~fatiando.mesher.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * shape : tuple = (nz, nx, ny)
        Number of prisms in the z, x, and y directions, respectively.
    * props : dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.

    Examples:

        >>> mesh = PrismMesh(bounds=(-5, -1, 0, 3, 0, 3), shape=(2, 2, 3),
        ...                  props={'density': list(range(12))})
        >>> mesh.size
        12
        >>> mesh.dims  # (dx, dy, dz)
        (2.0, 1.0, 1.5)
        >>> # Loop over the mesh elements like a list
        >>> for prism in mesh:
        ...     print(prism.bounds, 'density:', prism.props['density'])
        [-5.0, -3.0, 0.0, 1.0, 0.0, 1.5] density: 0
        [-5.0, -3.0, 1.0, 2.0, 0.0, 1.5] density: 1
        [-5.0, -3.0, 2.0, 3.0, 0.0, 1.5] density: 2
        [-3.0, -1.0, 0.0, 1.0, 0.0, 1.5] density: 3
        [-3.0, -1.0, 1.0, 2.0, 0.0, 1.5] density: 4
        [-3.0, -1.0, 2.0, 3.0, 0.0, 1.5] density: 5
        [-5.0, -3.0, 0.0, 1.0, 1.5, 3.0] density: 6
        [-5.0, -3.0, 1.0, 2.0, 1.5, 3.0] density: 7
        [-5.0, -3.0, 2.0, 3.0, 1.5, 3.0] density: 8
        [-3.0, -1.0, 0.0, 1.0, 1.5, 3.0] density: 9
        [-3.0, -1.0, 1.0, 2.0, 1.5, 3.0] density: 10
        [-3.0, -1.0, 2.0, 3.0, 1.5, 3.0] density: 11
        >>> # Index the mesh like a list
        >>> mesh[1].bounds
        [-5.0, -3.0, 1.0, 2.0, 0.0, 1.5]
        >>> mesh[-5].bounds
        [-5.0, -3.0, 1.0, 2.0, 1.5, 3.0]
        >>> # You can get a list (array) of the x, y, and z coordinates of each
        >>> # prism edge in the mesh.
        >>> xe, ye, ze = mesh.edges
        >>> xe
        array([-5., -3., -1.])
        >>> ye
        array([ 0.,  1.,  2.,  3.])
        >>> ze
        array([ 0. ,  1.5,  3. ])
        >>> # You can also get a list of the x, y, z coordinates of the centers
        >>> # of the prisms
        >>> xc, yc, zc = mesh.centers
        >>> xc
        array([-4., -2.])
        >>> yc
        array([ 0.5,  1.5,  2.5])
        >>> zc
        array([ 0.75,  2.25])
        >>> # Slicing through the vertical layers of the mesh
        >>> for prism in mesh.get_layer(0):
        ...     print(prism.bounds)
        [-5.0, -3.0, 0.0, 1.0, 0.0, 1.5]
        [-5.0, -3.0, 1.0, 2.0, 0.0, 1.5]
        [-5.0, -3.0, 2.0, 3.0, 0.0, 1.5]
        [-3.0, -1.0, 0.0, 1.0, 0.0, 1.5]
        [-3.0, -1.0, 1.0, 2.0, 0.0, 1.5]
        [-3.0, -1.0, 2.0, 3.0, 0.0, 1.5]
        >>> # or using the "layers" iterator
        >>> for i, layer in enumerate(mesh.layers):
        ...     print('layer:', i)
        ...     for prism in layer:
        ...         print(prism.bounds)
        layer: 0
        [-5.0, -3.0, 0.0, 1.0, 0.0, 1.5]
        [-5.0, -3.0, 1.0, 2.0, 0.0, 1.5]
        [-5.0, -3.0, 2.0, 3.0, 0.0, 1.5]
        [-3.0, -1.0, 0.0, 1.0, 0.0, 1.5]
        [-3.0, -1.0, 1.0, 2.0, 0.0, 1.5]
        [-3.0, -1.0, 2.0, 3.0, 0.0, 1.5]
        layer: 1
        [-5.0, -3.0, 0.0, 1.0, 1.5, 3.0]
        [-5.0, -3.0, 1.0, 2.0, 1.5, 3.0]
        [-5.0, -3.0, 2.0, 3.0, 1.5, 3.0]
        [-3.0, -1.0, 0.0, 1.0, 1.5, 3.0]
        [-3.0, -1.0, 1.0, 2.0, 1.5, 3.0]
        [-3.0, -1.0, 2.0, 3.0, 1.5, 3.0]


    """

    celltype = Prism

    def __init__(self, bounds, shape, props=None):
        super().__init__(bounds, shape, props)
        nz, nx, ny = shape
        x1, x2, y1, y2, z1, z2 = bounds
        dx = (x2 - x1)/nx
        dy = (y2 - y1)/ny
        dz = (z2 - z1)/nz
        self.dims = (dx, dy, dz)
        # Wether or not to change heights to z coordinate
        self.zdown = True

    def _get_element(self, index):
        "Return a Prism corresponding to index in the mesh"
        nz, nx, ny = self.shape
        k = index//(nx*ny)
        i = (index - k*(nx*ny))//ny
        j = (index - k*(nx*ny) - i*ny)
        x1 = self.bounds[0] + self.dims[0]*i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1]*j
        y2 = y1 + self.dims[1]
        z1 = self.bounds[4] + self.dims[2]*k
        z2 = z1 + self.dims[2]
        return self.celltype(x1, x2, y1, y2, z1, z2)

    @property
    def edges(self):
        "Arrays with the x, y, and z coordinates of the edges of the prisms."
        x1, x2, y1, y2, z1, z2 = self.bounds
        nz, nx, ny = self.shape
        xs = np.linspace(x1, x2, nx + 1)
        ys = np.linspace(y1, y2, ny + 1)
        zs = np.linspace(z1, z2, nz + 1)
        return xs, ys, zs

    @property
    def centers(self):
        "Arrays with the x, y, and z coordinates of the centers of the prisms."
        xs, ys, zs = self.edges
        dx, dy, dz = self.dims
        xc = xs[:-1] + 0.5*dx
        yc = ys[:-1] + 0.5*dy
        zc = zs[:-1] + 0.5*dz
        return xc, yc, zc

    def get_layer(self, i):
        """
        Return the set of prisms corresponding to the ith layer of the mesh.

        Parameters:

        * i : int
            The index of the layer

        Returns:

        * prisms : list of :class:`~fatiando.mesher.Prism`
            The prisms in the ith layer

        """
        nz, nx, ny = self.shape
        if i >= nz or i < 0:
            raise IndexError('Layer index %d is out of range.' % (i))
        start = i * nx * ny
        end = (i + 1) * nx * ny
        layer = [self[p] for p in range(start, end)]
        return layer

    @property
    def layers(self):
        """
        Returns an iterator over the layers of the mesh.
        """
        nz, nx, ny = self.shape
        for i in range(nz):
            yield self.get_layer(i)

    def carvetopo(self, x, y, height, below=False):
        """
        Mask (remove) prisms from the mesh that are above a given topography.

        Accessing the ith prism will return None if it was masked (above the
        topography).
        Also mask prisms outside of the topography grid provided.
        The topography height information does not need to be on a regular
        grid, it will be interpolated.

        Parameters:

        * x, y : lists
            x and y coordinates of the grid points
        * height : list or array
            Array with the height of the topography
        * below : boolean
            Will mask prisms below the input surface if set to *True*.

        """
        x, y, height = map(np.array, [x, y, height])
        xc, yc, zc = self.centers
        area = [xc.min(), xc.max(), yc.min(), yc.max()]
        xc, yc, topo = gridder.interp(x, y, height, self.shape[1:],
                                      algorithm='cubic')
        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        # If the interpolated point is out of the data range, topo will be a
        # masked array.
        # Use this to remove all cells below a masked topo point (ie, one
        # with no height information)
        if np.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False]*topo.size
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
                if below:
                    if (masked or
                            (cellz > h and self.zdown) or
                            (cellz < h and not self.zdown)):
                        self.mask.append(c)
                else:
                    if (masked or
                            (cellz < h and self.zdown) or
                            (cellz > h and not self.zdown)):
                        self.mask.append(c)
                c += 1
