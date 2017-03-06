"""
3D mesh classes that behave as lists of their elements.
"""
from __future__ import division, print_function
from builtins import super
import numpy as np

from .base import RegularMesh
from .geometry import Prism, Tesseroid, Sphere


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

    def carvetopo(self, x, y, height, below=False):
        """
        Mask (remove) prisms from the mesh that are above the topography.

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
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        # The coordinates of the centers of the cells
        xc = np.arange(x1, x2, dx) + 0.5 * dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = np.arange(y1, y2, dy) + 0.5 * dy
        if len(yc) > ny:
            yc = yc[:-1]
        zc = np.arange(z1, z2, dz) + 0.5 * dz
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = np.meshgrid(xc, yc)
        topo = scipy.interpolate.griddata((x, y), height, (XC, YC),
                                          method='cubic').ravel()
        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # below a masked topo point (ie, one with no height information)
        if np.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in xrange(len(topo))]
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

    def get_xs(self):
        """
        Return an array with the x coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        xs = np.arange(x1, x2 + dx, dx)
        if xs.size > nx + 1:
            return xs[:-1]
        return xs

    def get_ys(self):
        """
        Return an array with the y coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        ys = np.arange(y1, y2 + dy, dy)
        if ys.size > ny + 1:
            return ys[:-1]
        return ys

    def get_zs(self):
        """
        Return an array with the z coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        zs = np.arange(z1, z2 + dz, dz)
        if zs.size > nz + 1:
            return zs[:-1]
        return zs

    def get_layer(self, i):
        """
        Return the set of prisms corresponding to the ith layer of the mesh.

        Parameters:

        * i : int
            The index of the layer

        Returns:

        * prisms : list of :class:`~fatiando.mesher.Prism`
            The prisms in the ith layer

        Examples::

            >>> mesh = PrismMesh((0, 2, 0, 2, 0, 2), (2, 2, 2))
            >>> layer = mesh.get_layer(0)
            >>> for p in layer:
            ...     print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:0 | z2:1
            x1:1 | x2:2 | y1:0 | y2:1 | z1:0 | z2:1
            x1:0 | x2:1 | y1:1 | y2:2 | z1:0 | z2:1
            x1:1 | x2:2 | y1:1 | y2:2 | z1:0 | z2:1
            >>> layer = mesh.get_layer(1)
            >>> for p in layer:
            ...     print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:1 | z2:2
            x1:1 | x2:2 | y1:0 | y2:1 | z1:1 | z2:2
            x1:0 | x2:1 | y1:1 | y2:2 | z1:1 | z2:2
            x1:1 | x2:2 | y1:1 | y2:2 | z1:1 | z2:2


        """
        nz, ny, nx = self.shape
        if i >= nz or i < 0:
            raise IndexError('Layer index %d is out of range.' % (i))
        start = i * nx * ny
        end = (i + 1) * nx * ny
        layer = [self.__getitem__(p) for p in xrange(start, end)]
        return layer

    def layers(self):
        """
        Returns an iterator over the layers of the mesh.

        Examples::

            >>> mesh = PrismMesh((0, 2, 0, 2, 0, 2), (2, 2, 2))
            >>> for layer in mesh.layers():
            ...     for p in layer:
            ...         print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:0 | z2:1
            x1:1 | x2:2 | y1:0 | y2:1 | z1:0 | z2:1
            x1:0 | x2:1 | y1:1 | y2:2 | z1:0 | z2:1
            x1:1 | x2:2 | y1:1 | y2:2 | z1:0 | z2:1
            x1:0 | x2:1 | y1:0 | y2:1 | z1:1 | z2:2
            x1:1 | x2:2 | y1:0 | y2:1 | z1:1 | z2:2
            x1:0 | x2:1 | y1:1 | y2:2 | z1:1 | z2:2
            x1:1 | x2:2 | y1:1 | y2:2 | z1:1 | z2:2

        """
        nz, ny, nx = self.shape
        for i in xrange(nz):
            yield self.get_layer(i)

    def dump(self, meshfile, propfile, prop):
        r"""
        Dump the mesh to a file in the format required by UBC-GIF program
        MeshTools3D.

        Parameters:

        * meshfile : str or file
            Output file to save the mesh. Can be a file name or an open file.
        * propfile : str or file
            Output file to save the physical properties *prop*. Can be a file
            name or an open file.
        * prop : str
            The name of the physical property in the mesh that will be saved to
            *propfile*.

        .. note:: Uses -10000000 as the dummy value for plotting topography

        Examples:

            >>> from StringIO import StringIO
            >>> meshfile = StringIO()
            >>> densfile = StringIO()
            >>> mesh = PrismMesh((0, 10, 0, 20, 0, 5), (1, 2, 2))
            >>> mesh.addprop('density', [1, 2, 3, 4])
            >>> mesh.dump(meshfile, densfile, 'density')
            >>> print meshfile.getvalue().strip()
            2 2 1
            0 0 0
            2*10
            2*5
            1*5
            >>> print densfile.getvalue().strip()
            1.0000
            3.0000
            2.0000
            4.0000

        """
        if prop not in self.props:
            raise ValueError("mesh doesn't have a '%s' property." % (prop))
        isstr = False
        if isinstance(meshfile, str):
            isstr = True
            meshfile = open(meshfile, 'w')
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        meshfile.writelines([
            "%d %d %d\n" % (ny, nx, nz),
            "%g %g %g\n" % (y1, x1, -z1),
            "%d*%g\n" % (ny, dy),
            "%d*%g\n" % (nx, dx),
            "%d*%g" % (nz, dz)])
        if isstr:
            meshfile.close()
        values = np.fromiter(self.props[prop], dtype=np.float)
        # Replace the masked cells with a dummy value
        values[self.mask] = -10000000
        reordered = np.ravel(np.reshape(values, self.shape), order='F')
        np.savetxt(propfile, reordered, fmt='%.4f')
