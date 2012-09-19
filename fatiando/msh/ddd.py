"""
Create and operate on meshes of 3D objects like prisms, polygonal prisms,
tesseroids, etc.

**Elements**

* :class:`~fatiando.msh.ddd.Prism`
* :class:`~fatiando.msh.ddd.PolygonalPrism`
* :class:`~fatiando.msh.ddd.Sphere`
* :class:`~fatiando.msh.ddd.Tesseroid`

**Meshes**

* :class:`~fatiando.msh.ddd.PrismMesh`
* :class:`~fatiando.msh.ddd.PrismRelief`

**Utility functions**

* :func:`~fatiando.msh.ddd.extract`: Extract the values of a physical
  property from the cells in a list
* :func:`~fatiando.msh.ddd.vfilter`: Remove cells whose physical property
  value falls outside a given range
* :func:`~fatiando.msh.ddd.vremove`: Remove the cells with a given physical
  property value

----

"""

import numpy
import matplotlib.mlab

import fatiando.log
from fatiando.msh.dd import Polygon
from fatiando.msh.base import GeometricElement


log = fatiando.log.dummy('fatiando.msh.ddd')


class Prism(GeometricElement):
    """
    Create a 3D right rectangular prism.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x1, x2 : float
        South and north borders of the prism
    * y1, y2 : float
        West and east borders of the prism
    * z1, z2 : float
        Top and bottom of the prism
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> from fatiando.msh.ddd import Prism
        >>> p = Prism(1, 2, 3, 4, 5, 6, {'density':200})
        >>> p.props['density']
        200
        >>> print p.get_bounds()
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:200
        >>> p = Prism(1, 2, 3, 4, 5, 6)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6
        >>> p.addprop('density', 2670)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:2670

    """

    def __init__(self, x1, x2, y1, y2, z1, z2, props=None):
        GeometricElement.__init__(self, props)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.z1 = float(z1)
        self.z2 = float(z2)

    def __str__(self):
        """Return a string representation of the prism."""
        names = [('x1', self.x1), ('x2', self.x2), ('y1', self.y1),
                 ('y2', self.y2), ('z1', self.z1), ('z2', self.z2)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the prism (i.e., the borders of the prism).

        Returns:

        * bounds : list
            ``[x1, x2, y1, y2, z1, z2]``, the bounds of the prism

        Examples:

            >>> p = Prism(1, 2, 3, 4, 5, 6)
            >>> print p.get_bounds()
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        """
        return [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]

    def center(self):
        """
        Return the coordinates of the center of the prism.

        Returns:

        * coords : list = [xc, yc, zc]
            Coordinates of the center

        Example:

            >>> prism = Prism(1, 2, 1, 3, 0, 2)
            >>> print prism.center()
            [1.5, 2.0, 1.0]

        """
        xc = 0.5*(self.x1 + self.x2)
        yc = 0.5*(self.y1 + self.y2)
        zc = 0.5*(self.z1 + self.z2)
        return [xc, yc, zc]

class Tesseroid(GeometricElement):
    """
    Create a tesseroid (spherical prism).

    Parameters:

    * w, e : float
        West and east borders of the tesseroid in decimal degrees
    * s, n : float
        South and north borders of the tesseroid in decimal degrees
    * bottom, top : float
        Bottom and top of the tesseroid with respect to the mean earth radius
        in meters. Ex: if the top is 100 meters above the mean earth radius,
        ``top=100``, if 100 meters bellow ``top=-100``.
    * props : dict
        Physical properties assigned to the tesseroid.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> from fatiando.msh.ddd import Tesseroid
        >>> t = Tesseroid(1, 2, 3, 4, 5, 6, {'density':200})
        >>> t.props['density']
        200
        >>> print t.get_bounds()
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> print t
        w:1 | e:2 | s:3 | n:4 | bottom:5 | top:6 | density:200
        >>> t = Tesseroid(1, 2, 3, 4, 5, 6)
        >>> print t
        w:1 | e:2 | s:3 | n:4 | bottom:5 | top:6
        >>> t.addprop('density', 2670)
        >>> print t
        w:1 | e:2 | s:3 | n:4 | bottom:5 | top:6 | density:2670

    """

    def __init__(self, w, e, s, n, bottom, top, props=None):
        GeometricElement.__init__(self, props)
        self.w = float(w)
        self.e = float(e)
        self.s = float(s)
        self.n = float(n)
        self.bottom = float(bottom)
        self.top = float(top)

    def __str__(self):
        """Return a string representation of the tesseroid."""
        names = [('w', self.w), ('e', self.e), ('s', self.s),
                 ('n', self.n), ('bottom', self.bottom), ('top', self.top)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the tesseroid (i.e., the borders).

        Returns:

        * bounds : list
            ``[w, e, s, n, bottom, top]``, the bounds of the tesseroid

        Examples:

            >>> t = Tesseroid(1, 2, 3, 4, 5, 6)
            >>> print t.get_bounds()
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        """
        return [self.w, self.e, self.s, self.n, self.bottom, self.top]

class Sphere(GeometricElement):
    """
    Create a sphere.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the sphere
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> s = Sphere(1, 2, 3, 10, {'magnetization':200})
        >>> s.props['magnetization']
        200
        >>> s.addprop('density', 20)
        >>> print s.props['density']
        20
        >>> print s
        x:1 | y:2 | z:3 | radius:10 | density:20 | magnetization:200
        >>> s = Sphere(1, 2, 3, 4)
        >>> print s
        x:1 | y:2 | z:3 | radius:4
        >>> s.addprop('density', 2670)
        >>> print s
        x:1 | y:2 | z:3 | radius:4 | density:2670

    """

    def __init__(self, x, y, z, radius, props=None):
        GeometricElement.__init__(self, props)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)

    def __str__(self):
        """Return a string representation of the sphere."""
        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('radius', self.radius)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

class PolygonalPrism(GeometricElement):
    """
    Create a 3D prism with polygonal crossection.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    .. note:: *vertices* must be **CLOCKWISE** or will give inverse result.

    Parameters:

    * vertices : list of lists
        Coordinates of the vertices. A list of ``[x, y]`` pairs.
    * z1, z2 : float
        Top and bottom of the prism
    * props :  dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
        >>> p = PolygonalPrism(verts, 0, 3, props={'temperature':25})
        >>> p.props['temperature']
        25
        >>> print p.x
        [ 1.  1.  2.  2.]
        >>> print p.y
        [ 1.  2.  2.  1.]
        >>> print p.z1, p.z2
        0.0 3.0
        >>> p.addprop('density', 2670)
        >>> print p.props['density']
        2670

    """
    def __init__(self, vertices, z1, z2, props=None):
        GeometricElement.__init__(self, props)
        x, y = numpy.array(vertices, dtype='f').T
        self.x = x
        self.y = y
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.nverts = len(vertices)

    def topolygon(self):
        """
        Get the polygon describing the prism viewed from above.

        Returns:

        * polygon : :func:`fatiando.msh.dd.Polygon`
            The polygon

        Example:

            >>> verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
            >>> p = PolygonalPrism(verts, 0, 100)
            >>> poly = p.topolygon()
            >>> print poly.x
            [ 1.  1.  2.  2.]
            >>> print poly.y
            [ 1.  2.  2.  1.]

        """
        verts = numpy.transpose([self.x, self.y])
        return Polygon(verts, self.props)

class PrismRelief(object):
    """
    Generate a 3D model of a relief (topography) using prisms.

    Use to generate:
    * topographic model
    * basin model
    * Moho model
    * etc

    PrismRelief can used as list of prisms. It acts as an iteratior (so you
    can loop over prisms). It also has a ``__getitem__`` method to access
    individual elements in the mesh.
    In practice, PrismRelief should be able to be passed to any function that
    asks for a list of prisms, like :func:`fatiando.pot.prism.gz`.

    Parameters:

    * ref : float
        Reference level. Prisms will have:
            * bottom on zref and top on z if z > zref;
            * bottom on z and top on zref otherwise.
    * dims :  tuple = (dy, dx)
        Dimensions of the prisms in the y and x directions
    * nodes : list of lists = [x, y, z]
        Coordinates of the center of the top face of each prism.x, y, and z are
        lists with the x, y and z coordinates on a regular grid.

    """

    def __init__(self, ref, dims, nodes):
        object.__init__(self)
        x, y, z = nodes
        if len(x) != len(y) != len(z):
            raise ValueError, "nodes has x,y,z coordinates of different lengths"
        self.x, self.y, self.z = x, y, z
        self.size = len(x)
        self.ref = ref
        self.dy, self.dx = dims
        self.props = {}
        log.info("Generating 3D relief with right rectangular prisms:")
        log.info("  number of prisms = %d" % (self.size))
        log.info("  reference level = %s" % (str(ref)))
        log.info("  dimensions of prisms = %g x %g" % (dims[0], dims[1]))
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, index):
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
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
        props = dict([p, self.props[p][index]] for p in self.props)
        return Prism(x1, x2, y1, y2, z1, z2, props=props)

    def next(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the prisms.

        .. warning:: If the z value of any point in the relief is bellow the
            reference level, its corresponding prism will have the physical
            property value with oposite sign than was assigned to it.

        Parameters:

        * prop : str
            Name of the physical property.
        * values : list
            List or array with the value of this physical property in each
            prism of the relief.

        """
        def correct(v, i):
            if self.z[i] > self.ref:
                return -v
            return v
        self.props[prop] = [correct(v, i) for i, v in enumerate(values)]

class PrismMesh(object):
    """
    Generate a 3D regular mesh of right rectangular prisms.

    Prisms are ordered as follows: first layers (z coordinate), then EW rows (y)
    and finaly x coordinate (NS).

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has z index
    1 (second layer), y index 1 (second row), and x index 2 (third element in
    the column).

    :class:`~fatiando.msh.ddd.PrismMesh` can used as list of prisms. It acts
    as an iteratior (so you can loop over prisms). It also has a ``__getitem__``
    method to access individual elements in the mesh.
    In practice, :class:`~fatiando.msh.ddd.PrismMesh` should be able to be
    passed to any function that asks for a list of prisms, like
    :func:`fatiando.pot.prism.gz`.

    To make the mesh incorporate a topography, use
    :meth:`~fatiando.msh.ddd.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * shape : tuple = (nz, ny, nx)
        Number of prisms in the x, y, and z directions, respectively.
    * props :  dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.

    Examples:

        >>> from fatiando.msh.ddd import PrismMesh
        >>> mesh = PrismMesh((0,1,0,2,0,3),(1,2,2))
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:0.5 | y1:0 | y2:1 | z1:0 | z2:3
        x1:0.5 | x2:1 | y1:0 | y2:1 | z1:0 | z2:3
        x1:0 | x2:0.5 | y1:1 | y2:2 | z1:0 | z2:3
        x1:0.5 | x2:1 | y1:1 | y2:2 | z1:0 | z2:3
        >>> print mesh[0]
        x1:0 | x2:0.5 | y1:0 | y2:1 | z1:0 | z2:3
        >>> print mesh[-1]
        x1:0.5 | x2:1 | y1:1 | y2:2 | z1:0 | z2:3

    One with physical properties::

        >>> props = {'density':[2670.0, 1000.0]}
        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2), props=props)
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:1 | y1:0 | y2:4 | z1:0 | z2:3 | density:2670
        x1:1 | x2:2 | y1:0 | y2:4 | z1:0 | z2:3 | density:1000

    You can use :meth:`~fatiando.msh.ddd.PrismMesh.get_xs` (and similar
    methods for y and z) to get the x coordinates os the prisms in the mesh::

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2))
        >>> print mesh.get_xs()
        [ 0.  1.  2.]
        >>> print mesh.get_ys()
        [ 0.  4.]
        >>> print mesh.get_zs()
        [ 0.  3.]

    """

    def __init__(self, bounds, shape, props=None):
        object.__init__(self)
        log.info("Generating 3D right rectangular prism mesh:")
        nz, ny, nx = shape
        size = int(nx*ny*nz)
        x1, x2, y1, y2, z1, z2 = bounds
        dx = float(x2 - x1)/nx
        dy = float(y2 - y1)/ny
        dz = float(z2 - z1)/nz
        self.shape = tuple(int(i) for i in shape)
        self.size = size
        self.dims = (dx, dy, dz)
        self.bounds = bounds
        if props is None:
            self.props = {}
        else:
            self.props = props
        log.info("  bounds = (x1, x2, y1, y2, z1, z2) = %s" % (str(bounds)))
        log.info("  shape = (nz, ny, nx) = %s" % (str(shape)))
        log.info("  number of prisms = %d" % (size))
        log.info("  prism dimensions = (dx, dy, dz) = %s" % (str(self.dims)))
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked prisms. Will return None if trying to access them
        self.mask = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        nz, ny, nx = self.shape
        k = index/(nx*ny)
        j = (index - k*(nx*ny))/nx
        i = (index - k*(nx*ny) - j*nx)
        x1 = self.bounds[0] + self.dims[0]*i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1]*j
        y2 = y1 + self.dims[1]
        z1 = self.bounds[4] + self.dims[2]*k
        z2 = z1 + self.dims[2]
        props = dict([p, self.props[p][index]] for p in self.props)
        return Prism(x1, x2, y1, y2, z1, z2, props=props)

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the cells in the mesh.

        Different physical properties of the mesh are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each prism of the mesh. For the
            ordering of prisms in the mesh see
            :class:`~fatiando.msh.ddd.PrismMesh`

        """
        self.props[prop] = values

    def carvetopo(self, x, y, height):
        """
        Mask (remove) prisms from the mesh that are above the topography.

        Accessing the ith prism will return None if it was masked (above the
        topography).
        Also mask prisms outside of the topography grid provided.
        The topography height information does not need to be on a regular grid,
        it will be interpolated.

        Parameters:

        * x, y : lists
            x and y coordinates of the grid points
        * height : list or array
            Array with the height of the topography

        """
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        # The coordinates of the centers of the cells
        xc = numpy.arange(x1, x2, dx) + 0.5*dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = numpy.arange(y1, y2, dy) + 0.5*dy
        if len(yc) > ny:
            yc = yc[:-1]
        zc = numpy.arange(z1, z2, dz) + 0.5*dz
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = numpy.meshgrid(xc, yc)
        # -1 if to transform height into z coordinate
        topo = -1*matplotlib.mlab.griddata(x, y, height, XC, YC).ravel()
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # bellow a masked topo point (ie, one with no height information)
        if numpy.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in xrange(len(topo))]
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
                if masked or cellz < h:
                    self.mask.append(c)
                c += 1

    def get_xs(self):
        """
        Return an array with the x coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        xs = numpy.arange(x1, x2 + dx, dx)
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
        ys = numpy.arange(y1, y2 + dy, dy)
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
        zs = numpy.arange(z1, z2 + dz, dz)
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

        * prisms : list of :class:`~fatiando.msh.ddd.Prism`
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
        start = i*nx*ny
        end = (i + 1)*nx*ny
        layer = [self.__getitem__(p) for p in xrange(start, end)]
        return layer

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
        values = (v if i not in self.mask else -10000000
                  for i, v in enumerate(self.props[prop]))
        numpy.savetxt(
            propfile,
            numpy.ravel(numpy.reshape(numpy.fromiter(values, 'f'),
                self.shape), order='F'),
            fmt='%.4f')

def extract(prop, prisms):
    """
    Extract the values of a physical property from the cells in a list.

    If a cell is `None` or doesn't have the physical property, a value of `None`
    will be put in it's place.

    Parameters:

    * prop : str
        The name of the physical property to extract
    * cells : list
        A list of cells (e.g., :class:`~fatiando.msh.ddd.Prism`,
        :class:`~fatiando.msh.ddd.PolygonalPrism`, etc)

    Returns:

    * values : array
        The extracted values

    Examples:

        >>> cells = [Prism(1, 2, 3, 4, 5, 6, {'foo':1}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':10}),
        ...          None,
        ...          Prism(1, 2, 3, 4, 5, 6, {'bar':2000})]
        >>> print extract('foo', cells)
        [1, 10, None, None]

    """
    def getprop(p):
        if p is None or prop not in p.props:
            return None
        return p.props[prop]
    return [getprop(p) for p in prisms]

def vfilter(vmin, vmax, prop, cells):
    """
    Remove cells whose physical property value falls outside a given range.

    If a cell is `None` or doesn't have the physical property, it will be not
    be included in the result.

    Parameters:

    * vmin : float
        Minimum value
    * vmax : float
        Maximum value
    * prop : str
        The name of the physical property used to filter
    * cells : list
        A list of cells (e.g., :class:`~fatiando.msh.ddd.Prism`,
        :class:`~fatiando.msh.ddd.PolygonalPrism`, etc)

    Returns:

    * filtered : list
        The cells that fall within the desired range

    Examples:

        >>> cells = [Prism(1, 2, 3, 4, 5, 6, {'foo':1}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':20}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':3}),
        ...          None,
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':4}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':200}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'bar':1000})]
        >>> for cell in vfilter(0, 10, 'foo', cells):
        ...     print cell
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:1
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:3
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:4

    """
    def isin(cell):
        if (cell is None or prop not in cell.props or cell.props[prop] < vmin
            or cell.props[prop] > vmax):
            return False
        return True
    return [c for c in cells if isin(c)]

def vremove(value, prop, cells):
    """
    Remove the cells with a given physical property value.

    If a cell is `None` it will be not be included in the result.

    If a cell doesn't have the physical property, it will be included in the
    result.

    Parameters:

    * value : float
        The value of the physical property to remove
    * prop : str
        The name of the physicaRemove cells whose physical property value falls outside a given rangel property
    * cells : list
        A list of cells (e.g., :class:`~fatiando.msh.ddd.Prism`,
        :class:`~fatiando.msh.ddd.PolygonalPrism`, etc)

    Returns:

    * removed : list
        A list of cells that have *prop* != *value*

    Examples:

        >>> cells = [Prism(1, 2, 3, 4, 5, 6, {'foo':1}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':20}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':3}),
        ...          None,
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':1}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'foo':200}),
        ...          Prism(1, 2, 3, 4, 5, 6, {'bar':1000})]
        >>> for cell in vremove(1, 'foo', cells):
        ...     print cell
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:20
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:3
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | foo:200
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | bar:1000

    """
    removed = [c for c in cells
        if c is not None and (prop not in c.props or c.props[prop] != value)]
    return removed
