"""
Generate and operate on various kinds of meshes and geometric elements

**Geometric elements**

* :class:`~fatiando.mesher.Polygon`
* :class:`~fatiando.mesher.Square`
* :class:`~fatiando.mesher.Prism`
* :class:`~fatiando.mesher.PolygonalPrism`
* :class:`~fatiando.mesher.Sphere`
* :class:`~fatiando.mesher.Tesseroid`

**Meshes**

* :class:`~fatiando.mesher.SquareMesh`
* :class:`~fatiando.mesher.PrismMesh`
* :class:`~fatiando.mesher.PrismRelief`
* :class:`~fatiando.mesher.TesseroidMesh`
* :class:`~fatiando.mesher.PointGrid`

**Utility functions**

* :func:`~fatiando.mesher.extract`: Extract the values of a physical
  property from the cells in a list
* :func:`~fatiando.mesher.vfilter`: Remove cells whose physical property
  value falls outside a given range
* :func:`~fatiando.mesher.vremove`: Remove the cells with a given physical
  property value

----

"""
from __future__ import division
import numpy
import scipy.special
import scipy.interpolate

from . import gridder
from . import utils


class GeometricElement(object):

    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value


class Polygon(GeometricElement):

    """
    Create a polygon object.

    .. note:: Most applications require the vertices to be **clockwise**!

    Parameters:

    * vertices : list of lists
        List of [x, y] pairs with the coordinates of the vertices.
    * props : dict
        Physical properties assigned to the polygon.
        Ex: ``props={'density':10, 'susceptibility':10000}``

    Examples::

        >>> poly = Polygon([[0, 0], [1, 4], [2, 5]], {'density': 500})
        >>> poly.props
        {'density': 500}
        >>> poly.nverts
        3
        >>> poly.vertices
        array([[0, 0],
               [1, 4],
               [2, 5]])
        >>> poly.x
        array([0, 1, 2])
        >>> poly.y
        array([0, 4, 5])

    """

    def __init__(self, vertices, props=None):
        super(Polygon, self).__init__(props)
        self._vertices = numpy.asarray(vertices)

    @property
    def vertices(self):
        return self._vertices

    @property
    def nverts(self):
        return len(self.vertices)

    @property
    def x(self):
        return self.vertices[:, 0]

    @property
    def y(self):
        return self.vertices[:, 1]


class Square(Polygon):

    """
    Create a square object.

    Parameters:

    * bounds : list = [x1, x2, y1, y2]
        Coordinates of the top right and bottom left corners of the square
    * props : dict
        Physical properties assigned to the square.
        Ex: ``props={'density':10, 'slowness':10000}``

    Example::

        >>> sq = Square([0, 1, 2, 4], {'density': 750})
        >>> sq.bounds
        [0, 1, 2, 4]
        >>> sq.x1
        0
        >>> sq.x2
        1
        >>> sq.props
        {'density': 750}
        >>> sq.addprop('magnetization', 100)
        >>> sq.props['magnetization']
        100

    A square can be used as a :class:`~fatiando.mesher.Polygon`::

        >>> sq.vertices
        array([[0, 2],
               [1, 2],
               [1, 4],
               [0, 4]])
        >>> sq.x
        array([0, 1, 1, 0])
        >>> sq.y
        array([2, 2, 4, 4])
        >>> sq.nverts
        4

    """

    def __init__(self, bounds, props=None):
        super(Square, self).__init__(None, props)
        self.x1, self.x2, self.y1, self.y2 = bounds

    @property
    def bounds(self):
        """
        The x, y boundaries of the square as [xmin, xmax, ymin, ymax]
        """
        return [self.x1, self.x2, self.y1, self.y2]

    @property
    def vertices(self):
        """
        The vertices of the square.
        """
        verts = numpy.array(
            [[self.x1, self.y1],
             [self.x2, self.y1],
             [self.x2, self.y2],
             [self.x1, self.y2]])
        return verts

    def __str__(self):
        """Return a string representation of the square."""
        names = [('x1', self.x1), ('x2', self.x2), ('y1', self.y1),
                 ('y2', self.y2)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)


class SquareMesh(object):

    """
    Generate a 2D regular mesh of squares.

    For all purposes, :class:`~fatiando.mesher.SquareMesh` can be used as a
    list of :class:`~fatiando.mesher.Square`. The order of the squares in the
    list is: x directions varies first, then y.

    Parameters:

    * bounds :  list = [x1, x2, y1, y2]
        Boundaries of the mesh
    * shape : tuple = (ny, nx)
        Number of squares in the y and x dimension, respectively
    * props : dict
        Physical properties of each square in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each square of the mesh.

    Examples:

        >>> mesh = SquareMesh((0, 4, 0, 6), (2, 2))
        >>> for s in mesh:
        ...     print s
        x1:0 | x2:2 | y1:0 | y2:3
        x1:2 | x2:4 | y1:0 | y2:3
        x1:0 | x2:2 | y1:3 | y2:6
        x1:2 | x2:4 | y1:3 | y2:6
        >>> print mesh[1]
        x1:2 | x2:4 | y1:0 | y2:3
        >>> print mesh[-1]
        x1:2 | x2:4 | y1:3 | y2:6

    With physical properties::

        >>> mesh = SquareMesh((0, 4, 0, 6), (2, 1), {'slowness':[3.4, 8.6]})
        >>> for s in mesh:
        ...     print s
        x1:0 | x2:4 | y1:0 | y2:3 | slowness:3.4
        x1:0 | x2:4 | y1:3 | y2:6 | slowness:8.6

    Or::

        >>> mesh = SquareMesh((0, 4, 0, 6), (2, 1))
        >>> mesh.addprop('slowness', [3.4, 8.6])
        >>> for s in mesh:
        ...     print s
        x1:0 | x2:4 | y1:0 | y2:3 | slowness:3.4
        x1:0 | x2:4 | y1:3 | y2:6 | slowness:8.6

    """

    def __init__(self, bounds, shape, props=None):
        object.__init__(self)
        ny, nx = shape
        size = int(nx * ny)
        x1, x2, y1, y2 = bounds
        dx = (x2 - x1)/nx
        dy = (y2 - y1)/ny
        self.bounds = bounds
        self.shape = tuple(int(i) for i in shape)
        self.size = size
        self.dims = (dx, dy)
        # props has to be None, not {} by default because {} would be permanent
        # for all instaces of the class (like a class variable) and changes
        # to one instace would lead to changes in another (and a huge mess)
        if props is None:
            self.props = {}
        else:
            self.props = props
        # The index of the current square in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked squares. Will return None if trying to access them
        self.mask = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        ny, nx = self.shape
        j = index//nx
        i = index - j*nx
        x1 = self.bounds[0] + self.dims[0] * i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1] * j
        y2 = y1 + self.dims[1]
        props = dict([p, self.props[p][index]] for p in self.props)
        return Square((x1, x2, y1, y2), props=props)

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.size:
            raise StopIteration
        square = self.__getitem__(self.i)
        self.i += 1
        return square

    def addprop(self, prop, values):
        """
        Add physical property values to the cells in the mesh.

        Different physical properties of the mesh are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property
        * values : list or array
            The value of this physical property in each square of the mesh.
            For the ordering of squares in the mesh see
            :class:`~fatiando.mesher.SquareMesh`

        """
        self.props[prop] = values

    def img2prop(self, fname, vmin, vmax, prop):
        """
        Load the physical property value from an image file.

        The image is converted to gray scale and the gray intensity of each
        pixel is used to set the value of the physical property of the
        cells in the mesh. Gray intensity values are scaled to the range
        ``[vmin, vmax]``.

        If the shape of image (number of pixels in y and x) is different from
        the shape of the mesh, the image will be interpolated to match the
        shape of the mesh.

        Parameters:

        * fname : str
            Name of the image file
        * vmax, vmin : float
            Range of physical property values (used to convert the gray scale
            to physical property values)
        * prop : str
            Name of the physical property

        """
        self.props[prop] = utils.fromimage(fname, ranges=[vmin, vmax],
                                           shape=self.shape)[::-1, :].ravel()

    def get_xs(self):
        """
        Get a list of the x coordinates of the corners of the cells in the
        mesh.

        If the mesh has nx cells, get_xs() will return nx + 1 values.
        """
        dx, dy = self.dims
        x1, x2, y1, y2 = self.bounds
        ny, nx = self.shape
        xs = numpy.arange(x1, x2 + dx, dx, 'f')
        if len(xs) == nx + 2:
            return xs[0:-1]
        elif len(xs) == nx:
            xs = xs.tolist()
            xs.append(x2)
            return numpy.array(xs)
        else:
            return xs

    def get_ys(self):
        """
        Get a list of the y coordinates of the corners of the cells in the
        mesh.

        If the mesh has ny cells, get_ys() will return ny + 1 values.
        """
        dx, dy = self.dims
        x1, x2, y1, y2 = self.bounds
        ny, nx = self.shape
        ys = numpy.arange(y1, y2, dy, 'f')
        if len(ys) == ny + 2:
            return ys[0:-1]
        elif len(ys) == ny:
            ys = ys.tolist()
            ys.append(y2)
            return numpy.array(ys)
        else:
            return ys


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

        >>> from fatiando.mesher import Prism
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
            [ 1.5  2.   1. ]

        """
        xc = 0.5 * (self.x1 + self.x2)
        yc = 0.5 * (self.y1 + self.y2)
        zc = 0.5 * (self.z1 + self.z2)
        return numpy.array([xc, yc, zc])


class Tesseroid(GeometricElement):

    """
    Create a tesseroid (spherical prism).

    Parameters:

    * w, e : float
        West and east borders of the tesseroid in decimal degrees
    * s, n : float
        South and north borders of the tesseroid in decimal degrees
    * top, bottom : float
        Bottom and top of the tesseroid with respect to the mean earth radius
        in meters. Ex: if the top is 100 meters above the mean earth radius,
        ``top=100``, if 100 meters below ``top=-100``.
    * props : dict
        Physical properties assigned to the tesseroid.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> from fatiando.mesher import Tesseroid
        >>> t = Tesseroid(1, 2, 3, 4, 6, 5, {'density':200})
        >>> t.props['density']
        200
        >>> print t.get_bounds()
        [1.0, 2.0, 3.0, 4.0, 6.0, 5.0]
        >>> print t
        w:1 | e:2 | s:3 | n:4 | top:6 | bottom:5 | density:200
        >>> t = Tesseroid(1, 2, 3, 4, 6, 5)
        >>> print t
        w:1 | e:2 | s:3 | n:4 | top:6 | bottom:5
        >>> t.addprop('density', 2670)
        >>> print t
        w:1 | e:2 | s:3 | n:4 | top:6 | bottom:5 | density:2670

    """

    def __init__(self, w, e, s, n, top, bottom, props=None):
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
                 ('n', self.n), ('top', self.top), ('bottom', self.bottom)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the tesseroid (i.e., the borders).

        Returns:

        * bounds : list
            ``[w, e, s, n, top, bottom]``, the bounds of the tesseroid

        Examples:

            >>> t = Tesseroid(1, 2, 3, 4, 6, 5)
            >>> print t.get_bounds()
            [1.0, 2.0, 3.0, 4.0, 6.0, 5.0]

        """
        return [self.w, self.e, self.s, self.n, self.top, self.bottom]

    def half(self, lon=True, lat=True, r=True):
        """
        Divide the tesseroid in 2 halfs for each dimension (total 8)

        The smaller tesseroids will share the large one's props.

        Parameters:

        * lon, lat, r : True or False
            Dimensions along which the tesseroid will be split in half.

        Returns:

        * tesseroids : list
            A list of maximum 8 tesseroids that make up the larger one.

        Examples::

            >>> tess = Tesseroid(-10, 10, -20, 20, 0, -40, {'density':2})
            >>> split = tess.half()
            >>> print len(split)
            8
            >>> for t in split:
            ...     print t
            w:-10 | e:0 | s:-20 | n:0 | top:-20 | bottom:-40 | density:2
            w:-10 | e:0 | s:-20 | n:0 | top:0 | bottom:-20 | density:2
            w:-10 | e:0 | s:0 | n:20 | top:-20 | bottom:-40 | density:2
            w:-10 | e:0 | s:0 | n:20 | top:0 | bottom:-20 | density:2
            w:0 | e:10 | s:-20 | n:0 | top:-20 | bottom:-40 | density:2
            w:0 | e:10 | s:-20 | n:0 | top:0 | bottom:-20 | density:2
            w:0 | e:10 | s:0 | n:20 | top:-20 | bottom:-40 | density:2
            w:0 | e:10 | s:0 | n:20 | top:0 | bottom:-20 | density:2
            >>> tess = Tesseroid(-15, 15, -20, 20, 0, -40)
            >>> split = tess.half(lat=False)
            >>> print len(split)
            4
            >>> for t in split:
            ...     print t
            w:-15 | e:0 | s:-20 | n:20 | top:-20 | bottom:-40
            w:-15 | e:0 | s:-20 | n:20 | top:0 | bottom:-20
            w:0 | e:15 | s:-20 | n:20 | top:-20 | bottom:-40
            w:0 | e:15 | s:-20 | n:20 | top:0 | bottom:-20

        """
        dlon = 0.5 * (self.e - self.w)
        dlat = 0.5 * (self.n - self.s)
        dh = 0.5 * (self.top - self.bottom)
        wests = [self.w, self.w + dlon]
        souths = [self.s, self.s + dlat]
        bottoms = [self.bottom, self.bottom + dh]
        if not lon:
            dlon *= 2
            wests.pop()
        if not lat:
            dlat *= 2
            souths.pop()
        if not r:
            dh *= 2
            bottoms.pop()
        split = [
            Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=self.props)
            for i in wests for j in souths for k in bottoms]
        return split

    def split(self, nlon, nlat, nh):
        """
        Split the tesseroid into smaller ones.

        The smaller tesseroids will share the large one's props.

        Parameters:

        * nlon, nlat, nh : int
            The number of sections to split in the longitudinal, latitudinal,
            and vertical dimensions

        Returns:

        * tesseroids : list
            A list of nlon*nlat*nh tesseroids that make up the larger one.


        Examples::

            >>> tess = Tesseroid(-10, 10, -20, 20, 0, -40, {'density':2})
            >>> split = tess.split(1, 2, 2)
            >>> print len(split)
            4
            >>> for t in split:
            ...     print t
            w:-10 | e:10 | s:-20 | n:0 | top:-20 | bottom:-40 | density:2
            w:-10 | e:10 | s:-20 | n:0 | top:0 | bottom:-20 | density:2
            w:-10 | e:10 | s:0 | n:20 | top:-20 | bottom:-40 | density:2
            w:-10 | e:10 | s:0 | n:20 | top:0 | bottom:-20 | density:2
            >>> tess = Tesseroid(-15, 15, -20, 20, 0, -40)
            >>> split = tess.split(3, 1, 1)
            >>> print len(split)
            3
            >>> for t in split:
            ...     print t
            w:-15 | e:-5 | s:-20 | n:20 | top:0 | bottom:-40
            w:-5 | e:5 | s:-20 | n:20 | top:0 | bottom:-40
            w:5 | e:15 | s:-20 | n:20 | top:0 | bottom:-40

        """
        wests = numpy.linspace(self.w, self.e, nlon + 1)
        souths = numpy.linspace(self.s, self.n, nlat + 1)
        bottoms = numpy.linspace(self.bottom, self.top, nh + 1)
        dlon = wests[1] - wests[0]
        dlat = souths[1] - souths[0]
        dh = bottoms[1] - bottoms[0]
        tesseroids = [
            Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=self.props)
            for i in wests[:-1] for j in souths[:-1] for k in bottoms[:-1]]
        return tesseroids


class Sphere(GeometricElement):

    """
    Create a sphere.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the sphere
    * radius : float
        The radius of the sphere
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
        self.center = numpy.array([x, y, z])

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
        self.x = numpy.fromiter((v[0] for v in vertices), dtype=numpy.float)
        self.y = numpy.fromiter((v[1] for v in vertices), dtype=numpy.float)
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.nverts = len(vertices)

    def topolygon(self):
        """
        Get the polygon describing the prism viewed from above.

        Returns:

        * polygon : :func:`fatiando.mesher.Polygon`
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


class PointGrid(object):
    """
    Create a regular grid of 3D point sources (spheres of unit volume).

    Use this as a 1D list of :class:`~fatiando.mesher.Sphere`.

    Grid points are ordered like a C matrix, first each row in a column, then
    change columns. In this case, the x direction (North-South) are the rows
    and y (East-West) are the columns.

    Parameters:

    * area : list = [x1, x2, y1, y2]
        The area where the grid will be spread out
    * z : float or 1d-array
        The z coordinates of each point in the grid (remember, z is positive downward).
    * shape : tuple = (nx, ny)
        The number of points in the x and y directions
    * props :  dict
        Physical properties of each point in the grid.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property for
        each point in the grid.

    Examples::

        >>> g = PointGrid([0, 10, 2, 6], 200, (2, 3))
        >>> g.shape
        (2, 3)
        >>> g.size
        6
        >>> g[0].center
        array([   0.,    2.,  200.])
        >>> g[-1].center
        array([  10.,    6.,  200.])
        >>> for p in g:
        ...     p.center
        array([   0.,    2.,  200.])
        array([   0.,    4.,  200.])
        array([   0.,    6.,  200.])
        array([  10.,    2.,  200.])
        array([  10.,    4.,  200.])
        array([  10.,    6.,  200.])
        >>> g.x.reshape(g.shape)
        array([[  0.,   0.,   0.],
               [ 10.,  10.,  10.]])
        >>> g.y.reshape(g.shape)
        array([[ 2.,  4.,  6.],
               [ 2.,  4.,  6.]])
        >>> g.dx, g.dy
        (10.0, 2.0)

    """

    def __init__(self, area, z, shape, props=None):
        object.__init__(self)
        self.area = area
        self.shape = shape
        if props is None:
            self.props = {}
        else:
            self.props = props
        nx, ny = shape
        self.size = nx*ny
        self.z = numpy.zeros(self.size) + z
        self.radius = scipy.special.cbrt(3. / (4. * numpy.pi))
        self.x, self.y = gridder.regular(area, shape)
        # The spacing between points
        self.dx, self.dy = gridder.spacing(area, shape)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < -self.size:
            raise IndexError('grid index out of range')
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        props = dict([p, self.props[p][index]] for p in self.props)
        sphere = Sphere(self.x[index], self.y[index], self.z[index], self.radius,
                        props=props)
        return sphere

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.size:
            raise StopIteration
        sphere = self.__getitem__(self.i)
        self.i += 1
        return sphere

    def addprop(self, prop, values):
        """
        Add physical property values to the points in the grid.

        Different physical properties of the grid are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each point of the grid

        """
        self.props[prop] = values

    def split(self, shape):
        """
        Divide the grid into subgrids.

        .. note::

            Remember that x is the North-South direction and y is East-West.

        Parameters:

        * shape : tuple = (nx, ny)
            Number of subgrids in the x and y directions, respectively.

        Returns:

        * subgrids : list
            List of :class:`~fatiando.mesher.PointGrid`

        Examples::

            >>> import numpy
            >>> z = numpy.linspace(0, 1100, 12)
            >>> g = PointGrid((0, 3, 0, 2), z, (4, 3))
            >>> g.addprop('bla', [1,   2,  3,
            ...                   4,   5,  6,
            ...                   7,   8,  9,
            ...                   10, 11, 12])
            >>> grids = g.split((2, 3))
            >>> for s in grids:
            ...     s.props['bla']
            array([1, 4])
            array([2, 5])
            array([3, 6])
            array([ 7, 10])
            array([ 8, 11])
            array([ 9, 12])
            >>> for s in grids:
            ...     s.x
            array([ 0.,  1.])
            array([ 0.,  1.])
            array([ 0.,  1.])
            array([ 2.,  3.])
            array([ 2.,  3.])
            array([ 2.,  3.])
            >>> for s in grids:
            ...     s.y
            array([ 0.,  0.])
            array([ 1.,  1.])
            array([ 2.,  2.])
            array([ 0.,  0.])
            array([ 1.,  1.])
            array([ 2.,  2.])
            >>> for s in grids:
            ...     s.z
            array([   0.,  300.])
            array([ 100.,  400.])
            array([ 200.,  500.])
            array([ 600.,  900.])
            array([  700.,  1000.])
            array([  800.,  1100.])

        """
        nx, ny = shape
        totalx, totaly = self.shape
        if totalx % nx != 0 or totaly % ny != 0:
            raise ValueError(
                'Cannot split! nx and ny must be divisible by grid shape')
        x1, x2, y1, y2 = self.area
        xs = numpy.linspace(x1, x2, totalx)
        ys = numpy.linspace(y1, y2, totaly)
        mx, my = (totalx//nx, totaly//ny)
        dx, dy = self.dx*(mx - 1), self.dy*(my - 1)
        subs = []
        for i, xstart in enumerate(xs[::mx]):
            for j, ystart in enumerate(ys[::my]):
                area = [xstart, xstart + dx, ystart, ystart + dy]
                props = {}
                for p in self.props:
                    pmatrix = numpy.reshape(self.props[p], self.shape)
                    props[p] = pmatrix[i*mx:(i + 1)*mx,
                                       j*my:(j + 1)*my].ravel()
                zmatrix = numpy.reshape(self.z, self.shape)
                zs = zmatrix[i*mx:(i + 1)*mx,
                             j*my:(j + 1)*my].ravel()
                subs.append(PointGrid(area, zs, (mx, my), props))
        return subs


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
    asks for a list of prisms, like :func:`fatiando.gravmag.prism.gz`.

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
            raise ValueError(
                "nodes has x, y, z coordinate arrays of different lengths")
        self.x, self.y, self.z = x, y, z
        self.size = len(x)
        self.ref = ref
        self.dy, self.dx = dims
        self.props = {}
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
        x1 = xc - 0.5 * self.dx
        x2 = xc + 0.5 * self.dx
        y1 = yc - 0.5 * self.dy
        y2 = yc + 0.5 * self.dy
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

        .. warning:: If the z value of any point in the relief is below the
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

    Prisms are ordered as follows: first layers (z coordinate),
    then EW rows (y) and finaly x coordinate (NS).

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has z
    index 1 (second layer), y index 1 (second row), and x index 2 (third
    element in the column).

    :class:`~fatiando.mesher.PrismMesh` can used as list of prisms. It acts
    as an iteratior (so you can loop over prisms). It also has a
    ``__getitem__`` method to access individual elements in the mesh.
    In practice, :class:`~fatiando.mesher.PrismMesh` should be able to be
    passed to any function that asks for a list of prisms, like
    :func:`fatiando.gravmag.prism.gz`.

    To make the mesh incorporate a topography, use
    :meth:`~fatiando.mesher.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * shape : tuple = (nz, ny, nx)
        Number of prisms in the x, y, and z directions.
    * props :  dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.

    Examples:

        >>> from fatiando.mesher import PrismMesh
        >>> mesh = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
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

    or equivalently::

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2))
        >>> mesh.addprop('density', [200, -1000.0])
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:1 | y1:0 | y2:4 | z1:0 | z2:3 | density:200
        x1:1 | x2:2 | y1:0 | y2:4 | z1:0 | z2:3 | density:-1000

    You can use :meth:`~fatiando.mesher.PrismMesh.get_xs` (and similar
    methods for y and z) to get the x coordinates of the prisms in the mesh::

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2))
        >>> print mesh.get_xs()
        [ 0.  1.  2.]
        >>> print mesh.get_ys()
        [ 0.  4.]
        >>> print mesh.get_zs()
        [ 0.  3.]

    The ``shape`` of the mesh must be integer!

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2.5))
        Traceback (most recent call last):
            ...
        AttributeError: Invalid mesh shape (1, 1, 2.5). shape must be integers

    """

    celltype = Prism

    def __init__(self, bounds, shape, props=None):
        object.__init__(self)
        nz, ny, nx = shape
        if (not isinstance(nx, int) or not isinstance(ny, int)
                or not isinstance(nz, int)):
            raise AttributeError(
                'Invalid mesh shape {}. shape must be integers'.format(
                    str(shape)))
        size = int(nx * ny * nz)
        x1, x2, y1, y2, z1, z2 = bounds
        dx = (x2 - x1)/nx
        dy = (y2 - y1)/ny
        dz = (z2 - z1)/nz
        self.shape = tuple(int(i) for i in shape)
        self.size = size
        self.dims = (dx, dy, dz)
        self.bounds = bounds
        if props is None:
            self.props = {}
        else:
            self.props = props
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked prisms. Will return None if trying to access them
        self.mask = []
        # Wether or not to change heights to z coordinate
        self.zdown = True

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < -self.size:
            raise IndexError('mesh index out of range')
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        nz, ny, nx = self.shape
        k = index//(nx*ny)
        j = (index - k*(nx*ny))//nx
        i = (index - k*(nx*ny) - j*nx)
        x1 = self.bounds[0] + self.dims[0] * i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1] * j
        y2 = y1 + self.dims[1]
        z1 = self.bounds[4] + self.dims[2] * k
        z2 = z1 + self.dims[2]
        props = dict([p, self.props[p][index]] for p in self.props)
        return self.celltype(x1, x2, y1, y2, z1, z2, props=props)

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
            :class:`~fatiando.mesher.PrismMesh`

        """
        self.props[prop] = values

    def carvetopo(self, x, y, height):
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

        """
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        # The coordinates of the centers of the cells
        xc = numpy.arange(x1, x2, dx) + 0.5 * dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = numpy.arange(y1, y2, dy) + 0.5 * dy
        if len(yc) > ny:
            yc = yc[:-1]
        zc = numpy.arange(z1, z2, dz) + 0.5 * dz
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = numpy.meshgrid(xc, yc)
        topo = scipy.interpolate.griddata((x, y), height, (XC, YC),
                                          method='cubic').ravel()
        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # below a masked topo point (ie, one with no height information)
        if numpy.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in xrange(len(topo))]
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
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
        values = numpy.fromiter(self.props[prop], dtype=numpy.float)
        # Replace the masked cells with a dummy value
        values[self.mask] = -10000000
        reordered = numpy.ravel(numpy.reshape(values, self.shape), order='F')
        numpy.savetxt(propfile, reordered, fmt='%.4f')


class TesseroidMesh(PrismMesh):

    """
    Generate a 3D regular mesh of tesseroids.

    Tesseroids are ordered as follows: first layers (height coordinate),
    then N-S rows and finaly E-W.

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has height
    index 1 (second layer), y index 1 (second row), and x index 2 (
    third element in the column).

    This class can used as list of tesseroids. It acts
    as an iteratior (so you can loop over tesseroids).
    It also has a ``__getitem__``
    method to access individual elements in the mesh.
    In practice, it should be able to be
    passed to any function that asks for a list of tesseroids, like
    :func:`fatiando.gravmag.tesseroid.gz`.

    To make the mesh incorporate a topography, use
    :meth:`~fatiando.mesher.TesseroidMesh.carvetopo`

    Parameters:

    * bounds : list = [w, e, s, n, top, bottom]
        Boundaries of the mesh. ``w, e, s, n`` in degrees, ``top`` and
        ``bottom`` are heights (positive upward) and in meters.
    * shape : tuple = (nr, nlat, nlon)
        Number of tesseroids in the radial, latitude, and longitude directions.
    * props :  dict
        Physical properties of each tesseroid in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each tesseroid of the mesh.

    Examples:

        >>> from fatiando.mesher import TesseroidMesh
        >>> mesh = TesseroidMesh((0, 1, 0, 2, 3, 0), (1, 2, 2))
        >>> for p in mesh:
        ...     print p
        w:0 | e:0.5 | s:0 | n:1 | top:3 | bottom:0
        w:0.5 | e:1 | s:0 | n:1 | top:3 | bottom:0
        w:0 | e:0.5 | s:1 | n:2 | top:3 | bottom:0
        w:0.5 | e:1 | s:1 | n:2 | top:3 | bottom:0
        >>> print mesh[0]
        w:0 | e:0.5 | s:0 | n:1 | top:3 | bottom:0
        >>> print mesh[-1]
        w:0.5 | e:1 | s:1 | n:2 | top:3 | bottom:0

    One with physical properties::

        >>> props = {'density':[2670.0, 1000.0]}
        >>> mesh = TesseroidMesh((0, 2, 0, 4, 3, 0), (1, 1, 2), props=props)
        >>> for p in mesh:
        ...     print p
        w:0 | e:1 | s:0 | n:4 | top:3 | bottom:0 | density:2670
        w:1 | e:2 | s:0 | n:4 | top:3 | bottom:0 | density:1000

    or equivalently::

        >>> mesh = TesseroidMesh((0, 2, 0, 4, 3, 0), (1, 1, 2))
        >>> mesh.addprop('density', [200, -1000.0])
        >>> for p in mesh:
        ...     print p
        w:0 | e:1 | s:0 | n:4 | top:3 | bottom:0 | density:200
        w:1 | e:2 | s:0 | n:4 | top:3 | bottom:0 | density:-1000

    You can use :meth:`~fatiando.mesher.PrismMesh.get_xs` (and similar
    methods for y and z) to get the x coordinates of the tesseroidss in the
    mesh::

        >>> mesh = TesseroidMesh((0, 2, 0, 4, 3, 0), (1, 1, 2))
        >>> print mesh.get_xs()
        [ 0.  1.  2.]
        >>> print mesh.get_ys()
        [ 0.  4.]
        >>> print mesh.get_zs()
        [ 3.  0.]

    You can iterate over the layers of the mesh::

        >>> mesh = TesseroidMesh((0, 2, 0, 2, 2, 0), (2, 2, 2))
        >>> for layer in mesh.layers():
        ...     for p in layer:
        ...         print p
        w:0 | e:1 | s:0 | n:1 | top:2 | bottom:1
        w:1 | e:2 | s:0 | n:1 | top:2 | bottom:1
        w:0 | e:1 | s:1 | n:2 | top:2 | bottom:1
        w:1 | e:2 | s:1 | n:2 | top:2 | bottom:1
        w:0 | e:1 | s:0 | n:1 | top:1 | bottom:0
        w:1 | e:2 | s:0 | n:1 | top:1 | bottom:0
        w:0 | e:1 | s:1 | n:2 | top:1 | bottom:0
        w:1 | e:2 | s:1 | n:2 | top:1 | bottom:0

    The ``shape`` of the mesh must be integer!

        >>> mesh = TesseroidMesh((0, 2, 0, 4, 0, 3), (1, 1, 2.5))
        Traceback (most recent call last):
            ...
        AttributeError: Invalid mesh shape (1, 1, 2.5). shape must be integers

    """

    celltype = Tesseroid

    def __init__(self, bounds, shape, props=None):
        PrismMesh.__init__(self, bounds, shape, props)
        self.zdown = False
        self.dump = None


def extract(prop, prisms):
    """
    Extract the values of a physical property from the cells in a list.

    If a cell is ``None`` or doesn't have the physical property, a value of
    ``None`` will be put in it's place.

    Parameters:

    * prop : str
        The name of the physical property to extract
    * cells : list
        A list of cells (e.g., :class:`~fatiando.mesher.Prism`,
        :class:`~fatiando.mesher.PolygonalPrism`, etc)

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
        A list of cells (e.g., :class:`~fatiando.mesher.Prism`,
        :class:`~fatiando.mesher.PolygonalPrism`, etc)

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
        if cell is None or prop not in cell.props:
            return False
        value = cell.props[prop]
        if not isinstance(value, float) and not isinstance(value, int):
            value = numpy.linalg.norm(cell.props[prop])
        if value < vmin or value > vmax:
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
        The value of the physical property to remove. If the physical property
        is a vector, will compare the norm of the vector to **value**.
    * prop : str
        The name of the physical property to remove
    * cells : list
        A list of cells (e.g., :class:`~fatiando.mesher.Prism`,
        :class:`~fatiando.mesher.PolygonalPrism`, etc)

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
    def keep(cell):
        if cell is None:
            return False
        if prop not in cell.props:
            return True
        p = cell.props[prop]
        if not isinstance(p, float) and not isinstance(p, int):
            p = numpy.linalg.norm(cell.props[prop])
        if p != value:
            return True
        return False
    return [c for c in cells if keep(c)]
