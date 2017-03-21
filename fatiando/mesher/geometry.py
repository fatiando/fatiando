"""
Defines geometric primitives like prisms, spheres, etc.
"""
from __future__ import division, absolute_import
from future.builtins import object, super
import copy as cp
import numpy as np


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

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)


class Polygon(GeometricElement):
    """
    A polygon object (2D).

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
        super().__init__(props)
        self._vertices = np.asarray(vertices)

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
    A square object (2D).

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
        super().__init__(None, props)
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
        verts = np.array(
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


class Prism(GeometricElement):
    """
    A 3D right rectangular prism.

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
        super().__init__(props)
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
        return np.array([xc, yc, zc])


class Tesseroid(GeometricElement):
    """
    A tesseroid (spherical prism).

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
        super().__init__(props)
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
        wests = np.linspace(self.w, self.e, nlon + 1)
        souths = np.linspace(self.s, self.n, nlat + 1)
        bottoms = np.linspace(self.bottom, self.top, nh + 1)
        dlon = wests[1] - wests[0]
        dlat = souths[1] - souths[0]
        dh = bottoms[1] - bottoms[0]
        tesseroids = [
            Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=self.props)
            for i in wests[:-1] for j in souths[:-1] for k in bottoms[:-1]]
        return tesseroids


class Sphere(GeometricElement):
    """
    A sphere.

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
        super().__init__(props)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)
        self.center = np.array([x, y, z])

    def __str__(self):
        """Return a string representation of the sphere."""
        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('radius', self.radius)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)


class PolygonalPrism(GeometricElement):
    """
    A 3D prism with polygonal cross-section.

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
        super().__init__(props)
        self.x = np.fromiter((v[0] for v in vertices), dtype=np.float)
        self.y = np.fromiter((v[1] for v in vertices), dtype=np.float)
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
        verts = np.transpose([self.x, self.y])
        return Polygon(verts, self.props)
