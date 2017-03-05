"""
Classes that define basic geometric primitives. All inherit from
``GeometricElement``.
"""
from __future__ import division, print_function
from builtins import object, super
import numpy as np
import copy


class GeometricElement(object):
    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        if props is None:
            self.props = dict()
        else:
            self.props = copy.deepcopy(props)

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
        return copy.deepcopy(self)


class Polygon(GeometricElement):
    """
    A polygon (2D).

    Parameters:

    * vertices : list of lists
        List of [x, y] pairs with the coordinates of the vertices.
    * props : dict
        Physical properties assigned to the polygon.
        Ex: ``props={'density':10, 'susceptibility':10000}``

    Examples::

        >>> poly = Polygon(vertices=[[0, 0], [1, 4], [2, 5]],
        ...                props={'density': 500})
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
    A square.

    Parameters:

    * bounds : list = [x1, x2, y1, y2]
        Coordinates of the top right and bottom left corners of the square
    * props : dict
        Physical properties assigned to the square.
        Ex: ``props={'density':10, 'slowness':10000}``

    Example::

        >>> sq = Square(bounds=[0, 1, 2, 4], props={'density': 750})
        >>> sq.bounds
        [0, 1, 2, 4]
        >>> sq.x1
        0
        >>> sq.x2
        1
        >>> sq.props
        {'density': 750}
        >>> sq.addprop('magnetization', [100, 200, 300])
        >>> sq.props['magnetization']
        [100, 200, 300]

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
        Ex: ``props={'density':10}``

    Examples:

        >>> p = Prism(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6,
        ...           props={'density': 200})
        >>> p.props['density']
        200
        >>> p.bounds
        [1, 2, 3, 4, 5, 6]
        >>> p.center
        array([ 1.5,  3.5,  5.5])
        >>> p.addprop('density', 2670)
        >>> p.props['density']
        2670

    """

    def __init__(self, x1, x2, y1, y2, z1, z2, props=None):
        super().__init__(props)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

    @property
    def bounds(self):
        "Get the bounding box of the prism (i.e., the borders of the prism)"
        return [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]

    @property
    def center(self):
        "The coordinates of the center of the prism"
        xc = 0.5*(self.x1 + self.x2)
        yc = 0.5*(self.y1 + self.y2)
        zc = 0.5*(self.z1 + self.z2)
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
        Ex: ``props={'density':10}``

    Examples:

        >>> t = Tesseroid(w=1, e=2, s=3, n=4, top=6, bottom=5,
        ...               props={'density': 200})
        >>> t.props['density']
        200
        >>> t.bounds
        [1, 2, 3, 4, 6, 5]
        >>> t.center
        array([ 1.5,  3.5,  5.5])
        >>> t.addprop('density', 2670)
        >>> t.props['density']
        2670

    """

    def __init__(self, w, e, s, n, top, bottom, props=None):
        super().__init__(props)
        self.w = w
        self.e = e
        self.s = s
        self.n = n
        self.bottom = bottom
        self.top = top

    @property
    def bounds(self):
        "The bounding box of the tesseroid (i.e., the borders)."
        return [self.w, self.e, self.s, self.n, self.top, self.bottom]

    @property
    def center(self):
        "The geometric center of the tesseroid"
        lonc = 0.5*(self.w + self.e)
        latc = 0.5*(self.s + self.n)
        hc = 0.5*(self.top + self.bottom)
        return np.array([lonc, latc, hc])

    def split(self, nlon, nlat, nheight):
        """
        Split the tesseroid into smaller ones.

        The smaller tesseroids will share the large one's physical properties
        (``props``).

        Parameters:

        * nlon, nlat, nheight : int
            The number of sections to split in the longitudinal, latitudinal,
            and vertical directions

        Returns:

        * tesseroids : list
            A list of nlon*nlat*nheight tesseroids that make up the larger one.


        Examples::

            >>> tess = Tesseroid(w=-10, e=10, s=-20, n=20, top=0, bottom=-30,
            ...                  props={'density': 300})
            >>> # Split in half along each dimension
            >>> split = tess.split(nlon=2, nlat=2, nheight=2)
            >>> len(split)
            8
            >>> for t in split:
            ...     print(t.bounds)
            [-10.0, 0.0, -20.0, 0.0, -15.0, -30.0]
            [-10.0, 0.0, -20.0, 0.0, 0.0, -15.0]
            [-10.0, 0.0, 0.0, 20.0, -15.0, -30.0]
            [-10.0, 0.0, 0.0, 20.0, 0.0, -15.0]
            [0.0, 10.0, -20.0, 0.0, -15.0, -30.0]
            [0.0, 10.0, -20.0, 0.0, 0.0, -15.0]
            [0.0, 10.0, 0.0, 20.0, -15.0, -30.0]
            [0.0, 10.0, 0.0, 20.0, 0.0, -15.0]
            >>> # If you don't want to split along a given dimension, use 1 as
            >>> # the number of splits. Ex, to split only along the vertical
            >>> split = tess.split(nlon=1, nlat=1, nheight=3)
            >>> len(split)
            3
            >>> for t in split:
            ...     print(t.bounds)
            [-10.0, 10.0, -20.0, 20.0, -20.0, -30.0]
            [-10.0, 10.0, -20.0, 20.0, -10.0, -20.0]
            [-10.0, 10.0, -20.0, 20.0, 0.0, -10.0]

        """
        wests = np.linspace(self.w, self.e, nlon + 1)
        souths = np.linspace(self.s, self.n, nlat + 1)
        bottoms = np.linspace(self.bottom, self.top, nheight + 1)
        dlon = wests[1] - wests[0]
        dlat = souths[1] - souths[0]
        dh = bottoms[1] - bottoms[0]
        tesseroids = [Tesseroid(w, w + dlon, s, s + dlat, bottom + dh, bottom,
                                props=self.props)
                      for w in wests[:-1]
                      for s in souths[:-1]
                      for bottom in bottoms[:-1]]
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
        Physical properties assigned to the sphere.
        Ex: ``props={'density':10}``

    Examples:

        >>> s = Sphere(x=1, y=2, z=3, radius=10, props={'density': 200})
        >>> s.center
        array([1, 2, 3])
        >>> s.diameter
        20
        >>> s.props['density']
        200
        >>> s.addprop('magnetization', [20, 1, 2])
        >>> s.props['magnetization']
        [20, 1, 2]

    """

    def __init__(self, x, y, z, radius, props=None):
        super().__init__(props)
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    @property
    def center(self):
        "The geometric center of the sphere"
        return np.array([self.x, self.y, self.z])

    @property
    def diameter(self):
        "The diameter of the sphere."
        return self.radius*2


class PolygonalPrism(Polygon):
    """
    A 3D prism with polygonal cross-section.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    .. note:: Many modeling functions require *vertices* to be **clockwise**.

    Parameters:

    * vertices : list of lists
        Coordinates of the vertices. A list of ``[x, y]`` pairs.
    * z1, z2 : float
        Top and bottom of the prism
    * props :  dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10}``

    Examples:

        >>> p = PolygonalPrism(vertices=[[1, 1], [1, 2], [2, 2], [2, 1]],
        ...                    z1=0, z2=3)
        >>> p.vertices
        array([[1, 1],
               [1, 2],
               [2, 2],
               [2, 1]])
        >>> p.nverts
        4
        >>> p.x
        array([1, 1, 2, 2])
        >>> p.y
        array([1, 2, 2, 1])
        >>> p.addprop('resistivity', 25)
        >>> p.props['resistivity']
        25
        >>> p.addprop('density', 2670)
        >>> p.props['density']
        2670
        >>> # You can convert the PolygonalPrism into a Polygon (as viewed from
        >>> # above). This might be useful for plotting on maps.
        >>> poly = p.polygon
        >>> isinstance(poly, Polygon)
        True
        >>> poly.vertices
        array([[1, 1],
               [1, 2],
               [2, 2],
               [2, 1]])

    """

    def __init__(self, vertices, z1, z2, props=None):
        super().__init__(vertices, props)
        self.z1 = z1
        self.z2 = z2

    @property
    def polygon(self):
        "The polygon describing the prism viewed from above."
        return Polygon(self.vertices, self.props)
