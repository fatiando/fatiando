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


class TriaxialEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented triaxial ellipsoid.

    Triaxial ellipsoids are those having different semi-axes.
    This code follows Clark et al. (1986) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * large_axis, intermediate_axis, small_axis: float
        Semi-axes forming the ellipsoid (in m).
    * strike, dip, rake
        Orientation angles of the ellipsoid (in degrees).
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[10, 25, 40],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              90, 0, 0]}``

    Examples:

        >>> e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
        ...                       intermediate_axis=5, small_axis=4,
        ...                       strike=10, dip=20, rake=30, props={
        ...                       'remanent magnetization': [10, 25, 40],
        ...                       'susceptibility tensor': [0.562, 0.485,
        ...                                                 0.25, 90, 0,
        ...                                                 0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 | \
strike:10 | dip:20 | rake:30 | density:20 | remanent magnetization:[10, 25, 40\
] | susceptibility tensor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = TriaxialEllipsoid(1, 2.7, 3, 6, 5, 4, 10, 20, 30)
        >>> print(e)
        x:1 | y:2.7 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 \
| strike:10 | dip:20 | rake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2.7 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 \
| strike:10 | dip:20 | rake:30 | density:2670

    """

    def __init__(self, x, y, z, large_axis, intermediate_axis, small_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.large_axis = large_axis
        self.intermediate_axis = intermediate_axis
        self.small_axis = small_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.intermediate_axis and \
            self.intermediate_axis > self.small_axis, "large_axis must be grea\
ter than intermediate_axis and intermediate_axis must greater than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the triaxial ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('large_axis', self.large_axis),
                 ('intermediate_axis', self.intermediate_axis),
                 ('small_axis', self.small_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = np.dot(U, np.diag([k1, k2, k3]))
            suscep_tensor = np.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


class ProlateEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented prolate ellipsoid.

    Prolate ellipsoids are those having symmetry around the large axes.
    This code follows Emerson et al. (1985) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C
    handheld computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * large_axis, small_axis: float
        Semi-axes forming the ellipsoid.
    * strike, dip, rake
        Orientation angles of the ellipsoid.
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[13, -5, 7.4],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              0, 7, 29.4]}``

    Examples:

        >>> e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
        ...                      strike=10, dip=20, rake=30, props={
        ...                      'remanent magnetization': [10, 25, 40],
        ...                      'susceptibility tensor': [0.562, 0.485,
        ...                                                0.25, 90, 0,
        ...                                                0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30 | density:20 | remanent magnetization:[10, 25, 40] | susceptibility ten\
sor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = ProlateEllipsoid(1, 2, 3, 6, 4, 10, 20, 30)
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30 | density:2670

    """

    def __init__(self, x, y, z, large_axis, small_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.large_axis = large_axis
        self.small_axis = small_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.small_axis, "large_axis must be greater \
than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the prolate ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('large_axis', self.large_axis),
                 ('small_axis', self.small_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = np.dot(U, np.diag([k1, k2, k3]))
            suscep_tensor = np.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


class OblateEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented oblate ellipsoid.

    Oblate ellipsoids are those having symmetry around the small axes.
    This code follows a convention similar to that defined by
    Emerson et al. (1985) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C
    handheld computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * small_axis, large_axis: float
        Semi-axes forming the ellipsoid.
    * strike, dip, rake
        Orientation angles of the ellipsoid.
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[10, 25, 40],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              90, 0, 0]}``

    Examples:

        >>> e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
        ...                     strike=10, dip=20, rake=30, props={
        ...                     'remanent magnetization': [10, 25, 40],
        ...                     'susceptibility tensor': [0.562, 0.485,
        ...                                               0.25, 90, 0,
        ...                                               0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:4 | large_axis:6 | strike:10 | dip:20 | r\
ake:30 | density:20 | remanent magnetization:[10, 25, 40] | susceptibility ten\
sor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = OblateEllipsoid(1, 2, 3, 2, 9, 10, 20, 30)
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:2 | large_axis:9 | strike:10 | dip:20 | r\
ake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:2 | large_axis:9 | strike:10 | dip:20 | r\
ake:30 | density:2670

    """

    def __init__(self, x, y, z, small_axis, large_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.small_axis = small_axis
        self.large_axis = large_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.small_axis, "large_axis must be greater \
than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the oblate ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('small_axis', self.small_axis),
                 ('large_axis', self.large_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = np.dot(U, np.diag([k1, k2, k3]))
            suscep_tensor = np.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


def _auxiliary_angles(strike, dip, rake):
    '''
    Calculate auxiliary angles alpha, gamma and delta (Clark et al., 1986)
    as functions of geological angles strike, dip and rake
    (Clark et al., 1986; Allmendinger et al., 2012), given in degrees.
    This function implements the formulas presented by
    Clark et al. (1986).

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    Allmendinger, R., Cardozo, N., and Fisher, D. M.:
    Structural geology algorithms : vectors and tensors,
    Cambridge University Press, 2012.
    '''

    strike_r = np.deg2rad(strike)
    cos_dip = np.cos(np.deg2rad(dip))
    sin_dip = np.sin(np.deg2rad(dip))
    cos_rake = np.cos(np.deg2rad(rake))
    sin_rake = np.sin(np.deg2rad(rake))

    aux = sin_dip*sin_rake
    aux1 = cos_rake/np.sqrt(1 - aux*aux)
    aux2 = sin_dip*cos_rake

    if aux1 > 1.:
        aux1 = 1.
    if aux1 < -1.:
        aux1 = -1.

    alpha = strike_r - np.arccos(aux1)
    if aux2 != 0:
        gamma = np.arctan(cos_dip/aux2)
    else:
        if cos_dip > 0:
            gamma = np.pi/2
        if cos_dip < 0:
            gamma = -np.pi/2
        if cos_dip == 0:
            gamma = 0
    delta = np.arcsin(aux)

    assert delta <= np.pi/2, 'delta must be lower than or equalt to 90 \
degrees'

    assert (gamma >= -np.pi/2) and (gamma <= np.pi/2), 'gamma must lie \
between -90 and 90 degrees.'

    return alpha, gamma, delta


def _coord_transf_matrix_triaxial(alpha, gamma, delta):
    '''
    Calculate the coordinate transformation matrix
    for triaxial or prolate ellipsoids by using the auxiliary angles
    alpha, gamma and delta.

    The columns of this matrix are defined according to the unit vectors
    v1, v2 and v3 presented by Clark et al. (1986, p. 192).

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*cos_delta, -sin_alpha*cos_delta,
                   -sin_delta])

    v2 = np.array([cos_alpha*cos_gamma*sin_delta +
                   sin_alpha*sin_gamma, sin_alpha*cos_gamma*sin_delta -
                   cos_alpha*sin_gamma, -cos_gamma*cos_delta])

    v3 = np.array([sin_alpha*cos_gamma - cos_alpha*sin_gamma*sin_delta,
                   -cos_alpha*cos_gamma -
                   sin_alpha*sin_gamma*sin_delta,
                   sin_gamma*cos_delta])

    transf_matrix = np.vstack((v1, v2, v3)).T

    return transf_matrix


def _coord_transf_matrix_oblate(alpha, gamma, delta):
    '''
    Calculate the coordinate transformation matrix
    for oblate ellipsoids by using the auxiliary angles
    alpha, gamma and delta.

    The columns of this matrix are defined by unit vectors
    v1, v2 and v3.
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*sin_gamma*sin_delta +
                   sin_alpha*cos_gamma, -sin_alpha*sin_gamma*sin_delta -
                   cos_alpha*cos_gamma, sin_gamma*cos_delta])

    v2 = np.array([-cos_alpha*cos_delta, -sin_alpha*cos_delta,
                   -sin_delta])

    v3 = np.array([sin_alpha*sin_gamma + cos_alpha*cos_gamma*sin_delta,
                   -cos_alpha*sin_gamma +
                   sin_alpha*cos_gamma*sin_delta,
                   -cos_gamma*cos_delta])

    transf_matrix = np.vstack((v1, v2, v3)).T

    return transf_matrix
