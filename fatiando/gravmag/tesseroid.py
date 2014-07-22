r"""
Calculates the potential fields of a tesseroid (spherical prism).

.. admonition:: Coordinate systems

    The gravitational attraction
    and gravity gradient tensor
    are calculated with respect to
    the local coordinate system of the computation point.
    This system has **x -> North**, **y -> East**, **z -> up**
    (radial direction).

**Gravity**

.. warning:: The :math:`g_z` component is an **exception** to this.
    In order to conform with the regular convention
    of z-axis pointing toward the center of the Earth,
    **this component only** is calculated with **z -> Down**.
    This way, gravity anomalies of
    tesseroids with positive density
    are positive, not negative.

Functions:
:func:`~fatiando.gravmag.prism.potential`,
:func:`~fatiando.gravmag.prism.gx`,
:func:`~fatiando.gravmag.prism.gy`,
:func:`~fatiando.gravmag.prism.gz`,
:func:`~fatiando.gravmag.prism.gxx`,
:func:`~fatiando.gravmag.prism.gxy`,
:func:`~fatiando.gravmag.prism.gxz`,
:func:`~fatiando.gravmag.prism.gyy`,
:func:`~fatiando.gravmag.prism.gyz`,
:func:`~fatiando.gravmag.prism.gzz`

The gravitational fields are calculated using the formula of Grombein et al.
(2013):

.. math::
    V(r,\phi,\lambda) = G \rho
        \displaystyle\int_{\lambda_1}^{\lambda_2}
        \displaystyle\int_{\phi_1}^{\phi_2}
        \displaystyle\int_{r_1}^{r_2}
        \frac{1}{\ell} \kappa \ d r' d \phi' d \lambda'

.. math::
    g_{\alpha}(r,\phi,\lambda) = G \rho
        \displaystyle\int_{\lambda_1}^{\lambda_2}
        \displaystyle\int_{\phi_1}^{\phi_2} \displaystyle\int_{r_1}^{r_2}
        \frac{\Delta_{\alpha}}{\ell^3} \kappa \ d r' d \phi' d \lambda'
        \ \ \alpha \in \{x,y,z\}

.. math::
    g_{\alpha\beta}(r,\phi,\lambda) = G \rho
        \displaystyle\int_{\lambda_1}^{\lambda_2}
        \displaystyle\int_{\phi_1}^{\phi_2} \displaystyle\int_{r_1}^{r_2}
        I_{\alpha\beta}({r'}, {\phi'}, {\lambda'} )
        \ d r' d \phi' d \lambda'
        \ \ \alpha,\beta \in \{x,y,z\}

.. math::
    I_{\alpha\beta}({r'}, {\phi'}, {\lambda'}) =
        \left(
            \frac{3\Delta_{\alpha} \Delta_{\beta}}{\ell^5} -
            \frac{\delta_{\alpha\beta}}{\ell^3}
        \right)
        \kappa\
        \ \ \alpha,\beta \in \{x,y,z\}

where :math:`\rho` is density,
:math:`\{x, y, z\}` correspond to the local coordinate system
of the computation point P,
:math:`\delta_{\alpha\beta}` is the `Kronecker delta`_, and

.. math::
   :nowrap:

    \begin{eqnarray*}
        \Delta_x &=& r' K_{\phi} \\
        \Delta_y &=& r' \cos \phi' \sin(\lambda' - \lambda) \\
        \Delta_z &=& r' \cos \psi - r\\
        \ell &=& \sqrt{r'^2 + r^2 - 2 r' r \cos \psi} \\
        \cos\psi &=& \sin\phi\sin\phi' + \cos\phi\cos\phi'
                     \cos(\lambda' - \lambda) \\
        K_{\phi} &=& \cos\phi\sin\phi' - \sin\phi\cos\phi'
                     \cos(\lambda' - \lambda)\\
        \kappa &=& {r'}^2 \cos \phi'
    \end{eqnarray*}


:math:`\phi` is latitude,
:math:`\lambda` is longitude, and
:math:`r` is radius.

.. _Kronecker delta: http://en.wikipedia.org/wiki/Kronecker_delta

**Numerical integration**

The above integrals are solved using the Gauss-Legendre Quadrature rule
(Asgharzadeh et al., 2007;
Wild-Pfeiffer, 2008):

.. math::
    g_{\alpha\beta}(r,\phi,\lambda) \approx G \rho
        \frac{(\lambda_2 - \lambda_1)(\phi_2 - \phi_1)(r_2 - r_1)}{8}
        \displaystyle\sum_{k=1}^{N^{\lambda}}
        \displaystyle\sum_{j=1}^{N^{\phi}}
        \displaystyle\sum_{i=1}^{N^r}
        W^r_i W^{\phi}_j W^{\lambda}_k
        I_{\alpha\beta}({r'}_i, {\phi'}_j, {\lambda'}_k )
        \ \alpha,\beta \in \{1,2,3\}

where :math:`W_i^r`, :math:`W_j^{\phi}`, and :math:`W_k^{\lambda}`
are weighting coefficients
and :math:`N^r`, :math:`N^{\phi}`, and :math:`N^{\lambda}`
are the number of quadrature nodes
(i.e., the order of the quadrature),
for the radius, latitude, and longitude, respectively.


**References**

Asgharzadeh, M. F., R. R. B. von Frese, H. R. Kim, T. E. Leftwich,
and J. W. Kim (2007),
Spherical prism gravity effects by Gauss-Legendre quadrature integration,
Geophysical Journal International, 169(1), 1-11,
doi:10.1111/j.1365-246X.2007.03214.x.

Grombein, T.; Seitz, K.; Heck, B. (2013), Optimized formulas for the
gravitational field of a tesseroid, Journal of Geodesy,
doi: 10.1007/s00190-013-0636-1

Wild-Pfeiffer, F. (2008),
A comparison of different mass elements for use in gravity gradiometry,
Journal of Geodesy, 82(10), 637-653, doi:10.1007/s00190-008-0219-8.


----

"""
import numpy

from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G
try:
    from fatiando.gravmag._tesseroid import *
except:
    def not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "Couldn't load C coded extension module.")
    _potential = not_implemented
    _gx = not_implemented
    _gy = not_implemented
    _gz = not_implemented
    _gxx = not_implemented
    _gxy = not_implemented
    _gxz = not_implemented
    _gyy = not_implemented
    _gyz = not_implemented
    _gzz = not_implemented
    _distance = not_implemented
    _too_close = not_implemented


def potential(lons, lats, heights, tesseroids, dens=None, ratio=0.5):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
                               _potential, ratio, dens)


def gx(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the x (North) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
                                         _gx, ratio, dens)


def gy(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the y (East) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
                                         _gy, ratio, dens)


def gz(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a
    tesseroid model.
    """
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    return -1*SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
                                              _gz, ratio, dens)


def gxx(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xx (North-North) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxx, ratio, dens)


def gxy(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xy (North-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxy, ratio, dens)


def gxz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xz (North-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxz, ratio, dens)


def gyy(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the yy (East-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
                                           _gyy, ratio, dens)


def gyz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the yz (East-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
                                           _gyz, ratio, dens)


def gzz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    ndata = len(lons)
    # Convert things to radians
    d2r = numpy.pi / 180.
    rlon = d2r*lons
    sinlat = numpy.sin(d2r*lats)
    coslat = numpy.cos(d2r*lats)
    # Transform the heights into radii
    radius = MEAN_EARTH_RADIUS + heights
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
    for tesseroid in tesseroids:
        if (tesseroid is None or
                ('density' not in tesseroid.props and dens is None)):
            continue
        if dens is not None:
            density = dens
        else:
            density = tesseroid.props['density']
        _with_optimal_division(tesseroid, density, rlon, sinlat, coslat,
                               radius, _gzz, ratio, result)
    result *= SI2EOTVOS*G
    return result


def _with_optimal_division(tesseroids, density, lon, sinlat, coslat, radius,
                           kernel, ratio, result):
    """
    Calculate the effect of a given kernel in the most precise way by
    adaptively discretizing the tesseroids into smaller ones.
    """
    ndata = len(lon)
    d2r = numpy.pi / 180.

    # Create some buffers to reduce memory allocation
    distances = numpy.zeros(ndata, numpy.float)
    lonc = numpy.zeros(2, numpy.float)
    sinlatc = numpy.zeros(2, numpy.float)
    coslatc = numpy.zeros(2, numpy.float)
    rc = numpy.zeros(2, numpy.float)
    allpoints = numpy.arange(ndata)

    queue = [(allpoints, tesseroid.get_bounds())]
    while queue:
        points, tess = queue.pop()
        w, e, s, n, top, bottom = tess
        size = max([MEAN_EARTH_RADIUS*d2r*(e - w),
                    MEAN_EARTH_RADIUS*d2r*(n - s),
                    top - bottom])
        _distance(tess, rlon, sinlat, coslat, radius, points, distances)
        need_divide, dont_divide = _too_close(points, distances,
                                              ratio*size)
        if len(need_divide):
            if len(queue) >= 1000:
                raise ValueError('Tesseroid queue overflow')
            queue.extend(_half(tess, need_divide))
        if len(dont_divide):
            kernel(tess, density, lon, sinlat, coslat, radius, lonc,
                   sinlatc, coslatc, rc, result, dont_divide)


def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio, dens):
    """
    Calculate the effect of a given kernel in the most precise way by
    adaptively discretizing the tesseroids into smaller ones.
    """
    ndata = len(lons)
    # Convert things to radians
    d2r = numpy.pi / 180.
    rlons = d2r*lons
    sinlats = numpy.sin(d2r*lats)
    coslats = numpy.cos(d2r*lats)
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Create some buffers to reduce memory allocation
    distances = numpy.zeros(ndata, numpy.float)
    lonc = numpy.zeros(2, numpy.float)
    sinlatc = numpy.zeros(2, numpy.float)
    coslatc = numpy.zeros(2, numpy.float)
    rc = numpy.zeros(2, numpy.float)
    allpoints = numpy.arange(ndata)
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
    for tesseroid in tesseroids:
        if (tesseroid is None or
                ('density' not in tesseroid.props and dens is None)):
            continue
        if dens is not None:
            density = dens
        else:
            density = tesseroid.props['density']
        queue = [(allpoints, tesseroid.get_bounds())]
        while queue:
            points, tess = queue.pop()
            w, e, s, n, top, bottom = tess
            size = max([MEAN_EARTH_RADIUS*d2r*(e - w),
                        MEAN_EARTH_RADIUS*d2r*(n - s),
                        top - bottom])
            _distance(tess, rlons, sinlats, coslats, radii, points, distances)
            need_divide, dont_divide = _too_close(points, distances,
                                                  ratio*size)
            if len(need_divide):
                if len(queue) >= 1000:
                    raise ValueError('Tesseroid queue overflow')
                queue.extend(_half(tess, need_divide))
            if len(dont_divide):
                kernel(tess, density, rlons, sinlats, coslats, radii, lonc,
                       sinlatc, coslatc, rc, result, dont_divide)
    result *= G
    return result


def _half(bounds, points):
    w, e, s, n, top, bottom = bounds
    dlon = 0.5*(e - w)
    dlat = 0.5*(n - s)
    dh = 0.5*(top - bottom)
    yield (points, (w, w + dlon, s, s + dlat, bottom + dh, bottom))
    yield (points, (w, w + dlon, s, s + dlat, top, bottom + dh))
    yield (points, (w, w + dlon, s + dlat, n, bottom + dh, bottom))
    yield (points, (w, w + dlon, s + dlat, n, top, bottom + dh))
    yield (points, (w + dlon, e, s, s + dlat, bottom + dh, bottom))
    yield (points, (w + dlon, e, s, s + dlat, top, bottom + dh))
    yield (points, (w + dlon, e, s + dlat, n, bottom + dh, bottom))
    yield (points, (w + dlon, e, s + dlat, n, top, bottom + dh))
