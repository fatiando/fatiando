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
from __future__ import division
import numpy

from ..constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G
try:
    from . import _tesseroid
except ImportError:
    pass

RATIO_POTENTIAL = 1
RATIO_G = 1.5
RATIO_GG = 8
QUEUE_SIZE = 100


def potential(lons, lats, heights, tesseroids, dens=None,
              ratio=RATIO_POTENTIAL):
    """
    Calculate the gravitational potential due to a tesseroid model.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in SI units

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.potential(tesseroid, density, ratio, QUEUE_SIZE, rlon,
                             sinlat, coslat, radius, result)
    result *= G
    return result


def gx(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_G):
    """
    Calculate the North component of the gravitational attraction.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in mGal

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gx(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                      coslat, radius, result)
    result *= SI2MGAL*G
    return result


def gy(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_G):
    """
    Calculate the East component of the gravitational attraction.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in mGal

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gy(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                      coslat, radius, result)
    result *= SI2MGAL*G
    return result


def gz(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_G):
    """
    Calculate the radial component of the gravitational attraction.

    .. warning::
        In order to conform with the regular convention of positive density
        giving positive gz values, **this component only** is calculated
        with **z axis -> Down**.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in mGal

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gz(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                      coslat, radius, result)
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    result *= -SI2MGAL*G
    return result


def gxx(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the xx component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gxx(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result


def gxy(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the xy component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gxy(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result


def gxz(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the xz component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gxz(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result


def gyy(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the yy component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gyy(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result


def gyz(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the yz component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gyz(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result


def gzz(lons, lats, heights, tesseroids, dens=None, ratio=RATIO_GG):
    """
    Calculate the zz component of the gravity gradient tensor.

    Parameters:

    * lons, lats, heights : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * tesseroids : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * radio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.

    Returns:

    * res : array
        The calculated field in Eotvos

    """
    if lons.shape != lats.shape or lons.shape != heights.shape:
        raise ValueError(
            "Input arrays lons, lats, and heights must have same length!")
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
        _tesseroid.gzz(tesseroid, density, ratio, QUEUE_SIZE, rlon, sinlat,
                       coslat, radius, result)
    result *= SI2EOTVOS*G
    return result
