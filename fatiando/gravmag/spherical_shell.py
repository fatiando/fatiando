"""
Calculate the gravitational effects of a spherical shell and half of a shell.

Calculates the effects on the polar axis.

But due to the symmetry of the shell, the values can be considered to be at any
latitude, longitude pair.

Components, gx, gy, gxy, gxz, and gyz are all equal to zero (0).

**Functions for the shell:**

* :func:`~fatiando.gravmag.spherical_shell.potential`: the gravitational
  potential
* :func:`~fatiando.gravmag.spherical_shell.gz`: the vertical component of the
  gravitational attraction
* :func:`~fatiando.gravmag.spherical_shell.gxx`: the xx (North-North) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.gyy`: the yy (East-East) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.gzz`: the zz (radial-radial)
  component of the gravity gradient tensor

**Functions for the half shell:**

* :func:`~fatiando.gravmag.spherical_shell.half_potential`: the gravitational
  potential
* :func:`~fatiando.gravmag.spherical_shell.half_gz`: the vertical component of
  the gravitational attraction
* :func:`~fatiando.gravmag.spherical_shell.half_gxx`: the xx (North-North)
  component of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.half_gyy`: the yy (East-East)
  component of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.half_gzz`: the zz (radial-radial)
  component of the gravity gradient tensor

----

"""
from __future__ import division
import numpy

from fatiando.constants import MEAN_EARTH_RADIUS, G, SI2MGAL, SI2EOTVOS


def half_potential(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gravitational potential of half a spherical shell on the
    polar axis.

    .. math::

        V(r) = 2\pi G \rho \left[ \dfrac{l^3 + {r'}^3}{3r} - 0.5 {r'}^2 \right]
        \Biggr\rvert_{r'=r_1}^{r'=r_2}

    where :math:`\rho` is the dentity, :math:`l = \sqrt{r^2 + {r'}^2}`,
    :math:`r` is the radius coordinate of the observation point, :math:`r_1`
    and :math:`r_2` are the bottom and top of the spherical shell,
    respectively.

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * potential : float or numpy array
        The gravitational potential in SI units.

    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r ** 2 + rl ** 2)
        res += ((-1)**i)*((l**3 + rl**3)/(3*r) - 0.5*rl**2)
    res *= 2*numpy.pi*G*density
    return res


def half_gz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate gz of half a spherical shell on the polar axis.

    gz is the radial derivative of the potential.
    The sign should be inverted because our convention is that z points down
    (toward the center of the Earth)


    .. math::

        g_z(r) = -\dfrac{\partial V}{\partial r} = 2\pi G \rho \left[
        \dfrac{l^3 + {r'}^3}{3r^2} - l \right] \Biggr\rvert_{r'=r_1}^{r'=r_2}

    where :math:`\rho` is the dentity, :math:`l = \sqrt{r^2 + {r'}^2}`,
    :math:`r` is the radius coordinate of the observation point, :math:`r_1`
    and :math:`r_2` are the bottom and top of the spherical shell,
    respectively.

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * gz : float or numpy array
        The gz component in mGal.

    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r**2 + rl**2)
        res += ((-1)**i)*((l**3 + rl**3)/(3*r**2) - l)
    res *= 2*numpy.pi*G*density*SI2MGAL
    return res


def half_gzz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gzz gradient tensor component of half a spherical shell on
    the polar axis.

    gzz is the second radial derivative of the potential.

    .. math::

        g_{zz}(r) = 2\pi G \rho \left[ 2\dfrac{l^3 + {r'}^3}{3r^3} -
        \dfrac{l}{r} + \dfrac{r}{l} \right] \Biggr\rvert_{r'=r_1}^{r'=r_2}

    where :math:`\rho` is the dentity, :math:`l = \sqrt{r^2 + {r'}^2}`,
    :math:`r` is the radius coordinate of the observation point, :math:`r_1`
    and :math:`r_2` are the bottom and top of the spherical shell,
    respectively.

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * gzz : float or numpy array
        The gzz component in Eotvos.

    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r**2 + rl**2)
        res += ((-1)**i)*(2*(l**3 + rl**3)/(3*r**3) - l/r + r/l)
    res *= 2*numpy.pi*G*density*SI2EOTVOS
    return res


def half_gxx(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gxx gradient tensor component of half a spherical shell on
    the polar axis.

    gxx is the second northward derivative of the potential.

    .. math::

        g_{xx}(r) = -\dfrac{1}{2} g_{zz}

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * gxx : float or numpy array
        The gxx component in Eotvos.

    """
    return -0.5*half_gzz(heights, top, bottom, density, radius)


def half_gyy(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gyy gradient tensor component of half a spherical shell on
    the polar axis.

    gyy is the second eastward derivative of the potential.

    .. math::

        g_{xx}(r) = -\dfrac{1}{2} g_{zz}

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * gyy : float or numpy array
        The gyy component in Eotvos.

    """
    return half_gxx(heights, top, bottom, density, radius)


def potential(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gravitational potential of a spherical shell on the polar
    axis.

    .. math::

        V(r) = \dfrac{4}{3}\pi G \rho \dfrac{r_2^3 - r_1^3}{r},

    where :math:`\rho` is the dentity, :math:`r` is the radius coordinate of
    the observation point, :math:`r_1` and :math:`r_2` are the bottom and top
    radius coordinates of the spherical shell, respectively.

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * result : float or numpy array
        The potential in SI.

    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = (4/3)*np.pi*G*density*(r2**3 - r1**3)/r
    return res


def gz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gz component of a spherical shell on the polar axis.

    gz is the radial derivative of the potential.

    .. math::

        g_z(r) = \dfrac{V(r)}{r},

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * result : float or numpy array
        The gz component in mGal.

    """
    r = heights + radius
    pot = potential(heights, top, bottom, density, radius)
    res = SI2MGAL*pot/r
    return res


def gzz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gzz component of a spherical shell on the polar axis.

    gzz is the second radial derivative of the potential.

    .. math::

        g_{zz}(r) = 2\dfrac{V(r)}{r^2},

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * result : float or numpy array
        The gzz component in EOTVOS.

    """
    r = heights + radius
    pot = potential(heights, top, bottom, density, radius)
    res = SI2EOTVOS*2*pot/(r**2)
    return res


def gxx(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gxx component of a spherical shell on the polar axis.

    gxx is the second northward derivative of the potential.

    .. math::

        g_{xx}(r) = -\dfrac{g_{zz}}{2},

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * result : float or numpy array
        The gxx component in EOTVOS.

    """
    res = -0.5*gzz(heights, top, bottom, density, radius)
    return res


def gyy(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    r"""
    Calculate the gyy component of a spherical shell on the polar axis.

    gyy is the second eastward derivative of the potential.

    .. math::

        g_{yy}(r) = -\dfrac{g_{zz}}{2},

    .. note:: Heights are measured with respect to the *radius* parameter.

    Parameters:

    * heights : float or numpy array
        The height of the computation point(s) in meters.
    * top : float
        The height of the top of the half shell in meters.
    * bottom : float
        The height of the bottom of the half shell in meters.
    * density : float
        The density of the half shell in :math:`kg.m^{-3}`

    Returns:

    * result : float or numpy array
        The gyy component in EOTVOS.

    """
    res = gxx(heights, top, bottom, density, radius)
    return res
