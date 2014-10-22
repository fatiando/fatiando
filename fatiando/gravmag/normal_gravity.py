"""
Gravity of ellipsoid models and common reductions (Bouguer, free-air)

**Reference ellipsoids**

This module uses instances of
:class:`~fatiando.gravmag.normal_gravity.ReferenceEllipsoid` to store the
physical constants of ellipsoids.
To create a new ellipsoid, just instantiate ``ReferenceEllipsoid`` and give it
the semimajor axis ``a``, the flattening ``f``, the geocentric gravitational
constant ``GM``, and the angular velocity ``omega``.
All  other quantities, like the gravity at the poles and equator,
eccentricities, etc, are computed by the class from these 4 parameters.

Available ellipsoids:

* ``WGS84``::

    >>> from fatiando.gravmag.normal_gravity import WGS84
    >>> print(WGS84.name)
    World Geodetic System 1984
    >>> print('{:.0f}'.format(WGS84.a))
    6378137
    >>> print('{:.17f}'.format(WGS84.f))
    0.00335281066474748
    >>> print('{:.10g}'.format(WGS84.GM))
    3.986004418e+14
    >>> print('{:.7g}'.format(WGS84.omega))
    7.292115e-05
    >>> print('{:.4f}'.format(WGS84.b))
    6356752.3142
    >>> print('{:.8f}'.format(WGS84.E)) # Linear eccentricity
    521854.00842339
    >>> print('{:.15f}'.format(WGS84.e_prime)) # second eccentricity
    0.082094437949696
    >>> print('{:.10f}'.format(WGS84.gamma_a)) # gravity at the equator
    9.7803253359
    >>> print('{:.11f}'.format(WGS84.gamma_b)) # gravity at the pole
    9.83218493786
    >>> print('{:.14f}'.format(WGS84.m))
    0.00344978650684


"""
from __future__ import division
import math
import numpy

from .. import utils

class ReferenceEllipsoid(object):
    """
    A generic reference ellipsoid.

    It stores the physical constants defining the ellipsoid and has properties
    for computing other (derived) quantities.

    All quantities are expected and returned in SI units.

    Parameters:

    * a : float
        The semimajor axis (the largest one, at the equator). In meters.
    * f : float
        The flattening. Adimensional.
    * GM : float
        The geocentric gravitational constant of the earth, including the
        atmosphere. In :math:`m^3 s^{-2}`.
    * omega : float
        The angular velocity of the earth. In :math:`rad s^{-1}`.

    """

    def __init__(self, name, a, f, GM, omega):
        self.name = name
        self._a = a
        self._f = f
        self._GM = GM
        self._omega = omega


    @property
    def a(self):
        return self._a

    @property
    def f(self):
        return self._f

    @property
    def GM(self):
        return self._GM

    @property
    def omega(self):
        return self._omega

    @property
    def b(self):
        return self.a*(1 - self.f)

    @property
    def E(self):
        return math.sqrt(self.a**2 - self.b**2)

    @property
    def e_prime(self):
        return self.E/self.b

    @property
    def m(self):
        return (self.omega**2)*(self.a**2)*self.b/self.GM

    @property
    def gamma_a(self):
        bE = self.b/self.E
        atanEb = math.atan2(self.E, self.b)
        q0 = 0.5*((1 + 3*bE**2)*atanEb - 3*bE)
        q0prime = 3*(1 + bE**2)*(1 - bE*atanEb) - 1
        m = self.m
        return self.GM*(1 - m - m*self.e_prime*q0prime/(6*q0))/(self.a*self.b)

    @property
    def gamma_b(self):
        bE = self.b/self.E
        atanEb = math.atan2(self.E, self.b)
        q0 = 0.5*((1 + 3*bE**2)*atanEb - 3*bE)
        q0prime = 3*(1 + bE**2)*(1 - bE*atanEb) - 1
        return self.GM*(1 + self.m*self.e_prime*q0prime/(3*q0))/self.a**2


WGS84 = ReferenceEllipsoid(a=6378137, f=1/298.257223563, GM=3986004.418e8,
                           omega=7292115e-11,
                           name="World Geodetic System 1984")


def gamma_somigliana(latitude, ellipsoid=WGS84):
    '''
    Calculate the normal gravity by using Somigliana's formula.

    This formula computes normal gravity **on** the ellipsoid (height = 0).

    Parameters:

    * latitude : float or numpy array
        The latitude where the normal gravity will be computed (in degrees)
    * ellipsoid : :class:`~fatiando.gravmag.normal_gravity.ReferenceEllipsoid`
        The reference ellipsoid used.

    Returns:

    * gamma : float or numpy array
        The computed normal gravity (in mGal).

    '''
    lat = numpy.deg2rad(latitude)
    sin2 = numpy.sin(lat)**2
    cos2 = numpy.cos(lat)**2
    top = ((ellipsoid.a*ellipsoid.gamma_a)*cos2
           + (ellipsoid.b*ellipsoid.gamma_b)*sin2)
    bottom = numpy.sqrt(ellipsoid.a**2*cos2 + ellipsoid.b**2*sin2)
    gamma = top/bottom
    return utils.si2mgal(gamma)


def gamma_somigliana_free_air(latitude, height, ellipsoid=WGS84):
    r'''
    Calculate the normal gravity at a height using Somigliana's formula and the
    free-air correction.

    Parameters:

    * latitude : float or numpy array
        The latitude where the normal gravity will be computed (in degrees)
    * height : float or numpy array
        The height of computation (in meters). Should be ellipsoidal
        (geometric) heights for geophysical purposes.
    * ellipsoid : :class:`~fatiando.gravmag.normal_gravity.ReferenceEllipsoid`
        The reference ellipsoid used.

    Returns:

    * gamma : float or numpy array
        The computed normal gravity (in mGal).

    '''
    fa = -0.3086*height
    gamma = gamma_somigliana(latitude, ellipsoid=ellipsoid) + fa
    return gamma


def gamma_closed_form(latitude, height, ellipsoid=WGS84):
    """
    Calculate normal gravity at a height using the closed form expression of
    Li and Gotze (2001).

    Parameters:

    * latitude : float or numpy array
        The latitude where the normal gravity will be computed (in degrees)
    * height : float or numpy array
        The height of computation (in meters). Should be ellipsoidal
        (geometric) heights for geophysical purposes.
    * ellipsoid : :class:`~fatiando.gravmag.normal_gravity.ReferenceEllipsoid`
        The reference ellipsoid used.

    Returns:

    * gamma : float or numpy array
        The computed normal gravity (in mGal).

    """
    E2 = ellipsoid.E**2
    bE = ellipsoid.b/ellipsoid.E
    atanEb = math.atan2(ellipsoid.E, ellipsoid.b)
    lat = numpy.deg2rad(latitude)
    coslat = numpy.cos(lat)
    sinlat = numpy.sin(lat)
    tanlat = sinlat/coslat
    beta = numpy.arctan2(ellipsoid.b*tanlat, ellipsoid.a)
    sinbeta = numpy.sin(beta)
    cosbeta = numpy.cos(beta)
    zl2 = (ellipsoid.b*sinbeta + height*sinlat)**2
    rl2 = (ellipsoid.a*cosbeta + height*coslat)**2
    D = (rl2 - zl2)/E2
    R = (rl2 + zl2)/E2
    cosbetal = numpy.sqrt(0.5*(1 + R) - numpy.sqrt(0.25*(1 + R**2) - 0.5*D))
    cosbetal2 = cosbetal**2
    sinbetal2 = 1 - cosbetal2
    bl = numpy.sqrt(rl2 + zl2 - E2*cosbetal2)
    bl2 = bl**2
    blE = bl/ellipsoid.E
    atanEbl = numpy.arctan2(ellipsoid.E, bl)
    q0 = 0.5*((1 + 3*bE**2)*atanEb - 3*bE)
    q0l = 3*(1 + blE**2)*(1 - blE*atanEbl) - 1
    W = numpy.sqrt((bl2 + E2*sinbetal2)/(bl2 + E2))
    omega2 = ellipsoid.omega**2
    a2 = ellipsoid.a**2
    part1 = ellipsoid.GM/(bl2 + E2)
    part2 = -cosbetal2*bl*ellipsoid.omega**2
    part3 = 0.5*sinbetal2 - 1/6
    part4 = a2*ellipsoid.E*q0l*omega2/((bl2 + E2)*q0)
    gamma = utils.si2mgal((part1 + part2 + part3*part4)/W)
    return gamma
