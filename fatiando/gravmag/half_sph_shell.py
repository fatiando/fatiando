"""
Forward modeling of the gravitational effects of half a spherical shell.

Can only calculate the effects in a point located at the pole at different
heights.

Components, gx, gy, gxy, gxz, and gyz are all equal to zero (0).

**Functions:**

* :func:`~fatiando.gravmag.half_sph_shell.potential`: the gravitational
  potential
* :func:`~fatiando.gravmag.half_sph_shell.gz`: the vertical component of the
  gravitational attraction
* :func:`~fatiando.gravmag.half_sph_shell.gxx`: the xx (North-North) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.half_sph_shell.gyy`: the yy (East-East) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.half_sph_shell.gzz`: the zz (radial-radial) component
  of the gravity gradient tensor


"""
import numpy

from fatiando.constants import MEAN_EARTH_RADIUS, G, SI2MGAL, SI2EOTVOS

def potential(heights, top, bottom, density, ref_radius=MEAN_EARTH_RADIUS):
    r = heights + ref_radius
    r1 = bottom + ref_radius
    r2 = top + ref_radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r**2 + rl**2)
        res += (-1)**(i)*((l**3 + rl**3)/(3.*r) - 0.5*rl**2)
    res *= 2*numpy.pi*G*density
    return res

def gz(heights, top, bottom, density, ref_radius=MEAN_EARTH_RADIUS):
    r = heights + ref_radius
    r1 = bottom + ref_radius
    r2 = top + ref_radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r**2 + rl**2)
        res += (-1)**(i)*((l**3 + rl**3)/(3.*r**2) - l)
    res *= 2*numpy.pi*G*density*SI2MGAL
    return res

def gzz(heights, top, bottom, density, ref_radius=MEAN_EARTH_RADIUS):
    r = heights + ref_radius
    r1 = bottom + ref_radius
    r2 = top + ref_radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r**2 + rl**2)
        res += (-1)**(i)*(2.*(l**3 + rl**3)/(3.*r**3) - l/r + r/l)
    res *= 2*numpy.pi*G*density*SI2EOTVOS
    return res

def gxx(heights, top, bottom, density, ref_radius=MEAN_EARTH_RADIUS):
    return -0.5*gzz(heights, top, bottom, density, ref_radius)

def gyy(heights, top, bottom, density, ref_radius=MEAN_EARTH_RADIUS):
    return gxx(heights, top, bottom, density, ref_radius)
