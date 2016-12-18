r"""
Calculate the potential fields of a homogeneous sphere.

**Magnetic**

Calculates the magnetic effect produced by an sphere. The functions are
based on Blakely (1995).

* :func:`~fatiando.gravmag.sphere.tf`: calculates the total-field anomaly
* :func:`~fatiando.gravmag.sphere.bx`: calculates the x component of the
  induction
* :func:`~fatiando.gravmag.sphere.by`: calculates the y component of the
  induction
* :func:`~fatiando.gravmag.sphere.bz`: calculates the z component of the
  induction

Remember that:

The magnetization :math:`\mathbf{M}` and the dipole moment :math:`\mathbf{m}`
are related with the volume V:

.. math::

    \mathbf{M} = \dfrac{\mathbf{m}}{V}.

The total-field anomaly is:

.. math::

    \Delta T = |\mathbf{T}| - |\mathbf{F}|,

where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is a
reference (regional) field. The forward modeling functions
:func:`~fatiando.gravmag.sphere.bx`, :func:`~fatiando.gravmag.sphere.by`,
and :func:`~fatiando.gravmag.sphere.bz` calculate the 3 components of the
field perturbation :math:`\Delta\mathbf{F}`

.. math::

    \Delta\mathbf{F} = \mathbf{T} - \mathbf{F}.

Then the total-field anomaly caused by the sphere is

.. math::

    \Delta T \approx \hat{\mathbf{F}}\cdot\Delta\mathbf{F}.

**Gravity**

Calculates the gravitational acceleration and gravity gradient tensor
components.

* :func:`~fatiando.gravmag.sphere.gz`
* :func:`~fatiando.gravmag.sphere.gxx`
* :func:`~fatiando.gravmag.sphere.gxy`
* :func:`~fatiando.gravmag.sphere.gxz`
* :func:`~fatiando.gravmag.sphere.gyy`
* :func:`~fatiando.gravmag.sphere.gyz`
* :func:`~fatiando.gravmag.sphere.gzz`


**Auxiliary Functions**

Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

with respect to the variables :math:`x`, :math:`y`, and :math:`z`. In
this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2},

and :math:`R` is the radius of a sphere with centre at the Cartesian
coordinates :math:`\nu`, :math:`\eta` and :math:`\zeta`.

These second derivatives are used to calculate the total field magnetic anomaly
and the gravity gradient tensor components.

* :func:`~fatiando.gravmag.sphere.kernelxx`
* :func:`~fatiando.gravmag.sphere.kernelxy`
* :func:`~fatiando.gravmag.sphere.kernelxz`
* :func:`~fatiando.gravmag.sphere.kernelyy`
* :func:`~fatiando.gravmag.sphere.kernelyz`
* :func:`~fatiando.gravmag.sphere.kernelzz`

**References**

Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic Applications,
Cambridge University Press.

----

"""
from __future__ import division

import numpy
import numpy as np

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils


# These are the second derivatives of the V = 1/r function that is used by the
# magnetic field component, total-field magnetic anomaly, gravity gradients,
# and the kernel functions.
def _v_xx(x, y, z, r_sqr, r_5):
    return (3*x**2 - r_sqr)/r_5


def _v_xy(x, y, z, r_sqr, r_5):
    return 3*x*y/r_5


def _v_xz(x, y, z, r_sqr, r_5):
    return 3*x*z/r_5


def _v_yy(x, y, z, r_sqr, r_5):
    return (3*y**2 - r_sqr)/r_5


def _v_yz(x, y, z, r_sqr, r_5):
    return 3*y*z/r_5


def _v_zz(x, y, z, r_sqr, r_5):
    return (3*z**2 - r_sqr)/r_5


def tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    """
    The total-field magnetic anomaly of homogeneous spheres.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * inc, dec : floats
        The inclination and declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    """
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~2x slower than this
        dotprod = mx*x + my*y + mz*z
        bx = (3*dotprod*x - r_sqr*mx)/r_5
        by = (3*dotprod*y - r_sqr*my)/r_5
        bz = (3*dotprod*z - r_sqr*mz)/r_5
        res += volume*(fx*bx + fy*by + fz*bz)
    res *= CM*T2NT
    return res


def bx(xp, yp, zp, spheres, pmag=None):
    """
    Calculates the x component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bx: array
        The x component of the magnetic induction

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*x - r_sqr*mx)/r_5
    res *= CM * T2NT
    return res


def by(xp, yp, zp, spheres, pmag=None):
    """
    Calculates the y component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * by: array
        The y component of the magnetic induction

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*y - r_sqr*my)/r_5
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, spheres, pmag=None):
    """
    Calculates the z component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bz: array
        The z component of the magnetic induction

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*z - r_sqr*mz)/r_5
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, spheres, dens=None):
    """
    The :math:`g_z` gravitational acceleration component.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in mGal.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r = np.sqrt(x**2 + y**2 + z**2)
        # This is faster than r3 = r_sqrt**1.5
        r_cb = r*r*r
        mass = density*4*np.pi*(sphere.radius**3)/3
        res += mass*z/r_cb
    res *= G*SI2MGAL
    return res


def gxx(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_xx(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def gxy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_xy(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def gxz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_xz(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def gyy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_yy(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def gyz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_yz(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def gzz(xp, yp, zp, spheres, dens=None):
    """
    The :math:`g_{zz}` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        res += density*volume*_v_zz(x, y, z, r_sqr, r_5)
    res *= G*SI2EOTVOS
    return res


def kernelxx(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_xx(x, y, z, r_sqr, r_5)
    return res


def kernelxy(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial y},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_xy(x, y, z, r_sqr, r_5)
    return res


def kernelxz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial z},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_xz(x, y, z, r_sqr, r_5)
    return res


def kernelyy(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_yy(x, y, z, r_sqr, r_5)
    return res


def kernelyz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y \partial z},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_yz(x, y, z, r_sqr, r_5)
    return res


def kernelzz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial z^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    x = sphere.x - xp
    y = sphere.y - yp
    z = sphere.z - zp
    r_sqr = x**2 + y**2 + z**2
    # This is faster than r5 = r_sqrt**2.5
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
    volume = 4*np.pi*(sphere.radius**3)/3
    res = volume*_v_zz(x, y, z, r_sqr, r_5)
    return res
