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

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils

try:
    from . import _sphere
except ImportError:
    _sphere = None


def tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    """
    Calculate the total-field anomaly of spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pmx, pmy, pmz = pmag * fx, pmag * fy, pmag * fz
        else:
            pmx, pmy, pmz = pmag
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props
                              and pmag is None):
            continue
        # Get the intensity and unit vector from the magnetization
        if pmag is None:
            mag = sphere.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        else:
            mx, my, mz = pmx, pmy, pmz
        _sphere.tf(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, fx, fy, fz, res)
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, spheres):
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.bx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


def by(xp, yp, zp, spheres):
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.by(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, spheres):
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.bz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in mGal!

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   density, res)
    res *= G * SI2MGAL
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gyy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gyz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gzz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gyy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gyz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gzz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res
