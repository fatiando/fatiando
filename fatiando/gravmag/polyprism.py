r"""
Calculate the potential fields of the 3D prism with polygonal crossection using
the formula of Plouff (1976).

**Gravity**

First and second derivatives of the gravitational potential:

* :func:`~fatiando.gravmag.polyprism.gz`
* :func:`~fatiando.gravmag.polyprism.gxx`
* :func:`~fatiando.gravmag.polyprism.gxy`
* :func:`~fatiando.gravmag.polyprism.gxz`
* :func:`~fatiando.gravmag.polyprism.gyy`
* :func:`~fatiando.gravmag.polyprism.gyz`
* :func:`~fatiando.gravmag.polyprism.gzz`

**Magnetic**

There are functions to calculate the total-field anomaly and the 3 components
of magnetic induction:

* :func:`~fatiando.gravmag.polyprism.tf`
* :func:`~fatiando.gravmag.polyprism.bx`
* :func:`~fatiando.gravmag.polyprism.by`
* :func:`~fatiando.gravmag.polyprism.bz`

**Auxiliary Functions**

Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \int\int\int \frac{1}{r}
                  \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

with respect to the variables :math:`x`, :math:`y`, and :math:`z`.
In this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}

and :math:`\nu`, :math:`\eta`, :math:`\zeta` are the Cartesian
coordinates of an element inside the volume of a 3D prism with
polygonal crossection. These second derivatives are used to calculate
the total field anomaly and the gravity gradient tensor
components produced by a 3D prism with polygonal crossection.

* :func:`~fatiando.gravmag.polyprism.kernelxx`
* :func:`~fatiando.gravmag.polyprism.kernelxy`
* :func:`~fatiando.gravmag.polyprism.kernelxz`
* :func:`~fatiando.gravmag.polyprism.kernelyy`
* :func:`~fatiando.gravmag.polyprism.kernelyz`
* :func:`~fatiando.gravmag.polyprism.kernelzz`

**References**

Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
applications to magnetic terrain corrections, Geophysics, 41(4), 727-741.

----

"""
from __future__ import division

import numpy
from numpy import arctan2, log, sqrt

from .. import utils
from ..constants import SI2MGAL, SI2EOTVOS, G, CM, T2NT
try:
    from . import _polyprism
except ImportError:
    _polyprism = None


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    r"""
    Calculate the total-field anomaly of polygonal prisms.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pmx, pmy, pmz = pmag * fx, pmag * fy, pmag * fz
        else:
            pmx, pmy, pmz = pmag
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                             and pmag is None):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        else:
            mx, my, mz = pmx, pmy, pmz
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        _polyprism.tf(xp, yp, zp, x, y, z1, z2, mx, my, mz, fx, fy, fz, res)
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, prisms):
    """
    Calculates the x component of the magnetic induction produced by 3D
    prisms with polygonal crosssection.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bx: array
        The x component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        _polyprism.bx(xp, yp, zp, x, y, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def by(xp, yp, zp, prisms):
    """
    Calculates the y component of the magnetic induction produced by 3D
    prisms with polygonal crosssection.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * by: array
        The y component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        _polyprism.by(xp, yp, zp, x, y, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, prisms):
    """
    Calculates the z component of the magnetic induction produced by 3D
    prisms with polygonal crosssection.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bz: array
        The z component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        _polyprism.bz(xp, yp, zp, x, y, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{z}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in mGal!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 10 ** (-10)
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gz(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2MGAL
    return res


def gxx(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gxx(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gxy(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gxz(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gyy(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gyy(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gyz(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, prisms):
    r"""
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        _polyprism.gzz(xp, yp, zp, x, y, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def kernelxx(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x^2},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gxx(xp, yp, zp, x, y, z1, z2, 1, res)
    return res


def kernelxy(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial y},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gxy(xp, yp, zp, x, y, z1, z2, 1, res)
    return res


def kernelxz(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial z},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gxz(xp, yp, zp, x, y, z1, z2, 1, res)
    return res


def kernelyy(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y^2},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gyy(xp, yp, zp, x, y, z1, z2, 1, res)
    return res


def kernelyz(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y \partial z},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gyz(xp, yp, zp, x, y, z1, z2, 1, res)
    return res


def kernelzz(xp, yp, zp, prism):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial z^2},

    where

    .. math::

        \phi(x,y,z) = \int \int \int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    _polyprism.gzz(xp, yp, zp, x, y, z1, z2, 1, res)
    return res
