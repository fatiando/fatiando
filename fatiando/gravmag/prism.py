"""
Calculate the potential fields of the 3D right rectangular prism.

.. note:: All input units are SI. Output is in conventional units: SI for the
    gravitatonal potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East and z -> Down.

**Gravity**

The gravitational fields are calculated using the formula of Nagy et al.
(2000). Available functions are:

* :func:`~fatiando.gravmag.prism.potential`
* :func:`~fatiando.gravmag.prism.gx`
* :func:`~fatiando.gravmag.prism.gy`
* :func:`~fatiando.gravmag.prism.gz`
* :func:`~fatiando.gravmag.prism.gxx`
* :func:`~fatiando.gravmag.prism.gxy`
* :func:`~fatiando.gravmag.prism.gxz`
* :func:`~fatiando.gravmag.prism.gyy`
* :func:`~fatiando.gravmag.prism.gyz`
* :func:`~fatiando.gravmag.prism.gzz`

**Magnetic**

Available fields are the total-field anomaly (using the formula of
Bhattacharyya, 1964) and x, y, z components of the magnetic induction:

* :func:`~fatiando.gravmag.prism.tf`.
* :func:`~fatiando.gravmag.prism.bx`.
* :func:`~fatiando.gravmag.prism.by`.
* :func:`~fatiando.gravmag.prism.bz`.

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

----
"""
from __future__ import division

import numpy

from .. import utils
from ..constants import G, SI2EOTVOS, CM, T2NT, SI2MGAL
try:
    from . import _prism
except ImportError:
    _prism = None


def potential(xp, yp, zp, prisms, dens=None):
    """
    Calculates the gravitational potential.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units(!)!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('potential', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G
    return res

def gx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gx', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2MGAL
    return res

def gy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gy', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2MGAL
    return res

def gz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gz', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2MGAL
    return res

def gxx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gxx', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gxy', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gxz', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gyy', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gyz', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gravity_kernels('gzz', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                               density, res)
    res *= G*SI2EOTVOS
    return res

def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    """
    Calculate the total-field magnetic anomaly of prisms.

    .. note:: Input units are SI. Output is in nT

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. *prisms* can also be a :class:`~fatiando.mesher.PrismMesh`.
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
    if len(xp) != len(yp) != len(zp):
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            mx, my, mz = pmag*fx, pmag*fy, pmag*fz
        else:
            mx, my, mz = pmag
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag*fx, mag*fy, mag*fz
            else:
                mx, my, mz = mag
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.magnetic_kernels('tf', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                                mx, my, mz, fx, fy, fz, res)
    res *= CM*T2NT
    return res

def bx(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the x component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.magnetic_kernels('bx', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                                mx, my, mz, 0, 0, 0, res)
    res *= CM*T2NT
    return res

def by(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the y component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.magnetic_kernels('by', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                                mx, my, mz, 0, 0, 0, res)
    res *= CM*T2NT
    return res

def bz(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the z component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.magnetic_kernels('bz', xp, yp, zp, x1,x2, y1, y2, z1, z2,
                                mx, my, mz, 0, 0, 0, res)
    res *= CM*T2NT
    return res

def kernelxx(xp, yp, zp, prism):
    """
    Calculates the :math:`V_1` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kxx = kernelxx(xp, yp, zp, model)
    >>> for k in kxx: print '%10.5f' % k
       0.01688
      -0.06776
      -0.06776
       0.01688
       0.14316
      -0.03364
      -0.03364
       0.14316
       0.01688
      -0.06776
      -0.06776
       0.01688

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((Y1*Z1),(X1*R111))
    res +=  numpy.arctan2((Y1*Z2),(X1*R112))
    res +=  numpy.arctan2((Y2*Z1),(X1*R121))
    res += -numpy.arctan2((Y2*Z2),(X1*R122))
    res +=  numpy.arctan2((Y1*Z1),(X2*R211))
    res += -numpy.arctan2((Y1*Z2),(X2*R212))
    res += -numpy.arctan2((Y2*Z1),(X2*R221))
    res +=  numpy.arctan2((Y2*Z2),(X2*R222))

    return res

def kernelyy(xp, yp, zp, prism):
    """
    Calculates the :math:`V_4` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kyy = kernelyy(xp, yp, zp, model)
    >>> for k in kyy: print '%10.5f' % k
       0.01283
       0.11532
       0.11532
       0.01283
      -0.09243
      -0.53592
      -0.53592
      -0.09243
       0.01283
       0.11532
       0.11532
       0.01283

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((X1*Z1),(Y1*R111))
    res +=  numpy.arctan2((X1*Z2),(Y1*R112))
    res +=  numpy.arctan2((X1*Z1),(Y2*R121))
    res += -numpy.arctan2((X1*Z2),(Y2*R122))
    res +=  numpy.arctan2((X2*Z1),(Y1*R211))
    res += -numpy.arctan2((X2*Z2),(Y1*R212))
    res += -numpy.arctan2((X2*Z1),(Y2*R221))
    res +=  numpy.arctan2((X2*Z2),(Y2*R222))

    return res

def kernelzz(xp, yp, zp, prism):
    """
    Calculates the :math:`V_6` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kzz = kernelzz(xp, yp, zp, model)
    >>> for k in kzz: print '%10.5f' % k
      -0.02971
      -0.04755
      -0.04755
      -0.02971
      -0.05072
       0.56956
       0.56956
      -0.05072
      -0.02971
      -0.04755
      -0.04755
      -0.02971

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((X1*Y1),(Z1*R111))
    res +=  numpy.arctan2((X1*Y1),(Z2*R112))
    res +=  numpy.arctan2((X1*Y2),(Z1*R121))
    res += -numpy.arctan2((X1*Y2),(Z2*R122))
    res +=  numpy.arctan2((X2*Y1),(Z1*R211))
    res += -numpy.arctan2((X2*Y1),(Z2*R212))
    res += -numpy.arctan2((X2*Y2),(Z1*R221))
    res +=  numpy.arctan2((X2*Y2),(Z2*R222))

    return res

def kernelxy(xp, yp, zp, prism):
    """
    Calculates the :math:`V_2` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kxy = kernelxy(xp, yp, zp, model)
    >>> for k in kxy: print '%10.5f' % k
       0.05550
       0.07324
      -0.07324
      -0.05550
      -0.00000
      -0.00000
       0.00000
       0.00000
      -0.05550
      -0.07324
       0.07324
       0.05550

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    dummy = 10.**(-10) # Used to avoid singularities
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((Z1 + R111) + dummy)
    res +=  numpy.log((Z2 + R112) + dummy)
    res +=  numpy.log((Z1 + R121) + dummy)
    res += -numpy.log((Z2 + R122) + dummy)
    res +=  numpy.log((Z1 + R211) + dummy)
    res += -numpy.log((Z2 + R212) + dummy)
    res += -numpy.log((Z1 + R221) + dummy)
    res +=  numpy.log((Z2 + R222) + dummy)

    return -res

def kernelxz(xp, yp, zp, prism):
    """
    Calculates the :math:`V_3` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kxz = kernelxz(xp, yp, zp, model)
    >>> for k in kxz: print '%10.5f' % k
       0.02578
       0.03466
      -0.03466
      -0.02578
       0.10661
       0.94406
      -0.94406
      -0.10661
       0.02578
       0.03466
      -0.03466
      -0.02578

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    dummy = 10.**(-10) # Used to avoid singularities
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((Y1 + R111) + dummy)
    res +=  numpy.log((Y1 + R112) + dummy)
    res +=  numpy.log((Y2 + R121) + dummy)
    res += -numpy.log((Y2 + R122) + dummy)
    res +=  numpy.log((Y1 + R211) + dummy)
    res += -numpy.log((Y1 + R212) + dummy)
    res += -numpy.log((Y2 + R221) + dummy)
    res +=  numpy.log((Y2 + R222) + dummy)

    return -res

def kernelyz(xp, yp, zp, prism):
    """
    Calculates the :math:`V_5` integral.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:

    >>> from fatiando import gridder
    >>> from fatiando.mesher import Prism
    >>> from fatiando.gravmag import prism
    >>> #Create a model
    >>> model = Prism(-200.0, 200.0, -300.0, 300.0, 100.0, 500.0,
    ...                                             {'density':1.})
    >>> # Create a regular grid at 100m height
    >>> shape = (3, 4)
    >>> area = [-900, 900, -900, 900]
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the function
    >>> kyz = kernelyz(xp, yp, zp, model)
    >>> for k in kyz: print '%10.5f' % k
       0.02463
       0.09778
       0.09778
       0.02463
      -0.00000
       0.00000
      -0.00000
      -0.00000
      -0.02463
      -0.09778
      -0.09778
      -0.02463

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    dummy = 10.**(-10) # Used to avoid singularities
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((X1 + R111) + dummy)
    res +=  numpy.log((X1 + R112) + dummy)
    res +=  numpy.log((X1 + R121) + dummy)
    res += -numpy.log((X1 + R122) + dummy)
    res +=  numpy.log((X2 + R211) + dummy)
    res += -numpy.log((X2 + R212) + dummy)
    res += -numpy.log((X2 + R221) + dummy)
    res +=  numpy.log((X2 + R222) + dummy)

    return -res
