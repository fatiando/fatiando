r"""
Calculate the potential fields of the 3D right rectangular prism.

.. note:: All input units are SI. Output is in conventional units: SI for the
    gravitatonal potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East and z -> Down.

The gravitational fields are calculated using the formula of Nagy et al.
(2000).

.. warning::

    The gxy, gxz, and gyz components have singularities when the computation
    point is aligned with the corners of the prism on the bottom, east, and
    north sides, respectively. In these cases, the above functions will move
    the computation point slightly to avoid these singularities. Unfortunately,
    this means that the result will not be as accurate **on those points**.



Available fields are the total-field anomaly (using the formula of
Bhattacharyya, 1964) and x, y, z components of the magnetic induction.


Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \int\int\int \frac{1}{r}
                  \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

with respect to the variables :math:`x`, :math:`y`, and :math:`z`.
In this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}

and :math:`\nu`, :math:`\eta`, :math:`\zeta` are the Cartesian
coordinates of an element inside the volume of a 3D prism.
These second derivatives are used to calculate
the total field anomaly and the gravity gradient tensor
components.

* :func:`~fatiando.gravmag.prism.kernelxx`
* :func:`~fatiando.gravmag.prism.kernelxy`
* :func:`~fatiando.gravmag.prism.kernelxz`
* :func:`~fatiando.gravmag.prism.kernelyy`
* :func:`~fatiando.gravmag.prism.kernelyz`
* :func:`~fatiando.gravmag.prism.kernelzz`

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

----
"""
from __future__ import division

import numpy as np

from ... import utils
from ...constants import G, SI2EOTVOS, CM, T2NT, SI2MGAL

def safe_atan(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    res = np.arctan2(y, x)
    res[y == 0] = 0
    res[(y > 0) & (x < 0)] -= np.pi
    res[(y < 0) & (x < 0)] += np.pi
    return res


def safe_log(x):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    res = np.log(x)
    res[x == 0] = 0
    return res


def potential(xp, yp, zp, prism, dens=None):
    """
    Calculates the gravitational potential of a prism.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prism : :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect. Prism
        must have the property ``'density'`` in it's ``props`` dictionary.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prism. Use this when you need to overwrite the prism's physical
        properties, like for sensitivity (Jacobian) matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = (x*y*safe_log(z + r)
                          + y*z*safe_log(x + r)
                          + x*z*safe_log(y + r)
                          - (0.5*x*x)*safe_atan(z*y, x*r)
                          - (0.5*y*y)*safe_atan(z*x, y*r)
                          - (0.5*z*z)*safe_atan(x*y, z*r))
                res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res


def gx(xp, yp, zp, prism, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component of a prism.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units and output in **mGal**

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prism : :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect. Prism
        must have the property ``'density'`` in it's ``props`` dictionary.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prism. Use this when you need to overwrite the prism's physical
        properties, like for sensitivity (Jacobian) matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(y*safe_log(z + r)
                           + z*safe_log(y + r)
                           - x*safe_atan(z*y, x*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2MGAL
    return re


def gy(xp, yp, zp, prism, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component of a prism.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units and output in **mGal**

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prism : :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect. Prism
        must have the property ``'density'`` in it's ``props`` dictionary.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prism. Use this when you need to overwrite the prism's physical
        properties, like for sensitivity (Jacobian) matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(z*safe_log(x + r)
                           + x*safe_log(z + r)
                           - y*safe_atan(x*z, y*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2MGAL
    return res


def gz(xp, yp, zp, prism, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component of a prism.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units and output in **mGal**

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prism : :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect. Prism
        must have the property ``'density'`` in it's ``props`` dictionary.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prism. Use this when you need to overwrite the prism's physical
        properties, like for sensitivity (Jacobian) matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(x*safe_log(y + r)
                           + y*safe_log(x + r)
                           - z*safe_atan(x*y, z*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gxx(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the bottom side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gxy(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the east side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gxz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gyy(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the north side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gyz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
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
        _prism.gzz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2EOTVOS
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            mx, my, mz = pmag * fx, pmag * fy, pmag * fz
        else:
            mx, my, mz = pmag
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.tf(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, fx, fy, fz,
                  res)
    res *= CM * T2NT
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
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
        _prism.bx(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
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
        _prism.by(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
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
    if xp.shape != yp.shape or xp.shape != zp.shape:
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
        _prism.bz(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def kernelxx(xp, yp, zp, prism):
    r"""
    Calculates the xx derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gxx(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res


def kernelyy(xp, yp, zp, prism):
    r"""
    Calculates the yy derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gyy(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res


def kernelzz(xp, yp, zp, prism):
    r"""
    Calculates the zz derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gzz(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res


def kernelxy(xp, yp, zp, prism):
    r"""
    Calculates the xy derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gxy(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res


def kernelxz(xp, yp, zp, prism):
    r"""
    Calculates the xz derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gxz(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res


def kernelyz(xp, yp, zp, prism):
    r"""
    Calculates the yz derivative of the function

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.Prism`
        The model used to calculate the field.

    Returns:

    * res : array
        The effect calculated on the computation points.

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    _prism.gyz(xp, yp, zp, x1, x2, y1, y2, z1, z2, 1, res)
    return res
