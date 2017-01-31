r"""
The potential fields of a homogeneous oblate ellipsoid.
"""
from __future__ import division

import numpy as np

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils
from .._our_duecredit import due, Doi


due.cite(Doi("XXXXXXXXXXXXXXXXX"),
         description='Forward modeling formula for oblate ellipsoids.',
         path='fatiando.gravmag.oblate_ellipsoid')


def tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    r"""
    The total-field magnetic anomaly.

    The anomaly is defined as (Blakely, 1995):

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is a
    reference (regional) field.

    The anomaly of a homogeneous sphere can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the sphere.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

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

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    The x component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

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
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    The y component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

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
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    The z component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

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
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bz : array
        The z component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_z` gravitational acceleration component.

    .. math::

        g_z(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3} \dfrac{z - z'}{r^3}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in mGal.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{xx}` gravity gradient component.

    .. math::

        g_{xx}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3 (x - x')^2 - r^2}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{xy}` gravity gradient component.

    .. math::

        g_{xy}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3(x - x')(y - y')}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{xz}` gravity gradient component.

    .. math::

        g_{xz}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3(x - x')(z - z')}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{yy}` gravity gradient component.

    .. math::

        g_{yy}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3(y - y')^2 - r^2}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{yz}` gravity gradient component.

    .. math::

        g_{yz}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3(y - y')(z - z')}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    r"""
    The :math:`g_{zz}` gravity gradient component.

    .. math::

        g_{zz}(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3}
            \dfrac{3(z - z')^2 - r^2}{r^5}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in Eotvos.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

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
    The second x derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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
    The xy derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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
    The xz derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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
    The second y derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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
    The yz derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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
    The second z derivative of the kernel function

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi radius^3 \frac{1}{r}

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : :class:`fatiando.mesher.Sphere`
        The sphere.

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

##############################################################################
    
def x1x2x3 (xp, yp, zp, xc, yc, zc, V):
    '''
    Calculates the x, y and z coordinates referred to the
    ellipsoid coordinate system.
    
    input
    xp: numpy array 1D - x coordinates in the main system (in meters).
    yp: numpy array 1D - y coordinates in the main system (in meters).
    zp: numpy array 1D - z coordinates in the main system (in meters).
    xc: float - x coordinate of the ellipsoid center in the main 
                system (in meters).
    yc: float - y coordinate of the ellipsoid center in the main 
                system (in meters).
    zc: float - z coordinate of the ellipsoid center in the main 
                system (in meters).
    V: numpy array 2D - coordinate transformation matrix.

    output
    x1: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    x2: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    x3: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    '''
    
    assert xp.size == yp.size == zp.size, \
        'xp, yp and zp must have the same size'
    
    assert np.allclose(np.dot(V.T, V), np.identity(3)), \
        'V must be a valid coordinate transformation matrix'
        
    x1 = V[0,0]*(xp - xc) + V[1,0]*(yp - yc) + V[2,0]*(zp - zc)
    x2 = V[0,1]*(xp - xc) + V[1,1]*(yp - yc) + V[2,1]*(zp - zc)
    x3 = V[0,2]*(xp - xc) + V[1,2]*(yp - yc) + V[2,2]*(zp - zc)
    
    return x1, x2, x3
    

##############################################################################

def K (k1, k2, k3, alpha, gamma, delta, ellipsoid):
    '''
    Calculates the susceptibility tensor (in SI) in the main system.
    
    input
    k1: float - maximum eigenvalue of the susceptibility matrix K.
    k2: float - intermediate eigenvalue of the susceptibility matrix K.
    k3: float - minimum eigenvalue of the susceptibility matrix K.
    alpha: float - orientation angle (in degrees) of the major 
    susceptibility axis.
    gamma: float - orientation angle (in degrees) of the intermediate 
    susceptibility axis.
    delta: float - orientation angle (in degrees) of the minor 
    susceptibility axis.
    ellipsoid : :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoid.
    
    output
    K: numpy array 2D - susceptibility tensor in the main system (in SI).
    '''
    
    assert k1 >= k2 >= k3, 'k1, k2 and k3 must be the maximum, \
        intermediate and minimum eigenvalues'
    
    assert (k1 > 0) and (k2 > 2) and (k3 > 0), 'k1, k2 and k3 must \
        be all positive'
    
    U = ellipsoid.V([azimuth, gamma, delta])
    
    K = np.dot(U, np.diag([k1,k2,k3]))
    K = np.dot(K, U.T)
    
    return K
    
def _lamb (x1, x2, x3, a, b):
    '''
    Calculates the parameter lambda.
    
    input
    x1: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    x2: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    x3: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    
    output
    lamb: numpy array 1D - parameter lambda for each point in the 
        ellipsoid system.
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be positive'
    
    # auxiliary variables
    p1 = a*a + b*b -x1*x1 - x2*x2 - x3*x3
    p0 = a*a*b*b -b*b*x1*x1 - a*a*(x2*x2 + x3*x3)

    delta = np.sqrt(p1**2 - 4*p0)

    lamb = (-p1 + delta)/2.
    
    assert (lamb**2 + p1*lamb + p0) <= 1e-15, \
        'lambda must be a root of the quadratic equation \
        (lamb**2 + p1*lamb + p0'
    
    return lamb
    
def _dlamb (x1, x2, x3, a, b, lamb, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.
    
    input
    x1: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    x2: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    x3: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the oblate 
        ellipsoid.
    deriv: string - defines the coordinate with respect to which the
        derivative will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    dlamb_dv: numpy array 1D - derivative of lambda with respect to the
        coordinate v = x, y, z in the ellipsoid system.
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be all positive'
        
    assert deriv in ['x','y','z'], 'deriv must represent a coordinate x, y or z'
    
    aux = _dlamb_aux(x1, x2, x3, a, b, lamb)
    
    if deriv is 'x':
        dlamb_dv = (x1/(a**2 + lamb))/aux
        
    if deriv is 'y':
        dlamb_dv = (x2/(b**2 + lamb))/aux
        
    if deriv is 'z':
        dlamb_dv = (x3/(b**2 + lamb))/aux
    
    return dlamb_dv
    
def _dlamb_aux (x1, x2, x3, a, b, lamb):
    '''
    Calculates an auxiliary variable used to calculate the spatial 
    derivatives of the parameter lambda with respect to the 
    coordinates x, y and z in the ellipsoid system.
    
    input
    x1: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    x2: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    x3: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the oblate 
        ellipsoid.
        
    output
    aux: numpy array 1D - auxiliary variable.
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be all positive'
    
    aux1 = x1/(a**2 + lamb)
    aux2 = x2/(b**2 + lamb)
    aux3 = x3/(b**2 + lamb)
    aux = aux1**2 + aux2**2 + aux3**2
    
    return aux

def demag_factors (a, b):
    '''
    Calculates the demagnetization factors n11 and n22.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    
    output
    n11: float - demagnetization factor along the semi-axis a (in SI).
    n22: float - demagnetization factor along the semi-axis b (in SI).
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be all positive'
    
    m = a/b
    
    aux1 = 1 - m*m
    aux2 = np.sqrt(aux1)
    
    n11 = (1/aux1)*(1 - (m*np.arccos(m))/aux2)
    n22 = 0.5*(1 - n11)
    
    return n11, n22
    
def magnetization(n11, n22, n33, K, H0, inc, dec, RM, incrm, decrm, V):
    '''
    Calculates the resultant magnetization corrected from
    demagnetization.
    
    input
    n11: float - demagnetization factor along the semi-axis a (in SI).
    n22: float - demagnetization factor along the semi-axis b (in SI).
    K: numpy array 2D - susceptibility tensor in the main system (in SI).
    H0: float - intensity of the local-geomagnetic field (in nT).
    inc: float - inclination of the local-geomagnetic field (in degrees)
        in the main coordinate system.
    dec: float - declination of the local-geomagnetic field (in degrees)
        in the main coordinate system.
    RM: float - intensity of the remanent magnetization (in A/m).
    incrm: float - inclination of the remanent magnetization (in degrees)
        in the main coordinate system.
    decrm: float - declination of the remanent magnetization (in degrees)
        in the main coordinate system.
    V: numpy array 2D - coordinate transformation matrix.

    output
    m: numpy array 1D - resultant magnetization (in A/m) in the 
        main coordinate system.
    '''
    
    assert np.allclose(np.dot(V.T, V), np.identity(3)), \
        'V must be a valid coordinate transformation matrix'
        
    assert n11 >= n22, 'n11 must be greater than n22'
        
    assert (n11 >= 0) and (n22 >= 0), 'n11 and n22 must \
        be all positive or zero (for neglecting the self-demagnetization)'
        
    assert np.allclose(K.T, K), 'the susceptibility is a symmetrical tensor'
        
    N_tilde = np.diag([n11, n22, n22])
    K_tilde = np.dot(V.T, np.dot(K, V))
    H0_tilde = np.dot(V.T, utils.ang2vec(H0, inc, dec))
    RM_tilde = np.dot(V.T, utils.ang2vec(RM, incrm, decrm))
    
    # resultant magnetization in the ellipsoid system
    M_tilde = np.linalg.solve(np.identity(3) - np.dot(K_tilde, N_tilde), \
                              np.dot(K_tilde, H0_tilde) + RM_tilde)
                              
    return np.dot(V, M_tilde)

def _hv (a, b, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the prolate 
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable hv will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    hv: numpy array 1D - auxiliary variable.
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be all positive'
    
    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"
    
    aux1 = a**2 + lamb
    aux2 = b**2 + lamb
    R = np.sqrt(aux1*aux2*aux2)
    
    if v is 'x':
        hv = 1./(aux1*R)
        
    if v is 'y' or 'z':
        hv = 1./(aux2*R)
    
    return hv
    
def _gv (a, b, lamb, v='x'):
    '''
    Diagonal terms of the depolarization tensor defined outside the 
    ellipsoidal body.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the oblate 
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    gv: numpy array 1D - auxiliary variable.
    '''
    
    assert b > a, 'b must be greater than a'
    
    assert (a > 0) and (b > 2), 'a and b must \
        be all positive'
    
    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"
    
    if v is 'x':
        aux1 = 1/((b*b - a*a)**1.5)
        aux2 = np.sqrt((b*b - a*a)/(a*a + lamb))
        aux3 = np.sqrt((b*b - a*a)*(a*a + lamb))/(b*b + lamb)
        gv = aux1*(np.atan(aux2) - aux3)
        
    if v is 'y' or 'z':
        aux1 = 2/((b*b - a*a)**1.5)
        aux2 = np.sqrt((b*b - a*a)/(a*a + lamb))
        gv = aux1*(aux2 - np.atan(aux2))
    
    return gv