r"""
The potential fields of a homogeneous sphere.
"""
from __future__ import division, absolute_import

import numpy as np

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils
from .._our_duecredit import due, Doi


due.cite(Doi("10.1017/CBO9780511549816"),
         description='Forward modeling formula for spheres.',
         path='fatiando.gravmag.sphere')


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
