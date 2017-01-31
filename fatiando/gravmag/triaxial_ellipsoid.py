r"""
The potential fields of a homogeneous triaxial ellipsoid.
"""
from __future__ import division

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils
from .._our_duecredit import due, Doi


due.cite(Doi("XXXXXXXXXXXXXXXXX"),
         description='Forward modeling formula for triaxial ellipsoids.',
         path='fatiando.gravmag.triaxial_ellipsoid')


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


#def bz(xp, yp, zp, ellipsoids, pmag=None):
#    """
#    The z component of the magnetic induction.
#
#    The coordinate system of the input parameters is x -> North, y -> East and
#    z -> Down.
#
#    Input units should be SI. Output is in nT.
#
#    Parameters:
#
#    * xp, yp, zp : arrays
#        The x, y, and z coordinates where the anomaly will be calculated
#    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidTriaxial`
#        The ellipsoids. Ellipsoids must have the physical property
#        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or without
#        ``'k'`` and ``'remanence'`` will be ignored.
#    * pmag : [mx, my, mz] or None
#        A magnetization vector. If not None, will use this value instead of the
#        resultant magnetization of the ellipsoids. Use this, e.g., for
#        sensitivity matrix building.
#
#    Returns:
#
#    * bz : array
#        The z component of the magnetic induction
#
#    References:
#
#    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
#    Magnetic and gravity anomalies of a triaxial ellipsoid.
#
#    """
#    if pmag is not None:
#        pmx, pmy, pmz = pmag
#    res = 0
#    for ellipsoid in ellipsoids:
#        if ellipsoid is None:
#            continue
#        if 'k' not in ellipsoid.props and 'remanence' not in ellipsoid.props 
#        and pmag is None:
#            continue
#        if pmag is None:
#            mx, my, mz = sphere.props['magnetization']
#        else:
#            mx, my, mz = pmx, pmy, pmz
#        x = sphere.x - xp
#        y = sphere.y - yp
#        z = sphere.z - zp
#        r_sqr = x**2 + y**2 + z**2
#        # This is faster than r5 = r_sqrt**2.5
#        r = np.sqrt(r_sqr)
#        r_5 = r*r*r*r*r
#        volume = 4*np.pi*(sphere.radius**3)/3
#        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
#        dotprod = mx*x + my*y + mz*z
#        res += volume*(3*dotprod*z - r_sqr*mz)/r_5
#    res *= CM * T2NT
#    return res


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


def gzz(xp, yp, zp, ellipsoids, dens=None):
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
    

def kernelxx(xp, yp, zp, ellipsoid):
    r"""
    The second x derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='x')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='x')
    g = _gv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='x')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x1 - g)
    return res


def kernelxy(xp, yp, zp, ellipsoid):
    r"""
    The xy derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='x')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x2)
    return res


def kernelxz(xp, yp, zp, ellipsoid):
    r"""
    The xz derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='x')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x3)
    return res


def kernelyy(xp, yp, zp, ellipsoid):
    r"""
    The second y derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='y')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    g = _gv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x2 - g)
    return res


def kernelyz(xp, yp, zp, ellipsoid):
    r"""
    The yz derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='y')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x3)
    return res


def kernelzz(xp, yp, zp, ellipsoid):
    r"""
    The second z derivative of the kernel function in the
    ellipsoid system.

    .. math::

        \phi(x,y,z) = \int\int\int \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in SI.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * ellipsoid : :class:`fatiando.mesher.ellipsoidTriaxial`
        The ellipsoid.

    Returns:

    * res : array
        The function calculated on xp, yp, zp

    """
    
    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    dlamb = _dlamb (x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c, 
                    lamb, deriv='z')
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    g = _gv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    res = 2.*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c*(dlamb*h*x3 - g)
    return res


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
    
    
    output
    K: numpy array 2D - susceptibility tensor in the main system (in SI).
    '''
    
    assert k1 >= k2 >= k3, 'k1, k2 and k3 must be the maximum, \
        intermediate and minimum eigenvalues'
    
    assert (k1 > 0) and (k2 > 2) and (k3 > 0), 'k1, k2 and k3 must \
        be all positive'
    
    U = ellipsoid.V([alpha, gamma, delta])
    
    K = np.dot(U, np.diag([k1,k2,k3]))
    K = np.dot(K, U.T)
    
    return K
    
def _lamb (x, y, z, a, b, c):
    '''
    Calculates the parameter lambda.
    
    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    
    output
    lamb: numpy array 1D - parameter lambda for each point in the 
        ellipsoid system.
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
    
    # auxiliary variables
    p2 = a*a + b*b + c*c - x*x - y*y - z*z
    p1 = (b*c*b*c) + (a*c*a*c) + (a*b*a*b) - (b*b + c*c)*(x*x) \
        - (a*a + c*c)*(y*y) - (a*a + b*b)*(z*z)
    p0 = (a*b*c*a*b*c) - (b*c*x*b*c*x) - (a*c*y*a*c*y) - (a*b*z*a*b*z)
    Q = (3.*p1 - p2*p2)/9.
    R = (9.*p1*p2 - 27.*p0 - 2.*p2*p2*p2)/54.
    
    p3 = R/np.sqrt(-(Q*Q*Q))
    
    assert np.alltrue(p3 <= 1.), 'arccos argument greater than 1'
    
    assert np.alltrue(Q*Q*Q + R*R < 0), 'the polynomial discriminant \
        must be negative'

    theta = np.arccos(p3)

    lamb = 2.*np.sqrt(-Q)*np.cos(theta/3.) - p2/3.
    
#    assert np.max(np.abs(lamb*lamb*lamb + p2*(lamb*lamb) \
#        + p1*lamb + p0)) < 1e-10, \
#        'lambda must be a root of the cubic equation \
#        (lamb**3 + p2*(lamb**2) + p1*lamb + p0'
    
    return lamb, p2, p1, p0
    
def _dlamb (x, y, z, a, b, c, lamb, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.
    
    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial 
        ellipsoid.
    deriv: string - defines the coordinate with respect to which the
        derivative will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    dlamb_dv: numpy array 1D - derivative of lambda with respect to the
        coordinate v = x, y, z in the ellipsoid system.
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
        
    assert deriv in ['x','y','z'], 'deriv must represent a coordinate x, y or z'
    
    aux = _dlamb_aux(x, y, z, a, b, c, lamb)
    
    if deriv is 'x':
        dlamb_dv = (x/(a**2 + lamb))/aux
        
    if deriv is 'y':
        dlamb_dv = (y/(b**2 + lamb))/aux
        
    if deriv is 'z':
        dlamb_dv = (z/(c**2 + lamb))/aux
    
    return dlamb_dv
    
def _dlamb_aux (x, y, z, a, b, c, lamb):
    '''
    Calculates an auxiliary variable used to calculate the spatial 
    derivatives of the parameter lambda with respect to the 
    coordinates x, y and z in the ellipsoid system.
    
    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial 
        ellipsoid.
        
    output
    aux: numpy array 1D - auxiliary variable.
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
    
    aux1 = x/(a**2 + lamb)
    aux2 = y/(b**2 + lamb)
    aux3 = z/(c**2 + lamb)
    aux = aux1**2 + aux2**2 + aux3**2
    
    return aux
    
def _E_F_demag(a, b, c):
    '''
    Calculates the Legendre's normal elliptic integrals of first 
    and second kinds which are used to calculate the demagnetization
    factors.
        
    input:
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
        
    output:
    F - Legendre's normal elliptic integrals of first kind.
    E - Legendre's normal elliptic integrals of second kind.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'

    kappa = np.sqrt(((a**2-b**2)/(a**2-c**2)))
    phi = np.arccos(c/a)
    
    E = ellipeinc(phi, kappa**2)
    F = ellipkinc(phi, kappa**2)
 
    return E,F
    
def demag_factors (a, b, c):
    '''
    Calculates the demagnetization factors n11, n22 and n33.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    
    output
    n11: float - demagnetization factor along the semi-axis a (in SI).
    n22: float - demagnetization factor along the semi-axis b (in SI).
    n33: float - demagnetization factor along the semi-axis c (in SI).
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
    
    E, F = _E_F_demag(a, b, c)
    
    aux1 = (a*b*c)/np.sqrt((a**2 - c**2))
    n11 = (aux1/(a*a - b*b))*(F - E)    
    n22 = -n11 + (aux1/(b*b - c*c))*E - (c**2)/(b**2 - c**2)
    n33 = -(aux1/(b*b - c*c))*E + (b**2)/(b**2 - c**2)
    
    return n11, n22, n33
    
def magnetization(n11, n22, n33, K, H0, inc, dec, RM, incrm, decrm, V):
    '''
    Calculates the resultant magnetization corrected from
    demagnetization.
    
    input
    n11: float - demagnetization factor along the semi-axis a (in SI).
    n22: float - demagnetization factor along the semi-axis b (in SI).
    n33: float - demagnetization factor along the semi-axis c (in SI).
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
        
    assert n11 <= n22 <= n33, 'n11 must be smaller than n22 and \
        n22 must be smaller than n33'
        
    assert (n11 >= 0) and (n22 >= 0) and (n33 >= 0), 'n11, n22 and n33 must \
        be all positive or zero (for neglecting the self-demagnetization)'
        
    assert np.allclose(K.T, K), 'the susceptibility is a symmetrical tensor'
        
    N_tilde = np.diag([n11, n22, n33])
    K_tilde = np.dot(V.T, np.dot(K, V))
    H0_tilde = np.dot(V.T, utils.ang2vec(H0, inc, dec))
    RM_tilde = np.dot(V.T, utils.ang2vec(RM, incrm, decrm))
    
    # resultant magnetization in the ellipsoid system
    M_tilde = np.linalg.solve(np.identity(3) - np.dot(K_tilde, N_tilde), \
                              np.dot(K_tilde, H0_tilde) + RM_tilde)
                              
    return np.dot(V, M_tilde)
    
def _E_F_field(a, b, c, args=True):
    '''
    Calculates the Legendre's normal elliptic integrals of first 
    and second kinds which are used to calculate the potential
    fields outside the body.
        
    input:
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    args: boolean - if True, the function also returns the
        arguments of the elliptic integrals
        
    output:
    F: numpy array 1D - Legendre's normal elliptic integrals of first kind.
    E: numpy array 1D - Legendre's normal elliptic integrals of second kind.
    kappa: float - a argument of the elliptic integrals.
    phi: numpy array 1D - a argument of the elliptic integrals.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'

    kappa = np.sqrt(((a**2-b**2)/(a**2-c**2)))
    phi = np.arcsin(np.sqrt((a**2-c**2)/(a**2+lamb)))
    
    E = ellipeinc(phi, kappa**2)
    F = ellipkinc(phi, kappa**2)
 
    if args is True:
        return E, F, kappa, phi
    else:
        return E, F
    
def _hv (a, b, c, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial 
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable hv will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    hv: numpy array 1D - auxiliary variable.
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
    
    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"
    
    aux1 = a**2 + lamb
    aux2 = b**2 + lamb
    aux3 = c**2 + lamb
    R = np.sqrt(aux1*aux2*aux3)
    
    if v is 'x':
        hv = 1./(aux1*R)
        
    if v is 'y':
        hv = 1./(aux2*R)
        
    if v is 'z':
        hv = 1./(aux3*R)
    
    return hv
    
def _gv (a, b, c, lamb, v='x'):
    '''
    Diagonal terms of the depolarization tensor defined outside the 
    ellipsoidal body. These terms depend on the 
    Legendre's normal elliptic integrals of first and second kinds.
    
    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial 
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.
        
    output
    gv: numpy array 1D - auxiliary variable.
    '''
    
    assert a > b > c, 'a must be greater than b and b must be greater than c'
    
    assert (a > 0) and (b > 2) and (c > 0), 'a, b and c must \
        be all positive'
    
    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"
    
    if v is 'x':
        E, F = _E_F_field(a, b, c, args=False)
        aux1 = 2./((a*a - b*b)*np.sqrt(a*a - c*c))
        gv = aux1*(F - E)
        
    if v is 'y':
        E, F, kappa, phi = _E_F_field(a, b, c)
        aux1 = 2*np.sqrt(a*a - c*c)/((a*a - b*b)*(b*b - c*c))
        aux2 = (b*b - c*c)/(a*a - c*c)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux3 = ((kappa*kappa)*sinphi*cosphi)/np.sqrt(1. - (kappa*sinphi*kappa*sinphi))
        gv = aux1*(E - aux2*F - aux3)
        
    if v is 'z':
        E, F, kappa, phi = _E_F_field(a, b, c)
        aux1 = 2./((a*a - b*b)*np.sqrt(a*a - c*c))
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux2 = (sinphi*np.sqrt(1. - (kappa*sinphi*kappa*sinphi)))/cosphi
        gv = aux1*(aux2 - E)
    
    return gv