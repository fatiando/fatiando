"""
The potential fields of a homogeneous 3D prism with polygonal cross-section.
"""
from __future__ import division, absolute_import
from future.builtins import range

import numpy as np

from .. import utils
from ..constants import SI2MGAL, SI2EOTVOS, G, CM, T2NT
from .._our_duecredit import due, Doi


due.cite(Doi("10.1190/1.1440645"),
         description='Forward modeling formula for polygonal prisms.',
         path='fatiando.gravmag.polyprism')


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    r"""
    The total-field magnetic anomaly of polygonal prisms.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    res = 0
    for prism in prisms:
        if prism is None:
            continue
        if 'magnetization' not in prism.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        else:
            mx, my, mz = pmag
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bx = v1*mx + v2*my + v3*mz
        by = v2*mx + v4*my + v5*mz
        bz = v3*mx + v5*my + v6*mz
        res += fx*bx + fy*by + fz*bz
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, prisms):
    """
    x component of magnetic induction of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        res += v1*mx + v2*my + v3*mz
    res *= CM * T2NT
    return res


def by(xp, yp, zp, prisms):
    """
    y component of magnetic induction of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        res += v2*mx + v4*my + v5*mz
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, prisms):
    """
    z component of magnetic induction of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        res += v3*mx + v5*my + v6*mz
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, prisms):
    r"""
    z component of gravitational acceleration of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        nverts = prism.nverts
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        Z1_sqr = Z1**2
        Z2_sqr = Z2**2
        kernel = 0
        for k in range(nverts):
            Xk1 = x[k] - xp
            Yk1 = y[k] - yp
            Xk2 = x[(k + 1) % nverts] - xp
            Yk2 = y[(k + 1) % nverts] - yp
            p = Xk1*Yk2 - Xk2*Yk1
            p_sqr = p**2
            Qk1 = (Yk2 - Yk1)*Yk1 + (Xk2 - Xk1)*Xk1
            Qk2 = (Yk2 - Yk1)*Yk2 + (Xk2 - Xk1)*Xk2
            Ak1 = Xk1**2 + Yk1**2
            Ak2 = Xk2**2 + Yk2**2
            R1k1 = np.sqrt(Ak1 + Z1_sqr)
            R1k2 = np.sqrt(Ak2 + Z1_sqr)
            R2k1 = np.sqrt(Ak1 + Z2_sqr)
            R2k2 = np.sqrt(Ak2 + Z2_sqr)
            Ak1 = np.sqrt(Ak1)
            Ak2 = np.sqrt(Ak2)
            Bk1 = np.sqrt(Qk1**2 + p_sqr)
            Bk2 = np.sqrt(Qk2**2 + p_sqr)
            E1k1 = R1k1*Bk1
            E1k2 = R1k2*Bk2
            E2k1 = R2k1*Bk1
            E2k2 = R2k2*Bk2
            # Simplifying these arctans with, e.g., (Z2 - Z1)*arctan2(Qk2*p -
            # Qk1*p, p*p + Qk2*Qk1) doesn't work because of the restrictions
            # regarding the angles for that identity. The regression tests
            # fail for some points by a large amount.
            kernel += (Z2 - Z1)*(np.arctan2(Qk2, p) - np.arctan2(Qk1, p))
            kernel += Z2*(np.arctan2(Z2*Qk1, R2k1*p) -
                          np.arctan2(Z2*Qk2, R2k2*p))
            kernel += Z1*(np.arctan2(Z1*Qk2, R1k2*p) -
                          np.arctan2(Z1*Qk1, R1k1*p))
            Ck1 = Qk1*Ak1
            Ck2 = Qk2*Ak2
            # dummy helps prevent zero division and log(0) errors (that's why I
            # need to add it twice)
            # Simplifying these two logs with a single one is not worth it
            # because it would introduce two pow operations.
            kernel += 0.5*p*Ak1/(Bk1 + dummy)*np.log(
                (E1k1 - Ck1)*(E2k1 + Ck1)/((E1k1 + Ck1)*(E2k1 - Ck1) + dummy) +
                dummy)
            kernel += 0.5*p*(Ak2/(Bk2 + dummy))*np.log(
                (E2k2 - Ck2)*(E1k2 + Ck2)/((E2k2 + Ck2)*(E1k2 - Ck2) + dummy) +
                dummy)
        res += kernel*density
    res *= G*SI2MGAL
    return res


def gxx(xp, yp, zp, prisms):
    r"""
    xx component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelxx(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms):
    r"""
    xy component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelxy(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms):
    r"""
    xz component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelxz(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def gyy(xp, yp, zp, prisms):
    r"""
    yy component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelyy(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms):
    r"""
    yz component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelyz(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, prisms):
    r"""
    zz component of the gravity gradient tensor of a polygonal prism.

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

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += kernelzz(xp, yp, zp, prism)*density
    res *= G * SI2EOTVOS
    return res


def kernelxx(xp, yp, zp, prism):
    r"""
    The xx second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        g = X1 - Y1*n
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = g*Y2*atan_diff_d2/(p*d2) + n*p*atan_diff_d2/(d2)
        tmp -= g*Y1*atan_diff_d1/(p*d1) + n*p*atan_diff_d1/(d1)
        tmp += n*np.log(
            (Z2 + R12)*(Z1 + R21)/((Z1 + R11)*(Z2 + R22) + dummy) + dummy)
        tmp *= -1/(1 + n*n)
        kernel += tmp
    return kernel


def kernelxy(xp, yp, zp, prism):
    r"""
    The xy second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        g = X1 - Y1*n
        g_sqr = g*g
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = (g_sqr + g*n*Y2)*atan_diff_d2/(p*d2) - p*atan_diff_d2/d2
        tmp -= (g_sqr + g*n*Y1)*atan_diff_d1/(p*d1) - p*atan_diff_d1/d1
        tmp += np.log(
            (Z2 + R22)*(Z1 + R11)/((Z1 + R21)*(Z2 + R12) + dummy) + dummy)
        tmp *= 1/(1 + n*n)
        kernel += tmp
    return kernel


def kernelxz(xp, yp, zp, prism):
    r"""
    The xz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        n_sqr_p1 = n*n + 1
        g = X1 - Y1*n
        ng = n*g
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        # Collapsing these logs decreases the precision too much leading to a
        # larger difference with the prism code.
        log_r22 = np.log((R22 - d2)/(R22 + d2) + dummy)
        log_r21 = np.log((R21 - d2)/(R21 + d2) + dummy)
        log_r12 = np.log((R12 - d1)/(R12 + d1) + dummy)
        log_r11 = np.log((R11 - d1)/(R11 + d1) + dummy)
        log_diff_d1 = (0.5/d1)*(log_r12 - log_r11)
        log_diff_d2 = (0.5/d2)*(log_r22 - log_r21)
        tmp = (Y2*n_sqr_p1 + ng)*log_diff_d2
        tmp -= (Y1*n_sqr_p1 + ng)*log_diff_d1
        tmp *= -1/n_sqr_p1
        kernel += tmp
    return kernel


def kernelyy(xp, yp, zp, prism):
    r"""
    The yy second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        m = deltay/deltax
        c = Y1 - X1*m
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = c*X2*atan_diff_d2/(p*d2) + m*p*atan_diff_d2/d2
        tmp -= c*X1*atan_diff_d1/(p*d1) + m*p*atan_diff_d1/d1
        tmp += m*np.log(
            (Z2 + R12)*(Z1 + R21)/((Z2 + R22)*(Z1 + R11)) + dummy)
        tmp *= 1/(1 + m*m)
        kernel += tmp
    return kernel


def kernelyz(xp, yp, zp, prism):
    r"""
    The yz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        m = deltay/deltax
        m_sqr_p1 = m*m + 1
        c = Y1 - X1*m
        cm = c*m
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        # Same remark about collapsing logs as kernelxz
        log_r11 = np.log((R11 - d1)/(R11 + d1) + dummy)
        log_r12 = np.log((R12 - d1)/(R12 + d1) + dummy)
        log_r21 = np.log((R21 - d2)/(R21 + d2) + dummy)
        log_r22 = np.log((R22 - d2)/(R22 + d2) + dummy)
        tmp = (X2*m_sqr_p1 + cm)*(0.5/d2)*(log_r22 - log_r21)
        tmp -= (X1*m_sqr_p1 + cm)*(0.5/d1)*(log_r12 - log_r11)
        tmp *= 1/m_sqr_p1
        kernel += tmp
    return kernel


def kernelzz(xp, yp, zp, prism):
    r"""
    The zz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1
        deltay = Y2 - Y1
        # dist is only used in divisions. Add dummy to avoid zero division
        # errors if the two vertices coincide.
        dist = np.sqrt(deltax*deltax + deltay*deltay) + dummy
        cross = X1*Y2 - X2*Y1
        p = cross/dist
        d1 = (deltax*X1 + deltay*Y1)/dist
        d2 = (deltax*X2 + deltay*Y2)/dist
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        kernel += (np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21) -
                   np.arctan2(Z2*d1, p*R12) + np.arctan2(Z1*d1, p*R11))
    return kernel
