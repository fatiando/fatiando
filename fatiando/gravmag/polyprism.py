"""
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

The Total Field magnetic anomaly:

* :func:`~fatiando.gravmag.polyprism.tf`

**References**

Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
applications to magnetic terrain corrections, Geophysics, 41(4), 727-741.

----

"""
import numpy
from numpy import arctan2, log, sqrt

from fatiando import utils
from fatiando.constants import SI2MGAL, SI2EOTVOS, G, CM, T2NT


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    """
    Calculate the total-field anomaly of polygonal prisms.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
            pintensity = pmag
            pmx, pmy, pmz = fx, fy, fz
        else:
            pintensity = numpy.linalg.norm(pmag)
            pmx, pmy, pmz = numpy.array(pmag)/pintensity
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                             and pmag is None):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                intensity = mag
                mx, my, mz = fx, fy, fz
            else:
                intensity = numpy.linalg.norm(mag)
                mx, my, mz = numpy.array(mag)/intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Now calculate the total field anomaly
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            X1 = x[k] - xp
            Y1 = y[k] - yp
            X2 = x[(k + 1)%nverts] - xp
            Y2 = y[(k + 1)%nverts] - yp
            v1 = _integral_v1(X1, X2, Y1, Y2, Z1, Z2)
            v2 = _integral_v2(X1, X2, Y1, Y2, Z1, Z2)
            v3 = _integral_v3(X1, X2, Y1, Y2, Z1, Z2)
            v4 = _integral_v4(X1, X2, Y1, Y2, Z1, Z2)
            v5 = _integral_v5(X1, X2, Y1, Y2, Z1, Z2)
            v6 = _integral_v6(X1, X2, Y1, Y2, Z1, Z2)
            res += intensity*(
                      mx*(v1*fx + v2*fy + v3*fz)
                    + my*(v2*fx + v4*fy + v5*fz)
                    + mz*(v3*fx + v5*fy + v6*fz))
    res *= CM*T2NT
    return res

def gz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{z}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    dummy = 10**(-10)
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        Z1_sqr = Z1**2
        Z2_sqr = Z2**2
        kernel = numpy.zeros_like(res)
        for k in range(nverts):
            Xk1 = x[k] - xp
            Yk1 = y[k] - yp
            Xk2 = x[(k + 1)%nverts] - xp
            Yk2 = y[(k + 1)%nverts] - yp
            p = Xk1*Yk2 - Xk2*Yk1
            p_sqr = p**2
            Qk1 = (Yk2 - Yk1)*Yk1 + (Xk2 - Xk1)*Xk1
            Qk2 = (Yk2 - Yk1)*Yk2 + (Xk2 - Xk1)*Xk2
            Ak1 = Xk1**2 + Yk1**2
            Ak2 = Xk2**2 + Yk2**2
            R1k1 = sqrt(Ak1 + Z1_sqr)
            R1k2 = sqrt(Ak2 + Z1_sqr)
            R2k1 = sqrt(Ak1 + Z2_sqr)
            R2k2 = sqrt(Ak2 + Z2_sqr)
            Ak1 = sqrt(Ak1)
            Ak2 = sqrt(Ak2)
            Bk1 = sqrt(Qk1**2 + p_sqr)
            Bk2 = sqrt(Qk2**2 + p_sqr)
            E1k1 = R1k1*Bk1
            E1k2 = R1k2*Bk2
            E2k1 = R2k1*Bk1
            E2k2 = R2k2*Bk2
            kernel += (Z2 - Z1)*(arctan2(Qk2, p) - arctan2(Qk1, p))
            kernel += Z2*(arctan2(Z2*Qk1, R2k1*p) - arctan2(Z2*Qk2, R2k2*p))
            kernel += Z1*(arctan2(Z1*Qk2, R1k2*p) - arctan2(Z1*Qk1, R1k1*p))
            Ck1 = Qk1*Ak1
            Ck2 = Qk2*Ak2
            # dummy helps prevent zero division errors
            kernel += 0.5*p*(Ak1/(Bk1 + dummy))*(
                log((E1k1 - Ck1)/(E1k1 + Ck1 + dummy) + dummy) -
                log((E2k1 - Ck1)/(E2k1 + Ck1 + dummy) + dummy))
            kernel += 0.5*p*(Ak2/(Bk2 + dummy))*(
                log((E2k2 - Ck2)/(E2k2 + Ck2 + dummy) + dummy) -
                log((E1k2 - Ck2)/(E1k2 + Ck2 + dummy) + dummy))
        res = res + kernel*density
    res *= G*SI2MGAL
    return res

def gxx(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v1(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v2(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v3(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v4(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v5(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

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
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += density*_integral_v6(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res

def _integral_v1(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the first element of the V matrix (gxx components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0/aux1)
    g = X1 - (Y1*n)
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux3 = (X1*Y2) - (X2*Y1)
    p = ((aux3/aux2)) + dummy
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2*d2), (p*R22))
    aux11 = arctan2((Z1*d2), (p*R21))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d2))
    aux14 = ((p*aux12)/d2)
    res = (g*Y2*aux13) + (n*aux14)
    aux10 = arctan2((Z2*d1), (p*R12))
    aux11 = arctan2((Z1*d1), (p*R11))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d1))
    aux14 = ((p*aux12)/d1)
    res -= (g*Y1*aux13) + (n*aux14)
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (n*(aux15 - aux14))
    aux0 = (1.0/(1.0 + (n*n)))
    res *= -aux0
    return res

def _integral_v2(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the second element of the V matrix (gxy components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0/aux1)
    g = X1 - (Y1*n)
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux3 = (X1*Y2) - (X2*Y1)
    p = ((aux3/aux2)) + dummy
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2*d2), (p*R22))
    aux11 = arctan2((Z1*d2), (p*R21))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d2))
    aux14 = ((p*aux12)/d2)
    res = (((g*g) + (g*n*Y2))*aux13) - aux14
    aux10 = arctan2((Z2*d1), (p*R12))
    aux11 = arctan2((Z1*d1), (p*R11))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d1))
    aux14 = ((p*aux12)/d1)
    res -= (((g*g) + (g*n*Y1))*aux13) - aux14
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (aux14 - aux15)
    aux0 = (1.0/(1.0 + (n*n)))
    res *= aux0
    return res

def _integral_v3(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the third element of the V matrix (gxz components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0/aux1)
    g = X1 - (Y1*n)
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = log((((R11 - d1)/(R11 + d1)) + dummy))
    aux11 = log((((R12 - d1)/(R12 + d1)) + dummy))
    aux12 = log((((R21 - d2)/(R21 + d2)) + dummy))
    aux13 = log((((R22 - d2)/(R22 + d2)) + dummy))
    aux14 = (1.0/(2*d1))
    aux15 = (1.0/(2*d2))
    aux16 = aux15*(aux13 - aux12)
    res = (Y2*(1.0 + (n*n)) + g*n)*aux16
    aux16 = aux14*(aux11 - aux10)
    res -= (Y1*(1.0 + (n*n)) + g*n)*aux16
    aux0 = (1.0/(1.0 + (n*n)))
    res *= -aux0
    return res

def _integral_v4(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the forth element of the V matrix (gyy components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    m = (aux1/aux0)
    c = Y1 - (X1*m)
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux3 = (X1*Y2) - (X2*Y1)
    p = ((aux3/aux2)) + dummy
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2*d2), (p*R22))
    aux11 = arctan2((Z1*d2), (p*R21))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d2))
    aux14 = ((p*aux12)/d2)
    res = (c*X2*aux13) + (m*aux14)
    aux10 = arctan2((Z2*d1), (p*R12))
    aux11 = arctan2((Z1*d1), (p*R11))
    aux12 = aux10 - aux11
    aux13 = (aux12/(p*d1))
    aux14 = ((p*aux12)/d1)
    res -= (c*X1*aux13) + (m*aux14)
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (m*(aux15 - aux14))
    aux1 = (1.0/(1.0 + (m*m)))
    res *= aux1
    return res

def _integral_v5(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the fith element of the V matrix (gyz components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    m = (aux1/aux0)
    c = Y1 - (X1*m)
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = log((((R11 - d1)/(R11 + d1)) + dummy))
    aux11 = log((((R12 - d1)/(R12 + d1)) + dummy))
    aux12 = log((((R21 - d2)/(R21 + d2)) + dummy))
    aux13 = log((((R22 - d2)/(R22 + d2)) + dummy))
    aux14 = (1.0/(2*d1))
    aux15 = (1.0/(2*d2))
    aux16 = aux15*(aux13 - aux12)
    res = (X2*(1.0 + (m*m)) + c*m)*aux16
    aux16 = aux14*(aux11 - aux10)
    res -= (X1*(1.0 + (m*m)) + c*m)*aux16
    aux1 = (1.0/(1.0 + (m*m)))
    res *= aux1
    return res

def _integral_v6(X1, X2, Y1, Y2, Z1, Z2):
    """
    Calculates the sixth element of the V matrix (gzz components)
    """
    dummy = 10.**(-10) # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    aux2 = sqrt((aux0*aux0) + (aux1*aux1))
    aux3 = (X1*Y2) - (X2*Y1)
    p = ((aux3/aux2)) + dummy
    aux4 = (aux0*X1) + (aux1*Y1)
    aux5 = (aux0*X2) + (aux1*Y2)
    d1 = ((aux4/aux2)) + dummy
    d2 = ((aux5/aux2)) + dummy
    aux6 = (X1*X1) + (Y1*Y1)
    aux7 = (X2*X2) + (Y2*Y2)
    aux8 = Z1*Z1
    aux9 = Z2*Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2*d2), (p*R22))
    aux11 = arctan2((Z1*d2), (p*R21))
    aux12 = aux10 - aux11
    res = aux12
    aux10 = arctan2((Z2*d1), (p*R12))
    aux11 = arctan2((Z1*d1), (p*R11))
    aux12 = aux10 - aux11
    res -= aux12
    return res

def kernelxx(xp, yp, zp, prisms):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x^2},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \int \int \int \frac{1}{r} 
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-5000, 5000, -7000, 7000]
    >>> shape = (8, 6)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[2301.2551, -2824.2678],
    ...             [2981.1716, -1255.2301],
    ...             [2824.2678, 993.72388],
    ...             [1830.5439, 2981.1716],
    ...             [261.50629, 3817.9917],
    ...             [-523.01257, 5543.9331],
    ...             [-2039.7489, 5334.728],
    ...             [-2144.3516, 4027.1965],
    ...             [-1621.3389, 2301.2551],
    ...             [-523.01257, 366.1088],
    ...             [-836.82007, -1150.6276],
    ...             [-2144.3516, -1830.5439],
    ...             [-3294.979, -2196.6528],
    ...             [-3556.4854, -3765.6904],
    ...             [-2510.4602, -5177.8242],
    ...             [-209.20502, -5177.8242],
    ...             [1673.6401, -5648.5356],
    ...             [3922.5942, -5177.8242],
    ...             [3451.8828, -3504.1841]]
    >>> model = [mesher.PolygonalPrism(vertices, 50, 760, {'density':1})]
    >>> # Calculate the kernelxx function
    >>> kxx = kernelxx(xp, yp, zp, model)
    >>> for k in kxx: print '%15.8e' % k
     6.21711351e-02
    -4.09875028e-02
    -1.58416927e-01
    -2.28240013e-01
    -1.29999533e-01
     7.23518953e-02
     3.12138319e-01
     7.34696925e-01
    -3.75256926e-01
    -6.47502840e-01
    -8.78984272e-01
     5.78951299e-01
     5.70478976e-01
    -1.53450799e+00
    -6.11316919e-01
    -9.20401096e-01
     5.15417218e-01
     4.90162164e-01
     2.59665459e-01
     2.63120294e-01
     9.25632834e-01
    -1.22487843e+00
     1.24791491e+00
     4.44839478e-01
     1.75949171e-01
     3.65906686e-01
     9.91675615e-01
    -1.30154729e+00
     2.02733922e+00
     3.74916345e-01
     1.90939516e-01
     7.55914092e-01
    -1.67440081e+00
    -1.12454820e+00
     5.29878557e-01
     1.90198869e-01
     1.38852909e-01
     6.64414763e-01
    -1.89679706e+00
     2.18068317e-01
     7.34073445e-02
     6.55422062e-02
     3.81460823e-02
    -1.32223545e-02
    -2.28415057e-01
    -6.28667846e-02
    -4.99223778e-03
     1.45762078e-02
    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None:
            continue
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        Z1 = z1 - zp
        Z2 = z2 - zp
        for k in range(nverts):
            res += _integral_v1(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    return res