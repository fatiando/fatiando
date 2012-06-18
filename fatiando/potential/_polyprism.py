"""
.. topic:: Python + Numpy implementation.

    Module :mod:`fatiando.potential.polyprism` loads all functions from
    ``fatiando.potential._polyprism``, which contain the Python + Numpy
    implementation. The slightly faster Cython module
    ``fatiando.potentia._cpolyprism`` will be out soon. If it is available,
    then will substitude the Python + Numpy functions with its functions. All
    input and output are the same but there is a slight speed increase.
    
"""
import numpy
from numpy import arctan2, log, sqrt, arctan

from fatiando import logger
from fatiando.constants import SI2MGAL, SI2EOTVOS, G


def gz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : listSI2MGAL
        List of :class:`fatiando.mesher.ddd.PolygonalPrism` objects.

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
    Calculates the :math:`g_xx` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list
        List of :class:`fatiando.mesher.ddd.PolygonalPrism` objects.

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
        for k in range(nverts):
            res += density*_integral_v1(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    res *= G*SI2EOTVOS
    return res
                    
def _integral_v1(x1, x2, y1, y2, z1, z2):
    """
    Calculates the first element of the V matrix (gxx components)
    """    
    dummy = 10.**(-10) # Used to avoid singularities
    dx = x2 - x1
    dy = y2 - y1
    dr = sqrt(dx**2 + dy**2)
    z1_sqr = z1**2
    z2_sqr = z2**2
    n = dx/(dy + dummy)
    g = x1 - y1*n
    p = (x1*y2 - x2*y1)/(dr + dummy)    
    
    r2_sqr = x2**2 + y2**2
    R21 = sqrt(r2_sqr + z1_sqr)
    R22 = sqrt(r2_sqr + z2_sqr)
    d2 = (dx*x2 + dy*y2)/(dr + dummy)
    aux = arctan2(z2*d2, p*R22) - arctan2(z1*d2, p*R21)
    res = g*y2*aux/(p*d2 + dummy) + n*p*aux/(d2 + dummy)
    
    r1_sqr = x1**2 + y1**2
    R11 = sqrt(r1_sqr + z1_sqr)
    R12 = sqrt(r1_sqr + z2_sqr)
    d1 = dx*x1 + dy*y1/(dr + dummy)
    aux = arctan2(z2*d1, p*R12) - arctan2(z1*d1, p*R11)
    res -= g*y1*aux/(p*d1 + dummy) + n*p*aux/(d1 + dummy)
    
    res += n*(log(z2 + R12 + dummy) - log(z1 + R11 + dummy)
           - log(z2 + R22 + dummy) + log(z1 + R21 + dummy))
    res *= -(1.0/(1.0 + n*n + dummy))
    return res
