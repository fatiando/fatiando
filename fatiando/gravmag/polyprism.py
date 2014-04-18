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

The Total Field magnetic anomaly:

* :func:`~fatiando.gravmag.polyprism.tf`

**Auxiliary Functions**

Calculates the second derivatives of the function

.. math:: 

    \phi(x,y,z) = \int \int \int \frac{1}{r}
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

from fatiando import utils
from fatiando.constants import SI2MGAL, SI2EOTVOS, G, CM, T2NT


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> inc, dec = -40, -13 # Geomagnetic Field direction
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(
    ...             vertices, 100, 700, 
    ...             {'magnetization':utils.ang2vec(10, 70, -5)})]
    >>> # Calculate the total field anomaly
    >>> for t in tf(xp, yp, zp, model, inc, dec): print '%12.5e' % t
     1.08730e+01
     1.99497e+01
     2.35799e+01
     1.28679e+01
     3.96559e+01
    -1.08891e+03
    -2.12606e+03
     3.84393e+01
     3.18080e+01
     3.22914e+02
     1.70247e+02
     2.34122e+01
     8.58015e+00
     1.18120e+01
     8.41901e+00
     5.60295e+00

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
    tf = numpy.zeros(len(xp), dtype='f')
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
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bx = (v1*mx + v2*my + v3*mz)
        by = (v2*mx + v4*my + v5*mz)
        bz = (v3*mx + v5*my + v6*mz)
        tf += intensity*(fx*bx + fy*by + fz*bz)
    tf *= CM*T2NT
    return tf

def gz(xp, yp, zp, prisms):
    r"""
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
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gxx component
    >>> for g in gxx(xp, yp, zp, model): print '%12.5e' % g
     7.20648e-04
    -1.99459e-03
    -1.61846e-03
     6.70913e-04
     4.36627e-03
    -1.13914e-01
    -9.18244e-02
     4.42844e-03
     2.71092e-03
    -4.71978e-03
     3.64223e-03
     2.64087e-03
     2.13059e-04
    -1.03988e-03
    -1.01310e-03
     2.11109e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelxx(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, prisms):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gxy component
    >>> for g in gxy(xp, yp, zp, model): print '%12.5e' % g
     1.49780e-03
     2.65742e-03
    -2.62714e-03
    -1.44730e-03
     1.22663e-03
     1.27725e-02
    -8.25573e-03
    -1.38994e-03
    -2.01592e-03
    -2.46837e-02
     2.36985e-02
     2.06093e-03
    -9.79015e-04
    -1.10491e-03
     1.12202e-03
     9.67007e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelxy(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, prisms):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gxz component
    >>> for g in gxz(xp, yp, zp, model): print '%12.5e' % g
     7.91936e-05
     1.51253e-04
    -1.50435e-04
    -7.42349e-05
     3.29126e-04
     9.37617e-02
    -5.93742e-02
    -3.38122e-04
     2.16512e-04
     4.87912e-03
    -4.96696e-03
    -2.12568e-04
     3.82278e-05
     4.53969e-05
    -4.63732e-05
    -3.75449e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelxz(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, prisms):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gyy component
    >>> for g in gyy(xp, yp, zp, model): print '%12.5e' % g
     3.28264e-04
     5.34592e-03
     4.73375e-03
     3.46348e-04
    -1.89187e-03
    -2.96631e-02
    -5.19217e-02
    -1.92960e-03
    -7.97694e-04
     3.22180e-02
     1.90333e-02
    -7.37962e-04
     4.69074e-04
     2.53840e-03
     2.49829e-03
     4.64505e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelyy(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, prisms):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gyz component
    >>> for g in gyz(xp, yp, zp, model): print '%12.5e' % g
     6.93660e-05
     4.97643e-04
     4.27799e-04
     6.68377e-05
     5.96159e-05
     7.09041e-03
     2.41931e-02
     7.03409e-05
    -9.75448e-05
    -1.13555e-02
    -6.68721e-03
    -1.01390e-04
    -4.36407e-05
    -1.62058e-04
    -1.59297e-04
    -4.29196e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelyz(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, prisms):
    r"""
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

    Example:
    
    >>> from fatiando import mesher, gridder, utils, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = [mesher.PolygonalPrism(vertices, 100, 700, {'density':1})]
    >>> # Calculate the gzz component
    >>> for g in gzz(xp, yp, zp, model): print '%12.5e' % g
    -1.04891e-03
    -3.35133e-03
    -3.11530e-03
    -1.01726e-03
    -2.47441e-03
     1.43577e-01
     1.43746e-01
    -2.49884e-03
    -1.91322e-03
    -2.74982e-02
    -2.26755e-02
    -1.90291e-03
    -6.82133e-04
    -1.49852e-03
    -1.48518e-03
    -6.75614e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density*kernelzz(xp, yp, zp, prism)
    res *= G*SI2EOTVOS
    return res

def _integral_v1(X1, X2, Y1, Y2, Z1, Z2):
    """
    Auxiliary function used in kernelxx
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
    Auxiliary function used in kernelxy
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
    Auxiliary function used in kernelxz
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
    Auxiliary function used in kernelyy
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
    Auxiliary function used in kernelyz
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
    Auxiliary function used in kernelzz
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelxx function
    >>> for k in kernelxx(xp, yp, zp, model): print '%12.5e' % k
     1.07995e-02
    -2.98905e-02
    -2.42538e-02
     1.00541e-02
     6.54319e-02
    -1.70708e+00
    -1.37606e+00
     6.63636e-02
     4.06252e-02
    -7.07295e-02
     5.45816e-02
     3.95754e-02
     3.19286e-03
    -1.55834e-02
    -1.51821e-02
     3.16363e-03

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v1(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelxy function
    >>> for k in kernelxy(xp, yp, zp, model): print '%12.5e' % k
     2.24457e-02
     3.98234e-02
    -3.93697e-02
    -2.16888e-02
     1.83820e-02
     1.91405e-01
    -1.23718e-01
    -2.08293e-02
    -3.02100e-02
    -3.69904e-01
     3.55140e-01
     3.08846e-02
    -1.46713e-02
    -1.65579e-02
     1.68144e-02
     1.44913e-02

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v2(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelxz function
    >>> for k in kernelxz(xp, yp, zp, model): print '%12.5e' % k
     1.18678e-03
     2.26665e-03
    -2.25439e-03
    -1.11247e-03
     4.93221e-03
     1.40509e+00
    -8.89768e-01
    -5.06702e-03
     3.24460e-03
     7.31174e-02
    -7.44337e-02
    -3.18550e-03
     5.72873e-04
     6.80307e-04
    -6.94938e-04
    -5.62639e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v3(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelyy function
    >>> for k in kernelyy(xp, yp, zp, model): print '%12.5e' % k
     4.91928e-03
     8.01127e-02
     7.09389e-02
     5.19029e-03
    -2.83511e-02
    -4.44525e-01
    -7.78086e-01
    -2.89165e-02
    -1.19541e-02
     4.82812e-01
     2.85229e-01
    -1.10589e-02
     7.02943e-03
     3.80398e-02
     3.74387e-02
     6.96097e-03

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v4(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelyz function
    >>> for k in kernelyz(xp, yp, zp, model): print '%12.5e' % k
     1.03950e-03
     7.45756e-03
     6.41089e-03
     1.00161e-03
     8.93389e-04
     1.06255e-01
     3.62553e-01
     1.05411e-03
    -1.46178e-03
    -1.70171e-01
    -1.00213e-01
    -1.51941e-03
    -6.53989e-04
    -2.42856e-03
    -2.38718e-03
    -6.43183e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v5(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
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

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    Example:
    
    >>> from fatiando import mesher, gridder, gravmag
    >>> from fatiando.gravmag import polyprism
    >>> # Construct a regular grid
    >>> area = [-10000, 10000, -10000, 10000]
    >>> shape = (4, 4)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Construct a model
    >>> vertices = [[3713.3892, -4288.7031],
    ...            [4393.3057, -52.301254],
    ...            [1516.7365, 2771.9666],
    ...            [-3817.9917, 1935.1465],
    ...            [-3661.0879, -5230.1255]]
    >>> model = mesher.PolygonalPrism(vertices, 100, 700)
    >>> # Calculate the kernelzz function
    >>> for k in kernelzz(xp, yp, zp, model): print '%12.5e' % k
    -1.57187e-02
    -5.02222e-02
    -4.66851e-02
    -1.52444e-02
    -3.70809e-02
     2.15161e+00
     2.15414e+00
    -3.74471e-02
    -2.86711e-02
    -4.12082e-01
    -3.39810e-01
    -2.85165e-02
    -1.02223e-02
    -2.24564e-02
    -2.22566e-02
    -1.01246e-02

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype='f')
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v6(x[k] - xp, x[(k + 1)%nverts] - xp,
                y[k] - yp, y[(k + 1)%nverts] - yp, Z1, Z2)
    return res