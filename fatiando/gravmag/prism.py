"""
Calculate the potential fields of the 3D right rectangular prism.

.. note:: All input units are SI. Output is in conventional units: SI for the
    gravitatonal potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East and z -> Down.

**Gravity**

The gravitational fields are calculated using the forumla of Nagy et al.
(2000). Available functions are:
:func:`~fatiando.gravmag.prism.potential`,
:func:`~fatiando.gravmag.prism.gx`,
:func:`~fatiando.gravmag.prism.gy`,
:func:`~fatiando.gravmag.prism.gz`,
:func:`~fatiando.gravmag.prism.gxx`,
:func:`~fatiando.gravmag.prism.gxy`,
:func:`~fatiando.gravmag.prism.gxz`,
:func:`~fatiando.gravmag.prism.gyy`,
:func:`~fatiando.gravmag.prism.gyz`,
:func:`~fatiando.gravmag.prism.gzz`,

All functions have the following call signature:

*fatiando.gravmag.prism.potential(xp, yp, zp, prisms, dens=None)*

Parameters:

* xp, yp, zp : arrays
    Arrays with the x, y, and z coordinates of the computation points.
* prisms : list of :class:`~fatiando.mesher.Prism`
    The density model used to calculate the gravitational effect.
    Prisms must have the property ``'density'``. Prisms that don't have this
    property will be ignored in the computations. Elements of *prisms* that
    are None will also be ignored. *prisms* can also be a
    :class:`fatiando.mesher.PrismMesh`.
* dens : float or None
    If not None, will use this value instead of the ``'density'`` property
    of the prisms. Use this, e.g., for sensitivity matrix building.

    .. warning:: Uses this value for **all** prisms! Not only the ones that
        have ``'density'`` as a property.

Returns:

* res : array
    The field calculated on xp, yp, zp


**Magnetic**

The Total Field anomaly is calculated using the formula of Bhattacharyya (1964)
in function :func:`~fatiando.gravmag.prism.tf`.

The call signature is:

*fatiando.gravmag.prism.tf(xp, yp, zp, prisms, inc, dec, pmag=None)*

Parameters:

* xp, yp, zp : arrays
    Arrays with the x, y, and z coordinates of the computation points.
* prisms : list of :class:`~fatiando.mesher.Prism`
    The model used to calculate the total field anomaly.
    Prisms without the physical property ``'magnetization'`` will
    be ignored. *prisms* can also be a :class:`fatiando.mesher.PrismMesh`.
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


**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

"""

#from __future__ import division

import numpy

try:
    from fatiando.gravmag._prism import *
except:
    def not_implemented(*args, **kwargs):
        raise NotImplementedError(
        "Couldn't load C coded extension module.")
    potential = not_implemented
    gx = not_implemented
    gy = not_implemented
    gz = not_implemented
    gxx = not_implemented
    gxy = not_implemented
    gxz = not_implemented
    gyy = not_implemented
    gyz = not_implemented
    gzz = not_implemented
    tf = not_implemented
    
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
    res += -numpy.log((Z1 + R111))
    res +=  numpy.log((Z2 + R112))
    res +=  numpy.log((Z1 + R121))
    res += -numpy.log((Z2 + R122))
    res +=  numpy.log((Z1 + R211))
    res += -numpy.log((Z2 + R212))
    res += -numpy.log((Z1 + R221))
    res +=  numpy.log((Z2 + R222))

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
    res += -numpy.log((Y1 + R111))
    res +=  numpy.log((Y1 + R112))
    res +=  numpy.log((Y2 + R121))
    res += -numpy.log((Y2 + R122))
    res +=  numpy.log((Y1 + R211))
    res += -numpy.log((Y1 + R212))
    res += -numpy.log((Y2 + R221))
    res +=  numpy.log((Y2 + R222))

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
    res += -numpy.log((X1 + R111))
    res +=  numpy.log((X1 + R112))
    res +=  numpy.log((X1 + R121))
    res += -numpy.log((X1 + R122))
    res +=  numpy.log((X2 + R211))
    res += -numpy.log((X2 + R212))
    res += -numpy.log((X2 + R221))
    res +=  numpy.log((X2 + R222))

    return -res