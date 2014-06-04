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

The Total Field anomaly is calculated using the formula of Bhattacharyya (1964)
in function :func:`~fatiando.gravmag.prism.tf`.

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

"""
from __future__ import division

import numpy

from fatiando.constants import G, SI2EOTVOS, CM, T2NT

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


def bx(xp, yp, zp, prisms):
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

    Returns:

    * bx: array
        The x component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    bx = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = numpy.array(prism.props['magnetization'])
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        bx += (v1*mx + v2*my + v3*mz)
    bx *= CM*T2NT
    return bx

def by(xp, yp, zp, prisms):
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

    Returns:

    * by: array
        The y component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    by = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = numpy.array(prism.props['magnetization'])
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        by += (v2*mx + v4*my + v5*mz)
    by *= CM*T2NT
    return by

def bz(xp, yp, zp, prisms):
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

    Returns:

    * bz: array
        The z component of the magnetic induction

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    bz = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = numpy.array(prism.props['magnetization'])
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bz += (v3*mx + v5*my + v6*mz)
    bz *= CM*T2NT
    return bz

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
