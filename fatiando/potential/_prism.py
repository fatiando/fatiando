"""
Pure Python + Numpy implementation of the potential field effects of right
rectangular prisms.
"""
__all__ = ['gx', 'gy', 'gz']

import numpy

from fatiando import logger

log = logger.dummy('fatiando.potential._prism')

SI2EOTVOS = 1000000000.0
SI2MGAL = 100000.0
G = 0.00000000006673


def gx(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_x` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None and 'density' not in prism.props:
            continue
        density = prism.props['density']
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (y[j]*numpy.log(z[k] + r) +
                              z[k]*numpy.log(y[j] + r) -
                              x[i]*numpy.arctan2(z[k]*y[j], x[i]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
    return res

def gy(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_y` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None and 'density' not in prism.props:
            continue
        density = prism.props['density']
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (z[k]*numpy.log(x[i] + r) +
                              x[i]*numpy.log(z[k] + r) -
                              y[j]*numpy.arctan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
    return res

def gz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None and 'density' not in prism.props:
            continue
        density = prism.props['density']
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (x[i]*numpy.log(y[j] + r) +
                              y[j]*numpy.log(x[i] + r) -
                              z[k]*numpy.arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
    return res
