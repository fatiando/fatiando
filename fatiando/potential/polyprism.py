"""
Calculate the potential fields and derivatives of the 3D prism with polygonal
crossection using the forumla of Plouff (1976).

**Gravity**

* :func:`~fatiando.potential.polyprism.gz`

**References**

Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
applications to magnetic terrain corrections, Geophysics, 41(4), 727-741.

----

"""

import numpy

from fatiando.potential import _polyprism
from fatiando import logger


log = logger.dummy('fatiando.potential.polyprism')

def gz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list
        List of :func:`~fatiando.mesher.ddd.PolygonalPrism` objects.

    Returns:
    
    * gz : array
        The :math:`g_z` component calculated on the computation points.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for p in prisms:
        if p is not None:
            res += _polyprism.polyprism_gz(p['density'], p['z2'], p['z1'],
                                           p['x'], p['y'], xp, yp, zp)
    return res
