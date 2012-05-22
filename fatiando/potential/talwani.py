"""
Calculate the gravitational attraction of a 2D body with polygonal vertical
cross-section using the formula of Talwani et al. (1959)

Use the :func:`~fatiando.mesher.dd.Polygon` object to create polygons.

.. warning:: the vertices must be given clockwise! If not, the result will have
    an inverted sign.

**Components**

* :func:`~fatiando.potential.talwani.gz`

**References**

Talwani, M., J. L. Worzel, and M. Landisman (1959), Rapid Gravity Computations
for Two-Dimensional Bodies with Application to the Mendocino Submarine
Fracture Zone, J. Geophys. Res., 64(1), 49-59, doi:10.1029/JZ064i001p00049.

----

"""

import numpy

from fatiando.potential import _talwani
from fatiando import logger


log = logger.dummy('fatiando.potential.talwani')

def gz(xp, zp, polygons):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, zp : arrays
        The x and z coordinates of the computation points.        
    * polygons : list of :func:`~fatiando.mesher.dd.Polygon`
        The density model used.
        Polygons must have the property ``'density'``. Polygons that don't have
        this property will be ignored in the computations. Elements of
        *polygons* that are None will also be ignored.

        .. note:: The y coordinate of the polygons is used as z! 

    Returns:
    
    * gz : array
        The :math:`g_z` component calculated on the computation points

    """
    if xp.shape != zp.shape:
        raise ValueError("Input arrays xp and zp must have same shape!")
    xp = numpy.array(xp, dtype=numpy.float64)
    zp = numpy.array(zp, dtype=numpy.float64)
    res = numpy.zeros_like(xp)
    for p in polygons:
        if p is not None and 'density' in p:
            res += _talwani.talwani_gz(float(p['density']), p['x'], p['y'],
                xp, zp)
    return res
