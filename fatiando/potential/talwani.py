# Copyright 2012 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Calculate the gravitational attraction of a 2D body with polygonal vertical
cross-section using the formula of Talwani et al. (1959)

Use the :func:`fatiando.mesher.dd.Polygon` object.

**Remember**, the vertices must be given clockwise!
If not, the result will have an inverted sign. 

**Components**

* :func:`fatiando.potential.talwani.gz`

**References**

Talwani, M., J. L. Worzel, and M. Landisman (1959), Rapid Gravity Computations
    for Two-Dimensional Bodies with Application to the Mendocino Submarine
    Fracture Zone, J. Geophys. Res., 64(1), 49-59, doi:10.1029/JZ064i001p00049.

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 12-Jan-2012'

import numpy

from fatiando.potential import _talwani
from fatiando import logger


log = logger.dummy('fatiando.potential.talwani')

def gz(xp, zp, polygons):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    The coordinate system of the input parameters is z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, zp
        Arrays with x and z coordinates of the computation points.        
    * polygons
        List of :func:`fatiando.mesher.dd.Polygon` objects.
        The y coordinate of the polygons is used as z! 

    Returns:
    
    * List with the :math:`g_z` component calculated on the computation points

    """
    if xp.shape != zp.shape:
        raise ValueError, "Input arrays xp and zp must have same shape!"
    xp = numpy.array(xp, dtype=numpy.float64)
    zp = numpy.array(zp, dtype=numpy.float64)
    res = numpy.zeros_like(xp)
    for p in polygons:
        if p is not None:
            res += _talwani.talwani_gz(float(p['density']), p['x'], p['y'],
                xp, zp)
    return res
