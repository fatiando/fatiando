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
Create and operate on 2D meshes and objects like polygons, squares, and
triangles.

**Elements**

* :func:`fatiando.mesher.dd.Polygon`
* :func:`fatiando.mesher.dd.Square`

**Meshes**

* :class:`fatiando.mesher.dd.SquareMesh`

**Utility functions**


----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 12-Jan-2012'

import numpy

from fatiando import logger

log = logger.dummy()


def Polygon(vertices, props):
    """
    Create a polygon object.

    Note: Most applications require the vertices to be in a clockwise!

    Parameters:

    * vertices
        List of (x, y) pairs with the coordinates of the vertices.        
    * props
        Dictionary with the physical properties assigned to the polygon.
        Ex: ``props={'density':10, 'susceptibility':10000}``

    Returns:

    * Polygon object
    
    """    
    x, y = numpy.array(vertices, dtype=numpy.float64).T
    poly = {'x':x, 'y':y}
    for prop in props:
        poly[prop] = props[prop]
    return poly

def Square(bounds, props):
    """
    Create a square object.

    Example::

        >>> sq = Square([0, 1, 2, 4], {'density':750})
        >>> for k in sorted(sq):
        ...     print k, '=', sq[k]
        density = 750
        x1 = 0
        x2 = 1
        y1 = 2
        y2 = 4
        

    Parameters:

    * bounds
        ``[x1, x2, y1, y2]``: coordinates of the top right and bottom left
        corners of the square       
    * props
        Dictionary with the physical properties assigned to the square.
        Ex: ``props={'density':10, 'slowness':10000}``

    Returns:

    * Square object
    
    """
    x1, x2, y1, y2 = bounds
    square = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}
    for prop in props:
        square[prop] = props[prop]
    return square

class SquareMesh(object):
    """
    Create a mesh of squares.

    
    """

    def __init__(self, bounds, shape, props={}):
        pass
    

    def addprop(self):
        pass

    def img2prop(self):
        pass

    
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
