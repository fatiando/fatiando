"""
Points:
    Classes for different kinds of points (Cartesian, Spherical, Ellipsoidal,
    2D, and 3D)
"""


class Cart2DPoint():
    """
    Cartesian point with (x,y) coordinates.
    Access the coordinates using the x and y properties.
    
    Examples:
    
        >>> p = Cart2DPoint(1, 2)
        >>> print p.x
        1
        >> print p.y
        2    
    """
    
    def __init__(self, x, y):        
        self._x = x
        self._y = y
        
    def _getx(self):
        
        return self._x

    x = property(_getx)
    
    def _gety(self):
        
        return self._y
    
    y = property(_gety)