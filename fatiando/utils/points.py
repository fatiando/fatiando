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
    
    

class Cart3DPoint():
    """
    Cartesian point with (x,y,z) coordinates.
    Access the coordinates using the x, y, and z properties.
    
    Examples:
    
        >>> p = Cart3DPoint(1, 2, 3)
        >>> print p.x
        1
        >> print p.y
        2    
        >> print p.z
        3    
    """
    
    def __init__(self, x, y, z):        
        self._x = x
        self._y = y
        self._z = z
        
    def _getx(self):
        
        return self._x

    x = property(_getx)
    
    def _gety(self):
        
        return self._y
    
    y = property(_gety)
    
    def _getz(self):
        
        return self._z
    
    z = property(_getz)