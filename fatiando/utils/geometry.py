"""
Geometry:
    Basic geometric elements and discretizations.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'


class Prism():
    """
    Prism element with associated physical properties (density, magnetization, 
    etc.)
    """
    
    def __init__(self, x1, x2, y1, y2, z1, z2, dens=0):
       """
       Parameters:
       
           x1, x2, y1, y2, z1, z2: x, y, and z dimensions of the prism
           
           dens: density of the prism
       """ 
       
       self.x1 = x1
       self.x2 = x2
       self.y1 = y1
       self.y2 = y2
       self.z1 = z1
       self.z2 = z2
       self.dens = dens
       
       
class Sphere():
    """
    Sphere element with associated physical properties (density, magnetization, 
    etc.)
    """
    
    def __init__(self, x, y, z, radius, dens=0):
       """
       Parameters:
       
           x, y, z: coordinates of the center of the sphere
           
           radius: radius of the sphere
           
           dens: density of the prism
       """ 
       
       self.x = x
       self.y = y
       self.z = z
       self.radius = radius
       self.dens = dens