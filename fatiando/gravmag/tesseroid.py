"""
Calculates the potential fields of a tesseroid.
"""

from fatiando.gravmag._tesseroid import *

try:
    from fatiando.gravmag._ctesseroid import *
except ImportError:
    pass


def potential(tesseroids, lons, lats, heights, ratio=1.):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights, 
        _kernel_potential, ratio)

def gx(tesseroids, lons, lats, heights, ratio=2.):
    """
    Calculate the x (North) component of the gravitational attraction due to a 
    tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gx, ratio)

def gy(tesseroids, lons, lats, heights, ratio=2.):
    """
    Calculate the y (East) component of the gravitational attraction due to a 
    tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gy, ratio)

def gz(tesseroids, lons, lats, heights, ratio=2.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a 
    tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gz, ratio)

def gxx(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the xx (North-North) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxx, ratio)

def gxy(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the xy (North-East) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxy, ratio)

def gxz(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the xz (North-radial) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxz, ratio)

def gyy(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the yy (East-East) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gyy, ratio)

def gyz(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the yz (East-radial) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gyz, ratio)

def gzz(tesseroids, lons, lats, heights, ratio=4.):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor 
    due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gzz, ratio)
