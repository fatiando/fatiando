"""
Calculates the potential fields of a tesseroid.
"""
import numpy

from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G

try:
    from fatiando.gravmag import _ctesseroid as _kernels
except ImportError:
    from fatiando.gravmag import _tesseroid as _kernels




def potential(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.potential, ratio, dens)

def gx(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the x (North) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gx, ratio, dens)

def gy(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the y (East) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gy, ratio, dens)

def gz(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a
    tesseroid model.
    """
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    return -1*SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gz, ratio, dens)

def gxx(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the xx (North-North) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gxx, ratio, dens)

def gxy(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the xy (North-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gxy, ratio, dens)

def gxz(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the xz (North-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gxz, ratio, dens)

def gyy(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the yy (East-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gyy, ratio, dens)

def gyz(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the yz (East-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gyz, ratio, dens)


def gzz(lons, lats, heights, tesseroids, dens=None, ratio=3):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    result = SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernels.gzz, ratio, dens)
    return result

def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio, dens):
    """
    Calculate the effect of a given kernal in the most precise way by adaptively
    discretizing the tesseroids into smaller ones.
    """
    ndata = len(lons)
    # Convert things to radians
    d2r = numpy.pi/180.
    rlons = d2r*lons
    rlats = d2r*lats
    sinlats = numpy.sin(rlats)
    coslats = numpy.cos(rlats)
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
    buff = numpy.zeros(ndata, numpy.float)
    #maxsize = 10000
    for tesseroid in tesseroids:
        if (tesseroid is None or
            ('density' not in tesseroid.props and dens is None)):
            continue
        if dens is not None:
            density = dens
        else:
            density = tesseroid.props['density']
        lifo = [[numpy.arange(ndata), tesseroid]]
        while lifo:
            points_to_calc, tess = lifo.pop()
            size = max([MEAN_EARTH_RADIUS*d2r*(tess.e - tess.w),
                        MEAN_EARTH_RADIUS*d2r*(tess.n - tess.s),
                        tess.top - tess.bottom])
            distances = _kernels._distance(tess, rlons, sinlats, coslats,
                    radii, points_to_calc, buff)
            too_close = (distances > 0) & (distances < ratio*size)
            need_divide = points_to_calc[too_close]
            dont_divide = points_to_calc[~too_close]
            if len(need_divide):
                lifo.extend([need_divide, t] for t in tess.half())
            if len(dont_divide):
                result[dont_divide] += G*density*kernel(
                    tess, rlons[dont_divide], rlats[dont_divide],
                    radii[dont_divide])
    return result
