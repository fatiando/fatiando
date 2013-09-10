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
    sinlats = numpy.sin(d2r*lats)
    coslats = numpy.cos(d2r*lats)
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Create some buffers to reduce memory allocation
    buff = numpy.zeros(ndata, numpy.float)
    lonc = numpy.zeros(2, numpy.float)
    sinlatc = numpy.zeros(2, numpy.float)
    coslatc = numpy.zeros(2, numpy.float)
    rc = numpy.zeros(2, numpy.float)
    allpoints = numpy.arange(ndata)
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
    for tesseroid in tesseroids:
        if (tesseroid is None or
            ('density' not in tesseroid.props and dens is None)):
            continue
        if dens is not None:
            density = dens
        else:
            density = tesseroid.props['density']
        lifo = [[allpoints, tesseroid]]
        while lifo:
            points, tess = lifo.pop()
            size = max([MEAN_EARTH_RADIUS*d2r*(tess.e - tess.w),
                        MEAN_EARTH_RADIUS*d2r*(tess.n - tess.s),
                        tess.top - tess.bottom])
            distances = _kernels.distance(tess, rlons, sinlats, coslats,
                    radii, points, buff)
            need_divide, dont_divide = _kernels.too_close(points, distances,
                    ratio*size)
            if len(need_divide):
                lifo.extend([need_divide, t] for t in tess.half())
            if len(dont_divide):
                result[dont_divide] += density*kernel(
                    tess, rlons[dont_divide], sinlats[dont_divide],
                    coslats[dont_divide], radii[dont_divide], lonc, sinlatc,
                    coslatc, rc, buff)
    result *= G
    return result
