"""
Calculates the potential fields of a tesseroid.
"""
import numpy

from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G

try:
    from fatiando.gravmag import _ctesseroid as _kernels
except ImportError:
    from fatiando.gravmag import _tesseroid as _kernels


_glq_nodes = numpy.array([-0.577350269, 0.577350269])
_glq_weights = numpy.array([1., 1.])


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
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
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
            distances = _distance(tess, rlons, rlats, radii, points_to_calc)
            too_close = (distances > 0) & (distances < ratio*size)
            need_divide = points_to_calc[too_close]
            dont_divide = points_to_calc[~too_close]
            if len(need_divide):
                #if len(lifo) + 8 > maxsize:
                #    log.warning("Maximum LIFO size reached")
                #    dont_divide.extend(need_divide)
                #else:
                #    lifo.extend([need_divide, t] for t in tess.split(2, 2, 2)
                lifo.extend([need_divide, t] for t in tess.split(2, 2, 2))
            if len(dont_divide):
                result[dont_divide] += G*density*kernel(
                    tess, rlons[dont_divide], rlats[dont_divide],
                    radii[dont_divide], _glq_nodes, _glq_weights)
    return result

def _distance(tesseroid, lon, lat, radius, points):
    lons = lon[points]
    lats = lat[points]
    radii = radius[points]
    d2r = numpy.pi/180.
    tes_radius = tesseroid.top + MEAN_EARTH_RADIUS
    tes_lat = d2r*0.5*(tesseroid.s + tesseroid.n)
    tes_lon = d2r*0.5*(tesseroid.w + tesseroid.e)
    distance = numpy.sqrt(
        radii**2 + tes_radius**2 - 2.*radii*tes_radius*(
            numpy.sin(lats)*numpy.sin(tes_lat) +
            numpy.cos(lats)*numpy.cos(tes_lat)*
            numpy.cos(lons - tes_lon)
        ))
    return distance
