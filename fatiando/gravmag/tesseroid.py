"""
Calculates the potential fields of a tesseroid.
"""
import numpy

from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G
try:
    from fatiando.gravmag._tesseroid import *
except:
    def not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "Couldn't load C coded extension module.")
    _potential = not_implemented
    _gx = not_implemented
    _gy = not_implemented
    _gz = not_implemented
    _gxx = not_implemented
    _gxy = not_implemented
    _gxz = not_implemented
    _gyy = not_implemented
    _gyz = not_implemented
    _gzz = not_implemented
    _distance = not_implemented
    _too_close = not_implemented


def potential(lons, lats, heights, tesseroids, dens=None, ratio=0.5):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
                               _potential, ratio, dens)


def gx(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the x (North) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL * _optimal_discretize(tesseroids, lons, lats, heights,
                                         _gx, ratio, dens)


def gy(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the y (East) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL * _optimal_discretize(tesseroids, lons, lats, heights,
                                         _gy, ratio, dens)


def gz(lons, lats, heights, tesseroids, dens=None, ratio=1.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a
    tesseroid model.
    """
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    return -1 * SI2MGAL * _optimal_discretize(tesseroids, lons, lats, heights,
                                              _gz, ratio, dens)


def gxx(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xx (North-North) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxx, ratio, dens)


def gxy(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xy (North-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxy, ratio, dens)


def gxz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the xz (North-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                           _gxz, ratio, dens)


def gyy(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the yy (East-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                           _gyy, ratio, dens)


def gyz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the yz (East-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                           _gyz, ratio, dens)


def gzz(lons, lats, heights, tesseroids, dens=None, ratio=2.5):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    result = SI2EOTVOS * _optimal_discretize(tesseroids, lons, lats, heights,
                                             _gzz, ratio, dens)
    return result


def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio, dens):
    """
    Calculate the effect of a given kernel in the most precise way by
    adaptively discretizing the tesseroids into smaller ones.
    """
    ndata = len(lons)
    # Convert things to radians
    d2r = numpy.pi / 180.
    rlons = d2r * lons
    sinlats = numpy.sin(d2r * lats)
    coslats = numpy.cos(d2r * lats)
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Create some buffers to reduce memory allocation
    distances = numpy.zeros(ndata, numpy.float)
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
        queue = [(allpoints, tesseroid.get_bounds())]
        while queue:
            points, tess = queue.pop()
            w, e, s, n, top, bottom = tess
            size = max([MEAN_EARTH_RADIUS * d2r * (e - w),
                        MEAN_EARTH_RADIUS * d2r * (n - s),
                        top - bottom])
            _distance(tess, rlons, sinlats, coslats, radii, points, distances)
            need_divide, dont_divide = _too_close(points, distances,
                                                  ratio * size)
            if len(need_divide):
                if len(queue) >= 1000:
                    raise ValueError('Tesseroid queue overflow')
                queue.extend(_half(tess, need_divide))
            if len(dont_divide):
                kernel(tess, density, rlons, sinlats, coslats, radii, lonc,
                       sinlatc, coslatc, rc, result, dont_divide)
    result *= G
    return result


def _half(bounds, points):
    w, e, s, n, top, bottom = bounds
    dlon = 0.5 * (e - w)
    dlat = 0.5 * (n - s)
    dh = 0.5 * (top - bottom)
    yield (points, (w, w + dlon, s, s + dlat, bottom + dh, bottom))
    yield (points, (w, w + dlon, s, s + dlat, top, bottom + dh))
    yield (points, (w, w + dlon, s + dlat, n, bottom + dh, bottom))
    yield (points, (w, w + dlon, s + dlat, n, top, bottom + dh))
    yield (points, (w + dlon, e, s, s + dlat, bottom + dh, bottom))
    yield (points, (w + dlon, e, s, s + dlat, top, bottom + dh))
    yield (points, (w + dlon, e, s + dlat, n, bottom + dh, bottom))
    yield (points, (w + dlon, e, s + dlat, n, top, bottom + dh))
