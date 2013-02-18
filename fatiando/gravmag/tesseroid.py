"""
Calculates the potential fields of a tesseroid.
"""
import numpy

from fatiando.mesher import Tesseroid
from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G


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
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gx, ratio)

def gy(tesseroids, lons, lats, heights, ratio=2.):
    """
    Calculate the y (East) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gy, ratio)

def gz(tesseroids, lons, lats, heights, ratio=2.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gz, ratio)

def gxx(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the xx (North-North) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxx, ratio)

def gxy(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the xy (North-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxy, ratio)

def gxz(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the xz (North-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gxz, ratio)

def gyy(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the yy (East-East) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gyy, ratio)

def gyz(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the yz (East-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    return SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gyz, ratio)

glq_nodes = numpy.array([-0.577350269, 0.577350269])
glq_weights = numpy.array([1., 1.])

def gzz(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    kernel = _kernel_gzz
    ndata = len(lons)
    allpoints = set(range(ndata))
    # Convert things to radians
    d2r = numpy.pi/180.
    rlons = d2r*lons
    rlats = d2r*lats
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Start the computations
    result = numpy.zeros(ndata, numpy.float)
    for tesseroid in tesseroids:
        if tesseroid is None or 'density' not in tesseroid.props:
            continue
        dimension = max([MEAN_EARTH_RADIUS*d2r*(tesseroid.e - tesseroid.w),
                         MEAN_EARTH_RADIUS*d2r*(tesseroid.n - tesseroid.s),
                         tesseroid.top - tesseroid.bottom])
        distance = _distance(tesseroid, rlons, rlats, radii)
        need_divide = [i for i in xrange(ndata)
                       if distance[i] > 0 and distance[i] < ratio*dimension]
        dont_divide = list(allpoints.difference(set(need_divide)))
        if need_divide:
            split = _split(tesseroid)
            result[need_divide] += gzz(split, lons[need_divide],
                lats[need_divide], heights[need_divide], ratio)
        result[dont_divide] += G*SI2EOTVOS*tesseroid.props['density']*kernel(
            tesseroid, rlons[dont_divide], rlats[dont_divide], 
            radii[dont_divide])
    return result

def _scale_nodes(tesseroid, nodes):
    d2r = numpy.pi/180.
    dlon = tesseroid.e - tesseroid.w
    dlat = tesseroid.n - tesseroid.s
    dr = tesseroid.top - tesseroid.bottom
    # Scale the GLQ nodes to the integration limits
    nodes_lon = d2r*(0.5*dlon*nodes + 0.5*(tesseroid.e + tesseroid.w))
    nodes_lat = d2r*(0.5*dlat*nodes + 0.5*(tesseroid.n + tesseroid.s))
    nodes_r = (0.5*dr*nodes +
        0.5*(tesseroid.top + tesseroid.bottom + 2.*MEAN_EARTH_RADIUS))
    scale = d2r*dlon*d2r*dlat*dr*0.125
    return nodes_lon, nodes_lat, nodes_r, scale

def _kernel_gzz(tesseroid, lons, lats, radii, nodes=glq_nodes,
    weights=glq_weights):
    """
    Integrate gzz using the Gauss-Legendre Quadrature
    """
    nodes_lon, nodes_lat, nodes_r, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sin_nodes_lat = numpy.sin(nodes_lat)
    cos_nodes_lat = numpy.cos(nodes_lat)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for lonc, wlon in zip(nodes_lon, weights):
        coslon = numpy.cos(lons - lonc)
        sinlon = numpy.sin(lonc - lons)
        for sinlatc, coslatc, wlat in zip(sin_nodes_lat, cos_nodes_lat, weights):
            cospsi = sinlat*sinlatc + coslat*coslatc*coslon
            for rc, wr in zip(nodes_r, weights):
                l_sqr = (radii_sqr + rc**2 - 
                         2.*radii*rc*(sinlat*sinlatc + coslat*coslatc*coslon))
                kappa = (rc**2)*coslatc
                deltaz = rc*cospsi - radii
                result += wlon*wlat*wr*kappa*(3.*deltaz**2 - l_sqr)/(l_sqr**2.5)
    result *= scale
    return result

def _split(tesseroid):
    dlon = 0.5*(tesseroid.e - tesseroid.w)
    dlat = 0.5*(tesseroid.n - tesseroid.s)
    dh = 0.5*(tesseroid.top - tesseroid.bottom)
    wests = [tesseroid.w, tesseroid.w + dlon]
    souths = [tesseroid.s, tesseroid.s + dlat]
    bottoms = [tesseroid.bottom, tesseroid.bottom + dh]
    split = [
        Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=tesseroid.props)
        for i in wests for j in souths for k in bottoms]
    return split

def _distance(tesseroid, lon, lat, radius):
    d2r = numpy.pi/180.
    tes_radius = tesseroid.top + MEAN_EARTH_RADIUS
    tes_lat = d2r*0.5*(tesseroid.s + tesseroid.n)
    tes_lon = d2r*0.5*(tesseroid.w + tesseroid.e)
    distance = numpy.sqrt(
        radius**2 + tes_radius**2 - 2.*radius*tes_radius*(
            numpy.sin(lat)*numpy.sin(tes_lat) +
            numpy.cos(lat)*numpy.cos(tes_lat)*numpy.cos(lon - tes_lon)
        ))
    return distance
