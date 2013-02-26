"""
Calculates the potential fields of a tesseroid.
"""
import numpy

from fatiando.mesher import Tesseroid
from fatiando.constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G


try:
    from fatiando.gravmag._ctesseroid import *
except ImportError:
    from fatiando.gravmag._tesseroid import *

_glq_nodes = numpy.array([-0.577350269, 0.577350269])
_glq_weights = numpy.array([1., 1.])


def potential(tesseroids, lons, lats, heights, ratio=1.):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_potential, ratio)

def gx(tesseroids, lons, lats, heights, ratio=1.):
    """
    Calculate the x (North) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gx, ratio)

def gy(tesseroids, lons, lats, heights, ratio=1.):
    """
    Calculate the y (East) component of the gravitational attraction due to a
    tesseroid model.
    """
    return SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
        _kernel_gy, ratio)

def gz(tesseroids, lons, lats, heights, ratio=1.):
    """
    Calculate the z (radial) component of the gravitational attraction due to a
    tesseroid model.
    """
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    return -1*SI2MGAL*_optimal_discretize(tesseroids, lons, lats, heights,
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


def gzz(tesseroids, lons, lats, heights, ratio=3):
    """
    Calculate the zz (radial-radial) component of the gravity gradient tensor
    due to a tesseroid model.
    """
    result = SI2EOTVOS*_optimal_discretize(tesseroids, lons, lats, heights, 
        _kernel_gzz, ratio)
    return result

def _kernel_potential(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gx using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa/numpy.sqrt(l_sqr))
    result *= scale
    return result

def _kernel_gx(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gx using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*rc[k]*kphi/(l_sqr**1.5))
    result *= scale
    return result

def _kernel_gy(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gy using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        sinlon = numpy.sin(lonc[i] - lons)
        for j in xrange(order):
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
    result *= scale
    return result

def _kernel_gz(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gz using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*(rc[k]*cospsi - radii)/(l_sqr**1.5))
    result *= scale
    return result

def _kernel_gxx(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gxx using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*(3.*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5))
    result *= scale
    return result

def _kernel_gxy(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gxy using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        sinlon = numpy.sin(lonc[i] - lons)
        for j in xrange(order):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*3.*(rc[k]**2)*kphi*coslatc[j]*sinlon/(l_sqr**2.5))
    result *= scale
    return result

def _kernel_gxz(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gxz using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*3.*rc[k]*kphi*(rc[k]*cospsi - radii)/(l_sqr**2.5))
    result *= scale
    return result

def _kernel_gyy(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gyy using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        sinlon = numpy.sin(lonc[i] - lons)
        for j in xrange(order):
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*(3.*(deltay**2) - l_sqr)/(l_sqr**2.5))
    result *= scale
    return result

def _kernel_gyz(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gyz using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        sinlon = numpy.sin(lonc[i]- lons)
        for j in xrange(order):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radii
                result += (weights[i]*weights[j]*weights[k]*
                    kappa*3.*deltay*deltaz/(l_sqr**2.5))
    result *= scale
    return result

def _kernel_gzz(tesseroid, lons, lats, radii, nodes=_glq_nodes,
    weights=_glq_weights):
    """
    Integrate gzz using the Gauss-Legendre Quadrature
    """
    order = len(nodes)
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    sinlat = numpy.sin(lats)
    coslat = numpy.cos(lats)
    radii_sqr = radii**2
    # Start the numerical integration
    result = numpy.zeros(len(lons), numpy.float)
    for i in xrange(order):
        coslon = numpy.cos(lons - lonc[i])
        for j in xrange(order):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 - 
                         2.*radii*rc[k]*(
                            sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                deltaz = rc[k]*cospsi - radii
                result += weights[i]*weights[j]*weights[k]*kappa*(
                    3.*deltaz**2 - l_sqr)/(l_sqr**2.5)
    result *= scale
    return result

def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio):
    """
    Calculate the effect of a given kernal in the most precise way by adaptively
    discretizing the tesseroids into smaller ones.
    """
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
        size = max([MEAN_EARTH_RADIUS*d2r*(tesseroid.e - tesseroid.w),
                    MEAN_EARTH_RADIUS*d2r*(tesseroid.n - tesseroid.s),
                    tesseroid.top - tesseroid.bottom])
        distance = _distance(tesseroid, rlons, rlats, radii)
        need_divide = _need_to_divide(distance, size, ratio)
        dont_divide = list(allpoints.difference(set(need_divide)))
        if need_divide:
            split = _split(tesseroid)
            result[need_divide] += _optimal_discretize(split, lons[need_divide],
                lats[need_divide], heights[need_divide], kernel, ratio)
        result[dont_divide] += G*tesseroid.props['density']*kernel(
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
