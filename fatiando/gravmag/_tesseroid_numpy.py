"""
Pure Python implementations of functions in fatiando.gravmag.tesseroid.
Used instead of Cython versions if those are not available.
"""
import numpy

from fatiando.constants import MEAN_EARTH_RADIUS


def _scale_nodes(tesseroid, nodes):
    d2r = numpy.pi / 180.
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


def potential(tesseroid, lons, lats, radii, nodes, weights):
    """
    Integrate potential using the Gauss-Legendre Quadrature
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa / numpy.sqrt(l_sqr))
    result *= scale
    return result


def gx(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*rc[k]*kphi / (l_sqr**1.5))
    result *= scale
    return result


def gy(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*rc[k]*coslatc[j]*sinlon / (l_sqr**1.5))
    result *= scale
    return result


def gz(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*(rc[k]*cospsi - radii) / (l_sqr**1.5))
    result *= scale
    return result


def gxx(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*(3.*((rc[k]*kphi)**2) - l_sqr) / (l_sqr**2.5))
    result *= scale
    return result


def gxy(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*3.*(rc[k]**2)*kphi*coslatc[j]*sinlon /
                           (l_sqr**2.5))
    result *= scale
    return result


def gxz(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*3.*rc[k]*kphi*(rc[k]*cospsi - radii) /
                           (l_sqr**2.5))
    result *= scale
    return result


def gyy(tesseroid, lons, lats, radii, nodes, weights):
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
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*(3.*(deltay**2) - l_sqr) / (l_sqr**2.5))
    result *= scale
    return result


def gyz(tesseroid, lons, lats, radii, nodes, weights):
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
        sinlon = numpy.sin(lonc[i] - lons)
        for j in xrange(order):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in xrange(order):
                l_sqr = (radii_sqr + rc[k]**2 -
                         2.*radii*rc[k]*(
                    sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                kappa = (rc[k]**2)*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radii
                result += (weights[i]*weights[j]*weights[k] *
                           kappa*3.*deltay*deltaz / (l_sqr**2.5))
    result *= scale
    return result


def gzz(tesseroid, lons, lats, radii, nodes, weights):
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
                    3.*deltaz**2 - l_sqr) / (l_sqr**2.5)
    result *= scale
    return result
