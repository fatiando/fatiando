"""
Pure Python implementations of functions in fatiando.gravmag.tesseroid.
Used instead of Cython versions if those are not available.
"""
import numpy
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from libc.math cimport sin, cos, sqrt

from fatiando.constants import MEAN_EARTH_RADIUS


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

def potential(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate potential using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa/sqrt(l_sqr))
        result[l] = result[l]*scale
    return result

def gx(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gx using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T kphi
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*rc[k]*kphi/(l_sqr**1.5))
        result[l] = result[l]*scale
    return result

def gy(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gy using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T sinlon
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in xrange(order):
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
        result[l] = result[l]*scale
    return result

def gz(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gz using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T cospsi
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*(rc[k]*cospsi - radii[l])/(l_sqr**1.5))
        result[l] = result[l]*scale
    return result

def gxx(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gxx using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T kphi
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*(3.*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5))
        result[l] = result[l]*scale
    return result

def gxy(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gxy using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T kphi, sinlon
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in xrange(order):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*3.*(rc[k]**2)*kphi*coslatc[j]*sinlon/(l_sqr**2.5))
        result[l] = result[l]*scale
    return result

def gxz(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gxz using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T kphi, cospsi
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*3.*rc[k]*kphi*(rc[k]*cospsi - radii[l])/
                        (l_sqr**2.5))
        result[l] = result[l]*scale
    return result

def gyy(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gyy using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T sinlon, deltay
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in xrange(order):
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltay = rc[k]*coslatc[j]*sinlon
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*(3.*(deltay**2) - l_sqr)/(l_sqr**2.5))
        result[l] = result[l]*scale
    return result

def gyz(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gyz using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T sinlon, deltay, deltaz, cospsi
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in xrange(order):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltay = rc[k]*coslatc[j]*sinlon
                    deltaz = rc[k]*cospsi - radii[l]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*3.*deltay*deltaz/(l_sqr**2.5))
        result[l] = result[l]*scale
    return result

def gzz(tesseroid,
    numpy.ndarray[DTYPE_T, ndim=1] lons,
    numpy.ndarray[DTYPE_T, ndim=1] lats,
    numpy.ndarray[DTYPE_T, ndim=1] radii,
    numpy.ndarray[DTYPE_T, ndim=1] nodes,
    numpy.ndarray[DTYPE_T, ndim=1] weights):
    """
    Integrate gzz using the Gauss-Legendre Quadrature
    """
    cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[DTYPE_T, ndim=1] lonc, latc, rc, sinlatc, coslatc
    cdef numpy.ndarray[DTYPE_T, ndim=1] result
    cdef DTYPE_T scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef DTYPE_T cospsi, deltaz
    # Put the nodes in the corrent range
    lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    result = numpy.zeros(ndata, DTYPE)
    # Pre-compute sines, cossines and powers
    sinlatc = numpy.sin(latc)
    coslatc = numpy.cos(latc)
    # Start the numerical integration
    for l in xrange(ndata):
        sinlat = sin(lats[l])
        coslat = cos(lats[l])
        radii_sqr = radii[l]**2
        for i in xrange(order):
            coslon = cos(lons[l] - lonc[i])
            for j in xrange(order):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in xrange(order):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltaz = rc[k]*cospsi - radii[l]
                    result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        kappa*(3.*deltaz**2 - l_sqr)/(l_sqr**2.5))
        result[l] = result[l]*scale
    return result
