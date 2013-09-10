# cython: profile=True
"""
Pure Python implementations of functions in fatiando.gravmag.tesseroid.
Used instead of Cython versions if those are not available.
"""
import numpy
from libc.math cimport sin, cos, sqrt
from fatiando.constants import MEAN_EARTH_RADIUS
# Import Cython definitions for numpy
cimport numpy
cimport cython

cdef:
    double d2r = numpy.pi/180.
    double[::1] nodes = numpy.array([-0.577350269, 0.577350269])
    double[::1] weights = numpy.array([1., 1.])
    unsigned int order = len(nodes)

@cython.boundscheck(False)
@cython.wraparound(False)
def _distance(tesseroid,
    numpy.ndarray[double, ndim=1] lon,
    numpy.ndarray[double, ndim=1] sinlat,
    numpy.ndarray[double, ndim=1] coslat,
    numpy.ndarray[double, ndim=1] radius,
    numpy.ndarray[numpy.int_t, ndim=1] points,
    numpy.ndarray[double, ndim=1] distance):
    cdef:
        unsigned int i, l, size = len(points)
        double tes_radius, tes_lat, tes_lon
    tes_radius = tesseroid.top + MEAN_EARTH_RADIUS
    tes_lat = d2r*0.5*(tesseroid.s + tesseroid.n)
    tes_lon = d2r*0.5*(tesseroid.w + tesseroid.e)
    for l in range(size):
        i = points[l]
        distance[l] = sqrt(radius[i]**2 + tes_radius**2 -
            2.*radius[i]*tes_radius*(sinlat[i]*sin(tes_lat) +
                coslat[i]*cos(tes_lat)*cos(lon[i] - tes_lon)))
    return distance[:size]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _scale_nodes(tesseroid, double[::1] lonc,
        double[::1] sinlatc, double[::1] coslatc, double[::1] rc):
    cdef:
        double dlon, dlat, dr, mlon, mlat, mr, scale, latc
        unsigned int i
    dlon = tesseroid.e - tesseroid.w
    dlat = tesseroid.n - tesseroid.s
    dr = tesseroid.top - tesseroid.bottom
    mlon = 0.5*(tesseroid.e + tesseroid.w)
    mlat = 0.5*(tesseroid.n + tesseroid.s)
    mr = 0.5*(tesseroid.top + tesseroid.bottom + 2.*MEAN_EARTH_RADIUS)
    # Scale the GLQ nodes to the integration limits
    for i in range(order):
        lonc[i] = d2r*(0.5*dlon*nodes[i] + mlon)
        latc = d2r*(0.5*dlat*nodes[i] + mlat)
        sinlatc[i] = sin(latc)
        coslatc[i] = cos(latc)
        rc[i] = (0.5*dr*nodes[i] + mr)
    scale = d2r*dlon*d2r*dlat*dr*0.125
    return scale

#def potential(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate potential using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #for j in xrange(order):
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa/sqrt(l_sqr))
        #result[l] = result[l]*scale
    #return result

#def gx(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gx using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double kphi
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #for j in xrange(order):
                #kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*rc[k]*kphi/(l_sqr**1.5))
        #result[l] = result[l]*scale
    #return result

#def gy(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gy using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double sinlon
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #sinlon = sin(lonc[i] - lons[l])
            #for j in xrange(order):
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
        #result[l] = result[l]*scale
    #return result

#def gz(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gz using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double cospsi
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #for j in xrange(order):
                #cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*(rc[k]*cospsi - radii[l])/(l_sqr**1.5))
        #result[l] = result[l]*scale
    #return result

#def gxx(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gxx using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double kphi
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #for j in xrange(order):
                #kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*(3.*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5))
        #result[l] = result[l]*scale
    #return result

#def gxy(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gxy using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double kphi, sinlon
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #sinlon = sin(lonc[i] - lons[l])
            #for j in xrange(order):
                #kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*3.*(rc[k]**2)*kphi*coslatc[j]*sinlon/(l_sqr**2.5))
        #result[l] = result[l]*scale
    #return result

#def gxz(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gxz using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double kphi, cospsi
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #for j in xrange(order):
                #kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                #cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*3.*rc[k]*kphi*(rc[k]*cospsi - radii[l])/
                        #(l_sqr**2.5))
        #result[l] = result[l]*scale
    #return result

#def gyy(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gyy using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double sinlon, deltay
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #sinlon = sin(lonc[i] - lons[l])
            #for j in xrange(order):
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #deltay = rc[k]*coslatc[j]*sinlon
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*(3.*(deltay**2) - l_sqr)/(l_sqr**2.5))
        #result[l] = result[l]*scale
    #return result

#def gyz(tesseroid,
    #numpy.ndarray[double, ndim=1] lons,
    #numpy.ndarray[double, ndim=1] lats,
    #numpy.ndarray[double, ndim=1] radii,
    #numpy.ndarray[double, ndim=1] nodes,
    #numpy.ndarray[double, ndim=1] weights):
    #"""
    #Integrate gyz using the Gauss-Legendre Quadrature
    #"""
    #cdef unsigned int order = len(nodes), ndata = len(lons), i, j, k, l
    #cdef numpy.ndarray[double, ndim=1] lonc, latc, rc, sinlatc, coslatc
    #cdef numpy.ndarray[double, ndim=1] result
    #cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    #cdef double sinlon, deltay, deltaz, cospsi
    ## Put the nodes in the corrent range
    #lonc, latc, rc, scale = _scale_nodes(tesseroid, nodes)
    #result = numpy.zeros(ndata, numpy.float)
    ## Pre-compute sines, cossines and powers
    #sinlatc = numpy.sin(latc)
    #coslatc = numpy.cos(latc)
    ## Start the numerical integration
    #for l in xrange(ndata):
        #sinlat = sin(lats[l])
        #coslat = cos(lats[l])
        #radii_sqr = radii[l]**2
        #for i in xrange(order):
            #coslon = cos(lons[l] - lonc[i])
            #sinlon = sin(lonc[i] - lons[l])
            #for j in xrange(order):
                #cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                #for k in xrange(order):
                    #l_sqr = (radii_sqr + rc[k]**2 -
                             #2.*radii[l]*rc[k]*(
                                #sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    #kappa = (rc[k]**2)*coslatc[j]
                    #deltay = rc[k]*coslatc[j]*sinlon
                    #deltaz = rc[k]*cospsi - radii[l]
                    #result[l] = result[l] + (weights[i]*weights[j]*weights[k]*
                        #kappa*3.*deltay*deltaz/(l_sqr**2.5))
        #result[l] = result[l]*scale
    #return result

def gzz(tesseroid,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] lats,
    numpy.ndarray[double, ndim=1] radii):
    """
    Integrate gzz using the Gauss-Legendre Quadrature
    """
    cdef unsigned ndata = len(lons), i, j, k, l
    cdef numpy.ndarray[double, ndim=1] lonc, rc, sinlatc, coslatc
    cdef numpy.ndarray[double, ndim=1] result
    cdef double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
    cdef double cospsi, deltaz
    # Put the nodes in the corrent range
    lonc = numpy.zeros(order, numpy.float)
    sinlatc = numpy.zeros(order, numpy.float)
    coslatc = numpy.zeros(order, numpy.float)
    rc = numpy.zeros(order, numpy.float)
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    result = numpy.zeros(ndata, numpy.float)
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
