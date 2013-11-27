"""
Cython kernels for the fatiando.gravmag.tesseroid module.

Used to optimize some slow tasks and compute the actual gravitational fields.
"""
import numpy
from libc.math cimport sin, cos, sqrt
from fatiando.constants import MEAN_EARTH_RADIUS
# Import Cython definitions for numpy
cimport numpy
cimport cython

__all__ = ['_potential', '_gx', '_gy', '_gz', '_gxx', '_gxy', '_gxz',
    '_gyy', '_gyz', '_gzz', '_distance', '_too_close']


cdef:
    double d2r = numpy.pi/180.
    double[::1] nodes
nodes = numpy.array([-0.577350269, 0.577350269])

@cython.boundscheck(False)
@cython.wraparound(False)
def _too_close(numpy.ndarray[long, ndim=1] points,
        numpy.ndarray[double, ndim=1] distance, double value):
    """
    Separate 'points' into two lists, ones that are too close and ones that
    aren't. How close is allowed depends on 'value'. 'points' is a list of the
    indices corresponding to observation points.
    """
    cdef:
        int i, j, l, size = len(points)
        numpy.ndarray[long, ndim=1] buff
    buff = numpy.empty(size, dtype=numpy.int)
    i = 0
    j = size - 1
    for l in range(size):
        if distance[l] > 0 and distance[l] < value:
            buff[i] = points[l]
            i += 1
        else:
            buff[j] = points[l]
            j -= 1
    return buff[:i], buff[j + 1:size]

@cython.boundscheck(False)
@cython.wraparound(False)
def _distance(tesseroid,
    numpy.ndarray[double, ndim=1] lon,
    numpy.ndarray[double, ndim=1] sinlat,
    numpy.ndarray[double, ndim=1] coslat,
    numpy.ndarray[double, ndim=1] radius,
    numpy.ndarray[long, ndim=1] points,
    numpy.ndarray[double, ndim=1] buff):
    """
    Calculate the distance between a tesseroid and some observation points.
    Which points to calculate are specified by the indices in 'points'. Returns
    the values in 'buff'.
    """
    cdef:
        unsigned int i, l, size = len(points)
        double tes_radius, tes_lat, tes_lon
        double w, e, s, n, top, bottom
    w, e, s, n, top, bottom = tesseroid
    tes_radius = top + MEAN_EARTH_RADIUS
    tes_lat = d2r*0.5*(s + n)
    tes_lon = d2r*0.5*(w + e)
    for l in range(size):
        i = points[l]
        buff[l] = sqrt(radius[i]**2 + tes_radius**2 -
            2.*radius[i]*tes_radius*(sinlat[i]*sin(tes_lat) +
                coslat[i]*cos(tes_lat)*cos(lon[i] - tes_lon)))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _scale_nodes(tesseroid,
        numpy.ndarray[double, ndim=1] lonc,
        numpy.ndarray[double, ndim=1] sinlatc,
        numpy.ndarray[double, ndim=1] coslatc,
        numpy.ndarray[double, ndim=1] rc):
    "Put GLQ nodes in the integration limits for a tesseroid"
    cdef:
        double dlon, dlat, dr, mlon, mlat, mr, scale, latc
        unsigned int i
        double w, e, s, n, top, bottom
    w, e, s, n, top, bottom = tesseroid
    dlon = e - w
    dlat = n - s
    dr = top - bottom
    mlon = 0.5*(e + w)
    mlat = 0.5*(n + s)
    mr = 0.5*(top + bottom + 2.*MEAN_EARTH_RADIUS)
    # Scale the GLQ nodes to the integration limits
    for i in range(2):
        lonc[i] = d2r*(0.5*dlon*nodes[i] + mlon)
        latc = d2r*(0.5*dlat*nodes[i] + mlat)
        sinlatc[i] = sin(latc)
        coslatc[i] = cos(latc)
        rc[i] = (0.5*dr*nodes[i] + mr)
    scale = d2r*dlon*d2r*dlat*dr*0.125
    return scale

@cython.boundscheck(False)
@cython.wraparound(False)
def _potential(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(kappa/sqrt(l_sqr))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gx(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*rc[k]*kphi/(l_sqr**1.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gy(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in range(2):
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gz(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*(rc[k]*cospsi - radii[l])/(l_sqr**1.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gxx(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*(3.*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gxy(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in range(2):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*3.*(rc[k]**2)*kphi*coslatc[j]*sinlon/(
                            l_sqr**2.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gxz(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    result[l] += density*scale*(
                        kappa*3.*rc[k]*kphi*(rc[k]*cospsi - radii[l])/
                        (l_sqr**2.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gyy(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in range(2):
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltay = rc[k]*coslatc[j]*sinlon
                    result[l] += density*scale*(
                        kappa*(3.*(deltay**2) - l_sqr)/(l_sqr**2.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gyz(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            sinlon = sin(lonc[i] - lons[l])
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltay = rc[k]*coslatc[j]*sinlon
                    deltaz = rc[k]*cospsi - radii[l]
                    result[l] += density*scale*(
                        kappa*3.*deltay*deltaz/(l_sqr**2.5))

@cython.boundscheck(False)
@cython.wraparound(False)
def _gzz(tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] lonc,
    numpy.ndarray[double, ndim=1] sinlatc,
    numpy.ndarray[double, ndim=1] coslatc,
    numpy.ndarray[double, ndim=1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int i, j, k, l, p
        double scale, kappa, sinlat, coslat, radii_sqr, coslon, l_sqr
        double cospsi, deltaz
    # Put the nodes in the current range
    scale = _scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radii_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    l_sqr = (radii_sqr + rc[k]**2 -
                             2.*radii[l]*rc[k]*(
                                sinlat*sinlatc[j] + coslat*coslatc[j]*coslon))
                    kappa = (rc[k]**2)*coslatc[j]
                    deltaz = rc[k]*cospsi - radii[l]
                    result[l] += density*scale*(
                        kappa*(3.*deltaz**2 - l_sqr)/(l_sqr**2.5))
