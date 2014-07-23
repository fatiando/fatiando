"""
Cython kernels for the fatiando.gravmag.tesseroid module.

Used to optimize some slow tasks and compute the actual gravitational fields.
"""
import numpy

from ..constants import MEAN_EARTH_RADIUS, G

from libc.math cimport sin, cos, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython


cdef:
    double d2r = numpy.pi/180.
    double[::1] nodes
nodes = numpy.array([-0.577350269, 0.577350269])

@cython.boundscheck(False)
@cython.wraparound(False)
def too_close(numpy.ndarray[long, ndim=1] points,
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
def distance(
    tesseroid,
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
        double rt, latt, lont, sinlatt, coslatt
        double w, e, s, n, top, bottom
    w, e, s, n, top, bottom = tesseroid
    rt = top + MEAN_EARTH_RADIUS
    latt = d2r*0.5*(s + n)
    sinlatt = sin(latt)
    coslatt = cos(latt)
    lont = d2r*0.5*(w + e)
    for l in range(size):
        i = points[l]
        buff[l] = sqrt(
            radius[i]**2 + rt**2 - 2*radius[i]*rt*(
                sinlat[i]*sinlatt + coslat[i]*coslatt*cos(lon[i] - lont)))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double scale_nodes(
    tesseroid,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc):
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
def potential(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
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
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
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
def gx(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
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
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
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
def gy(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
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
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
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
def gz(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
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
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
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
def gxx(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelxx(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelxx(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi, kphi
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(3*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
def gxy(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelxy(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelxy(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, kphi
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        sinlon = sin(lonc[i] - lon)
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc_sqr*kphi*coslatc[j]*sinlon/(l_sqr**2.5)
    return result*scale

@cython.boundscheck(False)
@cython.wraparound(False)
def gxz(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelxz(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelxz(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, l_5, cospsi, kphi
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_5 = (r_sqr + rc_sqr - 2*radius*rc[k]*cospsi)**2.5
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc[k]*kphi*(rc[k]*cospsi - radius)/l_5
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
def gyy(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelyy(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelyy(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        sinlon = sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                result += kappa*(3*(deltay**2) - l_sqr)/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
def gyz(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelyz(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelyz(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double deltaz, result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        sinlon = sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radius
                result += kappa*3.*deltay*deltaz/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
def gzz(
    tesseroid,
    double density,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result,
    numpy.ndarray[long, ndim=1] points):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l, p
        double scale
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        result[l] += density*kernelzz(
            lons[l], sinlats[l], coslats[l], radii[l], scale, lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double kernelzz(
    double lon, double sinlat, double coslat, double radius, double scale,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, rc_sqr, l_sqr, cospsi, deltaz
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltaz = rc[k]*cospsi - radius
                result += kappa*(3*deltaz**2 - l_sqr)/(l_sqr**2.5)
    return result*scale
