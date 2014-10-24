"""
Cython kernels for the fatiando.gravmag.tesseroid module.

Used to optimize some slow tasks and compute the actual gravitational fields.
"""
from __future__ import division
import numpy

from ..constants import MEAN_EARTH_RADIUS, G

from libc.math cimport sin, cos, sqrt, atan2
# Import Cython definitions for numpy
cimport numpy
cimport cython

# To calculate sin and cos simultaneously
cdef extern from "math.h":
    void sincos(double x, double* sinx, double* cosx)

cdef:
    double d2r = numpy.pi/180.
    double[::1] nodes
nodes = numpy.array([-0.577350269189625731058868041146,
                     0.577350269189625731058868041146])


@cython.boundscheck(False)
@cython.wraparound(False)
def too_close(
    tesseroid,
    numpy.ndarray[double, ndim=1] lon,
    numpy.ndarray[double, ndim=1] sinlat,
    numpy.ndarray[double, ndim=1] coslat,
    numpy.ndarray[double, ndim=1] radius,
    double ratio,
    numpy.ndarray[long, ndim=1] points):
    """
    Separate 'points' in two:
      The first part doesn't need to be divided.
      The second part is too close and needs to be divided.
    Returns the index of the division point:
        points[:i] -> don't divide
        points[i:] -> divide
    How close is measured by:
        too close ->  distance < ratio*size
    'points' is a list of the indices corresponding to observation points that
    need to be checked.
    """
    cdef:
        int i, j, p
        double rt, rt_sqr, latt, lont, sinlatt, coslatt, distance, cospsi
        double w, e, s, n, top, bottom, size
    w, e, s, n, top, bottom = tesseroid
    rt = top + MEAN_EARTH_RADIUS
    rt_sqr = rt**2
    latt = d2r*0.5*(s + n)
    sincos(latt, &sinlatt, &coslatt)
    lont = d2r*0.5*(w + e)
    size = max([MEAN_EARTH_RADIUS*d2r*(e - w),
                MEAN_EARTH_RADIUS*d2r*(n - s),
                top - bottom])
    # Will compare with the distance**2 so I don't have to calculate sqrt
    value = (size*ratio)**2
    i = 0
    j = len(points) - 1
    while i <= j:
        p = points[i]
        cospsi = sinlat[p]*sinlatt + coslat[p]*coslatt*cos(lon[p] - lont)
        distance = radius[p]**2 + rt_sqr - 2*radius[p]*rt*cospsi
        if distance < 1e-20:
            raise ValueError("Can't calculate directly on the tesseroid")
        elif distance > value:
            i += 1
        else:
            points[i] = points[j]
            points[j] = p
            j -= 1
    return i


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
        sincos(latc, &sinlatc[i], &coslatc[i])
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
        double scale, kappa, sinlat, coslat, r_sqr, rc_sqr, coslon, l_sqr
        double cospsi
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        r_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    rc_sqr = rc[k]**2
                    l_sqr = r_sqr + rc_sqr - 2*radii[l]*rc[k]*cospsi
                    kappa = rc_sqr*coslatc[j]
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
        double scale, kappa, sinlat, coslat, rc_sqr, r_sqr, coslon, l_sqr
        double cospsi
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        r_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    rc_sqr = rc[k]**2
                    l_sqr = r_sqr + rc_sqr - 2*radii[l]*rc[k]*cospsi
                    kappa = rc_sqr*coslatc[j]
                    result[l] += density*scale*kappa*rc[k]*kphi/(l_sqr**1.5)


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
        double scale, kappa, sinlat, coslat, r_sqr, coslon, l_sqr
        double cospsi, sinlon, rc_sqr
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        r_sqr = radii[l]**2
        for i in range(2):
            sincos(lonc[i] - lons[l], &sinlon, &coslon)
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    rc_sqr = rc[k]**2
                    l_sqr = r_sqr + rc_sqr - 2*radii[l]*rc[k]*cospsi
                    kappa = rc_sqr*coslatc[j]
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
        double scale, kappa, sinlat, coslat, r_sqr, coslon, l_sqr
        double cospsi, rc_sqr
    # Put the nodes in the current range
    scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
    # Start the numerical integration
    for p in range(len(points)):
        l = points[p]
        sinlat = sinlats[l]
        coslat = coslats[l]
        r_sqr = radii[l]**2
        for i in range(2):
            coslon = cos(lons[l] - lonc[i])
            for j in range(2):
                cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                for k in range(2):
                    rc_sqr = rc[k]**2
                    l_sqr = r_sqr + rc_sqr - 2*radii[l]*rc[k]*cospsi
                    kappa = rc_sqr*coslatc[j]
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
        sincos(lonc[i] - lon, &sinlon, &coslon)
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
        sincos(lonc[i] - lon, &sinlon, &coslon)
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
        sincos(lonc[i] - lon, &sinlon, &coslon)
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
    double ratio,
    numpy.ndarray[double, ndim=1] lon,
    numpy.ndarray[double, ndim=1] sinlat,
    numpy.ndarray[double, ndim=1] coslat,
    numpy.ndarray[double, ndim=1] radius,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations (specified
    by the indices in 'points').
    """
    cdef:
        unsigned int l
    # Start the numerical integration
    for l in range(len(result)):
        result[l] += density*kernelzz([tesseroid], ratio,
            lon[l], sinlat[l], coslat[l], radius[l], lonc, sinlatc,
            coslatc, rc)


# Computes the kernel part of the gravity gradient component
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double kernelzz(
    tesseroids, double ratio,
    double lon, double sinlat, double coslat, double radius,
    double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double nlon, nlat, nr
        double w, e, s, n, top, bottom
        double rt, lont, latt, sinlatt, coslatt, distance
        double sinn, cosn, sins, coss, sindlon, cosdlon, dlon, dlat, dr
        double kappa, r_sqr, coslon, rc_sqr, l_sqr, l_5, cospsi, deltaz, scale
        double result
    result = 0
    for tesseroid in tesseroids:
        w, e, s, n, top, bottom = tesseroid
        # Calculate the distance to the observation point
        rt = top + MEAN_EARTH_RADIUS
        lont = d2r*0.5*(w + e)
        latt = d2r*0.5*(s + n)
        sinlatt = sin(latt)
        coslatt = cos(latt)
        cospsi = sinlat*sinlatt + coslat*coslatt*cos(lon - lont)
        distance = sqrt(radius**2 + rt**2 - 2*radius*rt*cospsi)
        # Calculate the dimensions of the tesseroid
        # Will use Vincenty's formula to calculate great-circle distance
        # for more accuracy (just in case)
        sinn = sin(d2r*n)
        cosn = cos(d2r*n)
        sins = sin(d2r*s)
        coss = cos(d2r*s)
        sindlon = sin(d2r*(e - w))
        cosdlon = cos(d2r*(e - w))
        dlon = MEAN_EARTH_RADIUS*atan2(
            sqrt((coslatt*sindlon)**2
                 + (coslatt*sinlatt - sinlatt*coslatt*cosdlon)**2),
            sinlatt**2 + cosdlon*coslatt**2)
        dlat = MEAN_EARTH_RADIUS*atan2(
            coss*sinn - sins*cosn, sins*sinn + coss*cosn)
        dr = top - bottom
        # Check which dimensions I have to divide
        nlon = 1
        nlat = 1
        nr = 1
        if distance < ratio*dlon:
            nlon = 2
        if distance < ratio*dlat:
            nlat = 2
        if distance < ratio*dr:
            nr = 2
        if nlon == 1 and nlat == 1 and nr == 1:
            # Put the nodes in the current range
            scale = scale_nodes(tesseroid, lonc, sinlatc, coslatc, rc)
            r_sqr = radius**2
            for i in range(2):
                coslon = cos(lon - lonc[i])
                for j in range(2):
                    cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
                    for k in range(2):
                        rc_sqr = rc[k]**2
                        l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                        l_5 = l_sqr**2.5
                        kappa = rc_sqr*coslatc[j]
                        deltaz = rc[k]*cospsi - radius
                        result += scale*kappa*(3*deltaz**2 - l_sqr)/l_5
        else:
            dlon = (e - w)/nlon
            dlat = (n - s)/nlat
            dr = (top - bottom)/nr
            split = [[w, w + dlon, s, s + dlat, bottom + dr, bottom]
                    for w in [w, w + dlon] for s in [s, s + dlat]
                    for bottom in [bottom, bottom + dr]]
            result += kernelzz(split, ratio, lon, sinlat, coslat, radius, lonc,
                    sinlatc, coslatc, rc)
    return result
