# cython: profile=False
"""
Cython implementation of the Kernel functions for calculating the potential 
fields of a tesseroid using a second order Gauss-Legendre Quadrature 
iintegration.
"""

import numpy

from libc.math cimport cos, sin, sqrt
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from fatiando.mesher import Tesseroid
from fatiando.constants import MEAN_EARTH_RADIUS, G

__all__ = ['_optimal_discretize']


def _split(tes):
    dlon = 0.5*(tes.e - tes.w)
    dlat = 0.5*(tes.n - tes.s)
    dh = 0.5*(tes.top - tes.bottom)
    wests = [tes.w, tes.w + dlon]
    souths = [tes.s, tes.s + dlat]
    bottoms = [tes.bottom, tes.bottom + dh]
    split = [
        Tesseroid(w, w + dlon, s, s + dlat, b + dh, b, props=dict(tes.props))
        for w in wests for s in souths for b in bottoms]
    return split

def _optimal_discretize(tesseroids, 
    numpy.ndarray[DTYPE_T, ndim=1] lons, 
    numpy.ndarray[DTYPE_T, ndim=1] lats, 
    numpy.ndarray[DTYPE_T, ndim=1] heights,
    kernel, 
    DTYPE_T ratio,
    numpy.ndarray[DTYPE_T, ndim=1] nodes=numpy.array([-0.577350269, 
        0.577350269]),
    numpy.ndarray[DTYPE_T, ndim=1] weights=numpy.array([1., 1.])):
    """
    """
    cdef int l, i, j, k, npoints, nnodes, lifo_maxsize
    cdef numpy.ndarray[DTYPE_T, ndim=1] result, radii, nodes_lon, nodes_lat
    cdef numpy.ndarray[DTYPE_T, ndim=1] nodes_r, sineslatc, cossineslatc
    cdef DTYPE_T earth_radius = MEAN_EARTH_RADIUS, d2r, coslat, sinlat, lon
    cdef DTYPE_T density, radius, distance, dimension, tes_radius, tes_lat
    cdef DTYPE_T tes_lon, dlon, dlat, dr, tmp, lonc, coslon, sinlon, sinlatc
    cdef DTYPE_T coslatc, rc, l_sqr, kappa

    if len(lons) != len(lats) != len(heights):
        raise ValueError('lons, lats, and heights must have the same len')
    lifo_maxsize = 1000
    # Convert things to radians
    d2r = numpy.pi/180.
    lons = d2r*lons
    lats = d2r*lats
    # Transform the heights into radii
    radii = earth_radius + heights
    # Get some lenghts
    npoints = len(lons)
    nnodes = len(nodes)
    result = numpy.zeros(npoints, dtype=DTYPE)
    for l in xrange(npoints):
        # Pre-compute the sine and cossine of latitude to save computations
        coslat = cos(lats[l])
        sinlat = sin(lats[l])
        lon = lons[l]
        radius = radii[l]
        for tesseroid in tesseroids:
            if 'density' not in tesseroid.props:
                continue
            lifo = [tesseroid]
            while lifo:
                tes = lifo.pop()
                tes_radius = tes.top + earth_radius
                tes_lat = d2r*0.5*(tes.s + tes.n)
                tes_lon = d2r*0.5*(tes.w + tes.e)
                distance = sqrt(
                    radius**2 + tes_radius**2 - 2.*radius*tes_radius*(
                        sinlat*sin(tes_lat) +
                        coslat*cos(tes_lat)*cos(lon - tes_lon)
                    ))
                dimension = max([earth_radius*d2r*(tes.e - tes.w),
                                 earth_radius*d2r*(tes.n - tes.s),
                                 tes.top - tes.bottom])
                if (distance > 0 and distance < ratio*dimension and
                    len(lifo) + 8 <= lifo_maxsize):
                        lifo.extend(_split(tes))
                else:
                    dlon = tes.e - tes.w
                    dlat = tes.n - tes.s
                    dr = tes.top - tes.bottom
                    # Scale the GLQ nodes to the integration limits
                    nodes_lon = d2r*(0.5*dlon*nodes + 0.5*(tes.e + tes.w))
                    nodes_lat = d2r*(0.5*dlat*nodes + 0.5*(tes.n + tes.s))
                    nodes_r = (0.5*dr*nodes +
                        0.5*(tes.top + tes.bottom + 2.*earth_radius))
                    # Pre-compute the sines and cossines to save time
                    sineslatc = numpy.sin(nodes_lat)
                    cossineslatc = numpy.cos(nodes_lat)
                    # Do the GLQ integration of the kernel
                    tmp = G*tes.props['density']*d2r*dlon*d2r*dlat*dr*0.125
                    for i in xrange(nnodes):
                        lonc = nodes_lon[i]
                        coslon = cos(lon - lonc)
                        sinlon = sin(lonc - lon)
                        for j in xrange(nnodes):
                            sinlatc = sineslatc[j]
                            coslatc = cossineslatc[j]
                            for k in xrange(nnodes):
                                rc = nodes_r[k]
                                l_sqr = (radius**2 + rc**2 - 2.*radius*rc*(
                                    sinlat*sinlatc + coslat*coslatc*coslon))
                                kappa = (rc**2)*coslatc
                                result[l] = result[l] + tmp*(
                                    weights[i]*weights[j]*weights[k]*
                                    kernel(radius, coslat, sinlat, coslon, 
                                        sinlon, sinlatc, coslatc, rc, l_sqr, 
                                        kappa))
    return result
