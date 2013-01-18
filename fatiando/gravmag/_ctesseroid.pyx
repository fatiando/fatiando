# cython: profile=True
"""
Cython implementation of the Kernel functions for calculating the potential 
fields of a tesseroid using a second order Gauss-Legendre Quadrature 
iintegration.
"""

import numpy

from libc.math cimport cos, sin, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython


DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from fatiando.constants import MEAN_EARTH_RADIUS, G

__all__ = ['_optimal_discretize', '_kernel_potential', '_kernel_gx', 
    '_kernel_gy', '_kernel_gz', '_kernel_gxx', '_kernel_gxy', '_kernel_gxz', 
    '_kernel_gyy', '_kernel_gyz', '_kernel_gzz']


def _split(w, e, s, n, top, bottom):
    dlon = 0.5*(e - w)
    dlat = 0.5*(n - s)
    dh = 0.5*(top - bottom)
    wests = [w, w + dlon]
    souths = [s, s + dlat]
    bottoms = [bottom, bottom + dh]
    split = [
        [i, i + dlon, j, j + dlat, k + dh, k]
        for i in wests for j in souths for k in bottoms]
    return split

@cython.boundscheck(False)
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
    cdef unsigned int l, t, npoints, nnodes, lifo_max
    cdef int lifo_top
    cdef numpy.ndarray[DTYPE_T, ndim=1] result, radii, nodes_lon, nodes_lat
    cdef numpy.ndarray[DTYPE_T, ndim=1] nodes_r, sineslatc, cossineslatc
    cdef numpy.ndarray[DTYPE_T, ndim=2] lifo
    cdef DTYPE_T d2r, coslat, sinlat, lon
    cdef DTYPE_T density, radius, distance, dimension
    cdef DTYPE_T dlon, dlat, dr, tmp
    cdef DTYPE_T w, e, s, n, top, bottom

    if len(lons) != len(lats) != len(heights):
        raise ValueError('lons, lats, and heights must have the same len')
    
    lifo_max = 1000
    lifo = numpy.zeros((lifo_max, 6), dtype=DTYPE)
    # Convert things to radians
    d2r = numpy.pi/180.
    lons = d2r*lons
    lats = d2r*lats
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Get some lenghts
    npoints = len(lons)
    nnodes = len(nodes)
    result = numpy.zeros(npoints, dtype=DTYPE)
    for tesseroid in tesseroids:
        if 'density' not in tesseroid.props:
            continue
        density = tesseroid.props['density']
        bounds = [d2r*tesseroid.w, d2r*tesseroid.e, d2r*tesseroid.s,
            d2r*tesseroid.n, tesseroid.top, tesseroid.bottom]
        for l in range(npoints):
            # Pre-compute the sine and cossine of latitude to save computations
            coslat = cos(lats[l])
            sinlat = sin(lats[l])
            lon = lons[l]
            radius = radii[l]
            lifo_top = 0
            lifo[lifo_top] = bounds 
            while lifo_top != -1:
                w, e, s, n, top, bottom = lifo[lifo_top]
                lifo_top -= 1
                distance, dimension = _measure(w, e, s, n, top, bottom, radius,
                    lon, coslat, sinlat)
                if (distance > 0 and distance < ratio*dimension and
                    lifo_top + 8 < lifo_max):
                        lifo[lifo_top + 1:lifo_top + 9, :] = _split(w, e, s, n,
                            top, bottom)
                        lifo_top += 8
                else:
                    dlon = e - w
                    dlat = n - s
                    dr = top - bottom
                    # Scale the GLQ nodes to the integration limits
                    nodes_lon = 0.5*dlon*nodes + 0.5*(e + w)
                    nodes_lat = 0.5*dlat*nodes + 0.5*(n + s)
                    nodes_r = (0.5*dr*nodes +
                        0.5*(top + bottom + 2.*MEAN_EARTH_RADIUS))
                    # Pre-compute the sines and cossines to save time
                    sineslatc = numpy.sin(nodes_lat)
                    cossineslatc = numpy.cos(nodes_lat)
                    # Do the GLQ integration of the kernel
                    tmp = G*density*dlon*dlat*dr*0.125
                    result[l] = result[l] + tmp*(
                        kernel(radius, lon, coslat, sinlat, nodes_lon, nodes_r, 
                            sineslatc, cossineslatc))
    return result

def _measure(w, e, s, n, top, bottom, radius, lon, coslat, sinlat):
    tes_radius = top + MEAN_EARTH_RADIUS
    tes_lat = 0.5*(s + n)
    tes_lon = 0.5*(w + e)
    distance = sqrt(
        radius**2 + tes_radius**2 - 2.*radius*tes_radius*(
            sinlat*sin(tes_lat) +
            coslat*cos(tes_lat)*cos(lon - tes_lon)
        ))
    dimension = max([MEAN_EARTH_RADIUS*(e - w),
                     MEAN_EARTH_RADIUS*(n - s),
                     top - bottom])
    return distance, dimension

def _kernel_potential(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, 
    rc, l_sqr, kappa):
    return kappa/sqrt(l_sqr)

def _kernel_gx(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    kphi = coslat*sinlatc - sinlat*coslatc*coslon
    return kappa*rc*kphi/(l_sqr**1.5)

def _kernel_gy(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    return kappa*rc*coslatc*sinlon/(l_sqr**1.5)

def _kernel_gz(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    cospsi = sinlat*sinlatc + coslat*coslatc*coslon
    return kappa*(rc*cospsi - radius)/(l_sqr**1.5)

def _kernel_gxx(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    kphi = coslat*sinlatc - sinlat*coslatc*coslon
    return kappa*(3.*((rc*kphi)**2) - l_sqr)/(l_sqr**2.5)

def _kernel_gxy(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    kphi = coslat*sinlatc - sinlat*coslatc*coslon
    return kappa*(3.*(rc**2)*kphi*coslatc*sinlon)/(l_sqr**2.5)

def _kernel_gxz(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    cospsi = sinlat*sinlatc + coslat*coslatc*coslon
    kphi = coslat*sinlatc - sinlat*coslatc*coslon
    return kappa*3.*rc*kphi*(rc*cospsi - radius)/(l_sqr**2.5)

def _kernel_gyy(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    deltay = rc*coslatc*sinlon
    return kappa*(3.*(deltay**2) - l_sqr)/(l_sqr**2.5)

def _kernel_gyz(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    cospsi = sinlat*sinlatc + coslat*coslatc*coslon
    deltay = rc*coslatc*sinlon
    deltaz = rc*cospsi - radius
    return kappa*3.*deltay*deltaz/(l_sqr**2.5)

@cython.boundscheck(False)
def _kernel_gzz(DTYPE_T radius, DTYPE_T lon, DTYPE_T coslat, DTYPE_T sinlat, 
    numpy.ndarray[DTYPE_T, ndim=1] nodes_lon, 
    numpy.ndarray[DTYPE_T, ndim=1] nodes_r, 
    numpy.ndarray[DTYPE_T, ndim=1] sineslatc, 
    numpy.ndarray[DTYPE_T, ndim=1] cossineslatc):
    cdef unsigned int i, j, k
    cdef DTYPE_T coslon, sinlon, sinlatc, coslatc, rc, l_sqr, kappa, result
    cdef DTYPE_T cospsi, deltaz
    result = 0.0
    for i in range(2):
        coslon = cos(lon - nodes_lon[i])
        sinlon = sin(nodes_lon[i] - lon)
        for j in range(2):
            sinlatc = sineslatc[j]
            coslatc = cossineslatc[j]
            cospsi = sinlat*sinlatc + coslat*coslatc*coslon
            for k in range(2):
                rc = nodes_r[k]
                l_sqr = (radius**2 + rc**2 - 2.*radius*rc*(
                    sinlat*sinlatc + coslat*coslatc*coslon))
                kappa = (rc**2)*coslatc
                deltaz = rc*cospsi - radius
                result += kappa*(3.*deltaz**2 - l_sqr)/(l_sqr**2.5)
    return result
