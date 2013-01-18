"""
Kernel functions for calculating the potential fields of a tesseroid using a 
second order Gauss-Legendre Quadrature integration.
"""

from math import cos, sin, sqrt

import numpy

from fatiando.mesher import Tesseroid
from fatiando.constants import MEAN_EARTH_RADIUS, G, SI2MGAL, SI2EOTVOS

__all__ = ['_optimal_discretize', '_kernel_potential', '_kernel_gx', 
    '_kernel_gy', '_kernel_gz', '_kernel_gxx', '_kernel_gxy', '_kernel_gxz', 
    '_kernel_gyy', '_kernel_gyz', '_kernel_gzz']

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

def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio,
    nodes=numpy.array([-0.577350269, 0.577350269]),
    weights=numpy.array([1., 1.])):
    """
    """
    if len(lons) != len(lats) != len(heights):
        raise ValueError('lons, lats, and heights must have the same len')
    lifo_maxsize = 1000
    # Convert things to radians
    d2r = numpy.pi/180.
    lons = d2r*lons
    lats = d2r*lats
    # Transform the heights into radii
    radii = MEAN_EARTH_RADIUS + heights
    # Get some lenghts
    npoints = len(lons)
    nnodes = len(nodes)
    result = numpy.zeros(npoints)
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
                tes_radius = tes.top + MEAN_EARTH_RADIUS
                tes_lat = d2r*0.5*(tes.s + tes.n)
                tes_lon = d2r*0.5*(tes.w + tes.e)
                distance = sqrt(
                    radius**2 + tes_radius**2 - 2.*radius*tes_radius*(
                        sinlat*sin(tes_lat) +
                        coslat*cos(tes_lat)*cos(lon - tes_lon)
                    ))
                dimension = max([MEAN_EARTH_RADIUS*d2r*(tes.e - tes.w),
                                 MEAN_EARTH_RADIUS*d2r*(tes.n - tes.s),
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
                        0.5*(tes.top + tes.bottom + 2.*MEAN_EARTH_RADIUS))
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
                                result[l] += tmp*(
                                    weights[i]*weights[j]*weights[k]*
                                    kernel(radius, coslat, sinlat, coslon, 
                                        sinlon, sinlatc, coslatc, rc, l_sqr, 
                                        kappa))
    return result

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

def _kernel_gzz(radius, coslat, sinlat, coslon, sinlon, sinlatc, coslatc, rc, 
    l_sqr, kappa):
    cospsi = sinlat*sinlatc + coslat*coslatc*coslon
    deltaz = rc*cospsi - radius
    return kappa*(3.*deltaz**2 - l_sqr)/(l_sqr**2.5)
