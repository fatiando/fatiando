"""
Calculates the potential fields of a tesseroid.
"""

import numpy
from numpy import sin, cos, sqrt

from fatiando.mesher import Tesseroid
from fatiando.constants import MEAN_EARTH_RADIUS, G, SI2MGAL, SI2EOTVOS

_ratio_potential = 1.
_glq_nodes_o2 = numpy.array([-0.577350269, 0.577350269])

def _kernel_potential(nodes_lon, nodes_lat, nodes_r, lon, lat, radius):
    """
    """
    coslat = cos(lat)
    sinlat = sin(lat)
    result = 0.0
    for lonc in nodes_lon:
        coslon = cos(lon - lonc)
        for latc in nodes_lat:
            sinlatc = sin(latc)
            coslatc = cos(latc)
            for rc in nodes_r:
                l_sqr = radius**2 + rc**2 - 2.*radius*rc*(sinlat*sinlatc +
                    coslat*coslatc*coslon)
                result += (rc**2)*coslatc/sqrt(l_sqr)
    return result

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

def _optimal_discretize(tesseroids, lons, lats, heights, kernel, ratio):
    if len(lons) != len(lats) != len(heights):
        raise ValueError('lons, lats, and heights must have the same len')
    lifo_maxsize = 1000
    d2r = numpy.pi/180.
    lons = d2r*lons
    lats = d2r*lats
    radius = MEAN_EARTH_RADIUS + heights
    npoints = len(lons)
    result = numpy.zeros(npoints)
    for tesseroid in tesseroids:
        if 'density' not in tesseroid.props:
            continue
        for i in xrange(npoints):
            lifo = [tesseroid]
            while lifo:
                tes = lifo.pop()
                tes_radius = tes.top + MEAN_EARTH_RADIUS
                tes_lat = d2r*0.5*(tes.s + tes.n)
                tes_lon = d2r*0.5*(tes.w + tes.e)
                distance = sqrt(
                    radius[i]**2 + tes_radius**2 - 2.*radius[i]*tes_radius*(
                        sin(lats[i])*sin(tes_lat) + 
                        cos(lats[i])*cos(tes_lat)*cos(lons[i] - tes_lon)
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
                    nodes_lon = d2r*(0.5*dlon*_glq_nodes_o2 + 
                        0.5*(tes.e + tes.w))
                    nodes_lat = d2r*(0.5*dlat*_glq_nodes_o2 + 
                        0.5*(tes.n + tes.s))
                    nodes_r = (0.5*dr*_glq_nodes_o2 + 
                        0.5*(tes.top + tes.bottom + 2.*MEAN_EARTH_RADIUS))
                    result[i] += (G*tes.props['density']*
                        d2r*dlon*d2r*dlat*dr*0.125*
                        kernel(nodes_lon, nodes_lat, nodes_r, lons[i], lats[i],
                            radius[i]))
    return result

def potential(tesseroids, lons, lats, heights):
    """
    Calculate the gravitational potential due to a tesseroid model.
    """
    return _optimal_discretize(tesseroids, lons, lats, heights, 
        _kernel_potential, _ratio_potential)
