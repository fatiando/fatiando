"""
A pure Python + numpy implementation of the tesseroid gravity effects.

This is **much** slower than the numba version. It is kept as a backup and to
test the numba implementation.
"""
from __future__ import division
import numpy as np

from ..constants import MEAN_EARTH_RADIUS


nodes = np.array([-0.577350269189625731058868041146,
                  0.577350269189625731058868041146])


def adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernel, result):
    """
    Perform the adaptive discretization of a tesseroid and compute the effect
    of the given kernel function.
    """
    for l in xrange(lon.size):
        stack = [tesseroid]
        while stack:
            t = stack.pop()
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], t)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + len(stack) > stack_size:
                    raise OverflowError('Tesseroid stack overflowed')
                stack.extend(t.split(nlon, nlat, nr))
            else:
                lonc, sinlatc, coslatc, rc, scale = scale_nodes(t.get_bounds(),
                                                                nodes)
                tmp = kernel(lon[l], coslat[l], sinlat[l], radius[l],
                             lonc, sinlatc, coslatc, rc)
                result[l] += scale*density*tmp


def divisions(distance, Llon, Llat, Lr, ratio):
    "How many divisions should be made per dimension"
    nlon = 1 if distance/Llon > ratio else 2
    nlat = 1 if distance/Llat > ratio else 2
    nr = 1 if distance/Lr > ratio else 2
    return nlon, nlat, nr, nlon*nlat*nr


def scale_nodes(bounds, nodes):
    "Put the GLQ nodes in the integration limit"
    w, e, s, n, top, bottom = bounds
    d2r = np.pi/180
    dlon = d2r*(e - w)
    dlat = d2r*(n - s)
    dr = top - bottom
    # Scale the GLQ nodes to the integration limits
    lonc = 0.5*dlon*nodes + d2r*0.5*(e + w)
    latc = 0.5*dlat*nodes + d2r*0.5*(n + s)
    sinlatc = np.sin(latc)
    coslatc = np.cos(latc)
    rc = (0.5*dr*nodes +
          0.5*(top + bottom) + MEAN_EARTH_RADIUS)
    scale = dlon*dlat*dr*0.125
    return lonc, sinlatc, coslatc, rc, scale


def distance_size(lon, coslat, sinlat, radius, cell):
    "Calculate the distance to the center of the tesseroid and its dimensions"
    w, e, s, n, top, bottom = cell.get_bounds()
    d2r = np.pi/180
    rt = 0.5*(top + bottom) + MEAN_EARTH_RADIUS
    lont = d2r*0.5*(w + e)
    latt = d2r*0.5*(s + n)
    sinlatt = np.sin(latt)
    coslatt = np.cos(latt)
    cospsi = sinlat*sinlatt + coslat*coslatt*np.cos(lon - lont)
    distance = np.sqrt(radius**2 + rt**2 - 2*radius*rt*cospsi)
    # Calculate the dimensions of the tesseroid in meters
    rtop = top + MEAN_EARTH_RADIUS
    Llon = rtop*np.arccos(sinlatt**2 + (coslatt**2)*np.cos(d2r*(e - w)))
    Llat = rtop*np.arccos(np.sin(d2r*n)*np.sin(d2r*s) +
                          np.cos(d2r*n)*np.cos(d2r*s))
    Lr = top - bottom
    return distance, Llon, Llat, Lr


# These are the engine functions that gravmag.tesseroid calls. They are
# basically just a call to adaptive discretization with the appropriate kernel.
def potential(lon, sinlat, coslat, radius, tesseroid, density, ratio,
              stack_size, result):
    """
    Calculate the potential of a single tesseroid using adaptive discretization

    Parameters:

    * lon : 1d-array
        The longitudes of the computation point in radians.
    * sinlat, coslat : 1d-array
        The sine and cossine of the latitudes of the computation points.
    * radius : 1d-array
        The radius coordinate of the computation point.
    * tesseroid : fatiando.mesher.Tesseroid
        The tesseroid.
    * density : float
        The density of the tesseroid.
    * ratio : float > 0
        The distance-size ratio used in the adaptive discretization.
    * stack_size : int > 0
        The maximum allowed size of the tesseroid stack used in the adaptive
        discretization.
    * result : 1d-array
        Buffer used to return the output. Should be initialized with zeros.

    """
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelV, result)


# Docstrings of the other engines are the same,
def gx(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelx, result)


def gy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernely, result)


def gz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelz, result)


def gxx(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelxx, result)


def gxy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelxy, result)


def gxz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelxz, result)


def gyy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelyy, result)


def gyz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelyz, result)


def gzz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    adaptive_discretization(lon, coslat, sinlat, radius, tesseroid, density,
                            ratio, stack_size, kernelzz, result)


# Kernel functions for tesseroid gravitational effects. This is where the
# physics is.
def kernelV(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    """
    Compute the kernel effect on a single point using GLQ integration.

    Parameters:

    * lon : float
        The longitude of the computation point in radians.
    * coslat, sinlat : float
        The sine and cosine of the latitude of the computation point.
    * radius : float
        The radius coordinate of the computation point.
    * lonc, sinlatc, coslatc, rc : 1d-arrays
        The coordinates of the GLQ nodes scaled to the integration limits (the
        dimensions of the tesseroid). sinlatc and coslatc are the sine and
        cosine of the latitude. lon should be in radians. rc is the radial
        coordinate.

    Returns:

    * result : float
        The kernel value

    """
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa/np.sqrt(l_sqr)
    return result


def kernelx(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*rc[k]*kphi/(l_sqr**1.5)
    return result


def kernely(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
    return result


def kernelz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(rc[k]*cospsi - radius)/(l_sqr**1.5)
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    result *= -1
    return result


def kernelxx(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(3*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5)
    return result


def kernelxy(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc_sqr*kphi*coslatc[j]*sinlon/(l_sqr**2.5)
    return result


def kernelxz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_5 = (r_sqr + rc_sqr - 2*radius*rc[k]*cospsi)**2.5
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc[k]*kphi*(rc[k]*cospsi - radius)/l_5
    return result


def kernelyy(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                result += kappa*(3*(deltay**2) - l_sqr)/(l_sqr**2.5)
    return result


def kernelyz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radius
                result += kappa*3.*deltay*deltaz/(l_sqr**2.5)
    return result


def kernelzz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                l_5 = l_sqr**2.5
                kappa = rc_sqr*coslatc[j]
                deltaz = rc[k]*cospsi - radius
                result += kappa*(3*deltaz**2 - l_sqr)/l_5
    return result
