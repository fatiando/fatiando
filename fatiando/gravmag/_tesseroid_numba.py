"""
A numba implementation of the tesseroid gravity effects.

These functions compute the effect of a single tesseroid. They are used by
fatiando.gravmag.tesseroid as a backend and are not meant to be used directly.
"""
from __future__ import division
import numba
import numpy as np

from ..constants import MEAN_EARTH_RADIUS
from . import _tesseroid_numpy


nodes = np.array([-0.577350269189625731058868041146,
                  0.577350269189625731058868041146])

# jit compile the kernel functions from the numpy implementation
kernelV = numba.jit(_tesseroid_numpy.kernelV, nopython=True)
kernelx = numba.jit(_tesseroid_numpy.kernelx, nopython=True)
kernely = numba.jit(_tesseroid_numpy.kernely, nopython=True)
kernelz = numba.jit(_tesseroid_numpy.kernelz, nopython=True)
kernelxx = numba.jit(_tesseroid_numpy.kernelxx, nopython=True)
kernelxy = numba.jit(_tesseroid_numpy.kernelxy, nopython=True)
kernelxz = numba.jit(_tesseroid_numpy.kernelxz, nopython=True)
kernelyy = numba.jit(_tesseroid_numpy.kernelyy, nopython=True)
kernelyz = numba.jit(_tesseroid_numpy.kernelyz, nopython=True)
kernelzz = numba.jit(_tesseroid_numpy.kernelzz, nopython=True)


@numba.jit(nopython=True)
def scale_nodes(w, e, s, n, top, bottom, nodes, lonc, sinlatc, coslatc, rc):
    d2r = np.pi/180
    dlon = d2r*(e - w)
    dlat = d2r*(n - s)
    dr = top - bottom
    # Scale the GLQ nodes to the integration limits
    for i in range(len(nodes)):
        lonc[i] = 0.5*dlon*nodes[i] + d2r*0.5*(e + w)
        latc = 0.5*dlat*nodes[i] + d2r*0.5*(n + s)
        sinlatc[i] = np.sin(latc)
        coslatc[i] = np.cos(latc)
        rc[i] = (0.5*dr*nodes[i] +
                 0.5*(top + bottom) + MEAN_EARTH_RADIUS)
    scale = dlon*dlat*dr*0.125
    return scale


@numba.jit(nopython=True)
def distance_size(lon, coslat, sinlat, radius, w, e, s, n, top, bottom):
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


@numba.jit(nopython=True)
def split(w, e, s, n, top, bottom, nlon, nlat, nr, stack, stktop):
    dlon = (e - w)/nlon
    dlat = (n - s)/nlat
    dr = (top - bottom)/nr
    for i in xrange(nlon):
        for j in xrange(nlat):
            for k in xrange(nr):
                stktop += 1
                stack[stktop, 0] = w + i*dlon
                stack[stktop, 1] = w + (i + 1)*dlon
                stack[stktop, 2] = s + j*dlat
                stack[stktop, 3] = s + (j + 1)*dlat
                stack[stktop, 4] = bottom + (k + 1)*dr
                stack[stktop, 5] = bottom + k*dr
    return stktop


@numba.jit(nopython=True)
def divisions(distance, Llon, Llat, Lr, ratio):
    nlon = 1 if distance/Llon > ratio else 2
    nlat = 1 if distance/Llat > ratio else 2
    nr = 1 if distance/Lr > ratio else 2
    return nlon, nlat, nr, nlon*nlat*nr


def make_buffers(tesseroid, stack_size):
    bounds = np.array(tesseroid.get_bounds())
    stack = np.empty((stack_size, 6))
    lonc = np.empty_like(nodes)
    sinlatc = np.empty_like(nodes)
    coslatc = np.empty_like(nodes)
    rc = np.empty_like(nodes)
    return bounds, stack, lonc, sinlatc, coslatc, rc


@numba.jit(looplift=True)
def gz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelz(lon[l], coslat[l], sinlat[l], radius[l],
                                 lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def potential(lon, sinlat, coslat, radius, tesseroid, density, ratio,
              stack_size, result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelV(lon[l], coslat[l], sinlat[l], radius[l],
                                 lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gx(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelx(lon[l], coslat[l], sinlat[l], radius[l],
                                 lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
       result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernely(lon[l], coslat[l], sinlat[l], radius[l],
                                 lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gxx(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelxx(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gxy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelxy(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gxz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelxz(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gyy(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelyy(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gyz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelyz(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel


@numba.jit(looplift=True)
def gzz(lon, sinlat, coslat, radius, tesseroid, density, ratio, stack_size,
        result):
    bounds, stack, lonc, sinlatc, coslatc, rc = make_buffers(tesseroid,
                                                             stack_size)
    for l in range(result.size):
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w, e, s, n, top, bottom = stack[stktop, :]
            stktop -= 1
            distance, Llon, Llat, Lr = distance_size(
                lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                bottom)
            nlon, nlat, nr, new_cells = divisions(distance, Llon, Llat, Lr,
                                                  ratio)
            if new_cells > 1:
                if new_cells + (stktop + 1) > stack_size:
                    raise OverflowError
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr, stack,
                               stktop)
            else:
                scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                    sinlatc, coslatc, rc)
                kernel = kernelzz(lon[l], coslat[l], sinlat[l], radius[l],
                                  lonc, sinlatc, coslatc, rc)
                result[l] += density*scale*kernel
