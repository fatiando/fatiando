"""
This is a Python + Numpy implementation of the potential field effects of
spheres.
"""
from __future__ import division

import numpy

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils


def tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pmx, pmy, pmz = pmag * fx, pmag * fy, pmag * fz
        else:
            pmx, pmy, pmz = pmag
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props
                              and pmag is None):
            continue
        radius = sphere.radius
        # Get the intensity and unit vector from the magnetization
        if pmag is None:
            mag = sphere.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        else:
            mx, my, mz = pmx, pmy, pmz
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        # Calculate the 3 components of B
        dotprod = mx * x + my * y + mz * z
        r_sqr = x ** 2 + y ** 2 + z ** 2
        r5 = r_sqr ** (2.5)
        moment = 4. * numpy.pi * (radius ** 3) / 3.
        bx = moment * (3 * dotprod * x - r_sqr * mx) / r5
        by = moment * (3 * dotprod * y - r_sqr * my) / r5
        bz = moment * (3 * dotprod * z - r_sqr * mz) / r5
        res += fx * bx + fy * by + fz * bz
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, spheres):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        radius = sphere.radius
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        v1 = kernelxx(xp, yp, zp, sphere)
        v2 = kernelxy(xp, yp, zp, sphere)
        v3 = kernelxz(xp, yp, zp, sphere)
        res += (v1 * mx + v2 * my + v3 * mz)
    res *= CM * T2NT
    return res


def by(xp, yp, zp, spheres):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        radius = sphere.radius
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        v2 = kernelxy(xp, yp, zp, sphere)
        v4 = kernelyy(xp, yp, zp, sphere)
        v5 = kernelyz(xp, yp, zp, sphere)
        res += (v2 * mx + v4 * my + v5 * mz)
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, spheres):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        radius = sphere.radius
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        v3 = kernelxz(xp, yp, zp, sphere)
        v5 = kernelyz(xp, yp, zp, sphere)
        v6 = kernelzz(xp, yp, zp, sphere)
        res += (v3 * mx + v5 * my + v6 * mz)
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        # Turns out that taking the sqrt and multiplying is ~4 times faster
        # than raising to 1.5 power.
        r = numpy.sqrt(dx*dx + dy*dy + dz*dz)
        r_cb = r*r*r
        mass = density*4*numpy.pi*(radius**3)/3
        res += mass*dz/r_cb
    res *= G * SI2MGAL
    return res


def gxx(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelxx(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelxy(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelxz(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def gyy(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelyy(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelyz(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, spheres, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        res += density * kernelzz(xp, yp, zp, sphere)
    res *= G * SI2EOTVOS
    return res


def kernelxx(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * (((3 * dx ** 2) - r_2) / r_5)


def kernelxy(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * ((3 * dx * dy) / r_5)


def kernelxz(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * ((3 * dx * dz) / r_5)


def kernelyy(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * (((3 * dy ** 2) - r_2) / r_5)


def kernelyz(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * ((3 * dy * dz) / r_5)


def kernelzz(xp, yp, zp, sphere):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    radius = sphere.radius
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z - zp
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = 4. * numpy.pi * (radius ** 3) / 3.
    return volume * (((3 * dz ** 2) - r_2) / r_5)
