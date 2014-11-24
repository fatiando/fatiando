# Numpy implementation for the potential fields of a polygonal prism
from __future__ import division

import numpy
from numpy import arctan2, log, sqrt

from .. import utils
from ..constants import SI2MGAL, SI2EOTVOS, G, CM, T2NT


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pmx, pmy, pmz = pmag * fx, pmag * fy, pmag * fz
        else:
            pmx, pmy, pmz = pmag
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                             and pmag is None):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        else:
            mx, my, mz = pmx, pmy, pmz
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bx = (v1 * mx + v2 * my + v3 * mz)
        by = (v2 * mx + v4 * my + v5 * mz)
        bz = (v3 * mx + v5 * my + v6 * mz)
        res += fx * bx + fy * by + fz * bz
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        res += (v1 * mx + v2 * my + v3 * mz)
    res *= CM * T2NT
    return res


def by(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        res += (v2 * mx + v4 * my + v5 * mz)
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        res += (v3 * mx + v5 * my + v6 * mz)
    res *= CM * T2NT
    return res


def gz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 10 ** (-10)
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        nverts = prism.nverts
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        Z1_sqr = Z1 ** 2
        Z2_sqr = Z2 ** 2
        kernel = numpy.zeros(size, dtype=numpy.float)
        for k in range(nverts):
            Xk1 = x[k] - xp
            Yk1 = y[k] - yp
            Xk2 = x[(k + 1) % nverts] - xp
            Yk2 = y[(k + 1) % nverts] - yp
            p = Xk1 * Yk2 - Xk2 * Yk1
            p_sqr = p ** 2
            Qk1 = (Yk2 - Yk1) * Yk1 + (Xk2 - Xk1) * Xk1
            Qk2 = (Yk2 - Yk1) * Yk2 + (Xk2 - Xk1) * Xk2
            Ak1 = Xk1 ** 2 + Yk1 ** 2
            Ak2 = Xk2 ** 2 + Yk2 ** 2
            R1k1 = sqrt(Ak1 + Z1_sqr)
            R1k2 = sqrt(Ak2 + Z1_sqr)
            R2k1 = sqrt(Ak1 + Z2_sqr)
            R2k2 = sqrt(Ak2 + Z2_sqr)
            Ak1 = sqrt(Ak1)
            Ak2 = sqrt(Ak2)
            Bk1 = sqrt(Qk1 ** 2 + p_sqr)
            Bk2 = sqrt(Qk2 ** 2 + p_sqr)
            E1k1 = R1k1 * Bk1
            E1k2 = R1k2 * Bk2
            E2k1 = R2k1 * Bk1
            E2k2 = R2k2 * Bk2
            kernel += (Z2 - Z1) * (arctan2(Qk2, p) - arctan2(Qk1, p))
            kernel += Z2 * \
                (arctan2(Z2 * Qk1, R2k1 * p) - arctan2(Z2 * Qk2, R2k2 * p))
            kernel += Z1 * \
                (arctan2(Z1 * Qk2, R1k2 * p) - arctan2(Z1 * Qk1, R1k1 * p))
            Ck1 = Qk1 * Ak1
            Ck2 = Qk2 * Ak2
            # dummy helps prevent zero division errors
            kernel += 0.5 * p * (Ak1 / (Bk1 + dummy)) * (
                log((E1k1 - Ck1) / (E1k1 + Ck1 + dummy) + dummy) -
                log((E2k1 - Ck1) / (E2k1 + Ck1 + dummy) + dummy))
            kernel += 0.5 * p * (Ak2 / (Bk2 + dummy)) * (
                log((E2k2 - Ck2) / (E2k2 + Ck2 + dummy) + dummy) -
                log((E1k2 - Ck2) / (E1k2 + Ck2 + dummy) + dummy))
        res = res + kernel * density
    res *= G * SI2MGAL
    return res


def gxx(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelxx(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelxy(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelxz(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def gyy(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelyy(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelyz(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        density = prism.props['density']
        res += density * kernelzz(xp, yp, zp, prism)
    res *= G * SI2EOTVOS
    return res


def _integral_v1(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0 / aux1)
    g = X1 - (Y1 * n)
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux3 = (X1 * Y2) - (X2 * Y1)
    p = ((aux3 / aux2)) + dummy
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2 * d2), (p * R22))
    aux11 = arctan2((Z1 * d2), (p * R21))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d2))
    aux14 = ((p * aux12) / d2)
    res = (g * Y2 * aux13) + (n * aux14)
    aux10 = arctan2((Z2 * d1), (p * R12))
    aux11 = arctan2((Z1 * d1), (p * R11))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d1))
    aux14 = ((p * aux12) / d1)
    res -= (g * Y1 * aux13) + (n * aux14)
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (n * (aux15 - aux14))
    aux0 = (1.0 / (1.0 + (n * n)))
    res *= -aux0
    return res


def _integral_v2(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0 / aux1)
    g = X1 - (Y1 * n)
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux3 = (X1 * Y2) - (X2 * Y1)
    p = ((aux3 / aux2)) + dummy
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2 * d2), (p * R22))
    aux11 = arctan2((Z1 * d2), (p * R21))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d2))
    aux14 = ((p * aux12) / d2)
    res = (((g * g) + (g * n * Y2)) * aux13) - aux14
    aux10 = arctan2((Z2 * d1), (p * R12))
    aux11 = arctan2((Z1 * d1), (p * R11))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d1))
    aux14 = ((p * aux12) / d1)
    res -= (((g * g) + (g * n * Y1)) * aux13) - aux14
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (aux14 - aux15)
    aux0 = (1.0 / (1.0 + (n * n)))
    res *= aux0
    return res


def _integral_v3(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    n = (aux0 / aux1)
    g = X1 - (Y1 * n)
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = log((((R11 - d1) / (R11 + d1)) + dummy))
    aux11 = log((((R12 - d1) / (R12 + d1)) + dummy))
    aux12 = log((((R21 - d2) / (R21 + d2)) + dummy))
    aux13 = log((((R22 - d2) / (R22 + d2)) + dummy))
    aux14 = (1.0 / (2 * d1))
    aux15 = (1.0 / (2 * d2))
    aux16 = aux15 * (aux13 - aux12)
    res = (Y2 * (1.0 + (n * n)) + g * n) * aux16
    aux16 = aux14 * (aux11 - aux10)
    res -= (Y1 * (1.0 + (n * n)) + g * n) * aux16
    aux0 = (1.0 / (1.0 + (n * n)))
    res *= -aux0
    return res


def _integral_v4(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    m = (aux1 / aux0)
    c = Y1 - (X1 * m)
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux3 = (X1 * Y2) - (X2 * Y1)
    p = ((aux3 / aux2)) + dummy
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2 * d2), (p * R22))
    aux11 = arctan2((Z1 * d2), (p * R21))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d2))
    aux14 = ((p * aux12) / d2)
    res = (c * X2 * aux13) + (m * aux14)
    aux10 = arctan2((Z2 * d1), (p * R12))
    aux11 = arctan2((Z1 * d1), (p * R11))
    aux12 = aux10 - aux11
    aux13 = (aux12 / (p * d1))
    aux14 = ((p * aux12) / d1)
    res -= (c * X1 * aux13) + (m * aux14)
    aux10 = log(((Z2 + R22) + dummy))
    aux11 = log(((Z1 + R21) + dummy))
    aux12 = log(((Z2 + R12) + dummy))
    aux13 = log(((Z1 + R11) + dummy))
    aux14 = aux10 - aux11
    aux15 = aux12 - aux13
    res += (m * (aux15 - aux14))
    aux1 = (1.0 / (1.0 + (m * m)))
    res *= aux1
    return res


def _integral_v5(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    m = (aux1 / aux0)
    c = Y1 - (X1 * m)
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = log((((R11 - d1) / (R11 + d1)) + dummy))
    aux11 = log((((R12 - d1) / (R12 + d1)) + dummy))
    aux12 = log((((R21 - d2) / (R21 + d2)) + dummy))
    aux13 = log((((R22 - d2) / (R22 + d2)) + dummy))
    aux14 = (1.0 / (2 * d1))
    aux15 = (1.0 / (2 * d2))
    aux16 = aux15 * (aux13 - aux12)
    res = (X2 * (1.0 + (m * m)) + c * m) * aux16
    aux16 = aux14 * (aux11 - aux10)
    res -= (X1 * (1.0 + (m * m)) + c * m) * aux16
    aux1 = (1.0 / (1.0 + (m * m)))
    res *= aux1
    return res


def _integral_v6(X1, X2, Y1, Y2, Z1, Z2):
    dummy = 10. ** (-10)  # Used to avoid singularities
    aux0 = X2 - X1 + dummy
    aux1 = Y2 - Y1 + dummy
    aux2 = sqrt((aux0 * aux0) + (aux1 * aux1))
    aux3 = (X1 * Y2) - (X2 * Y1)
    p = ((aux3 / aux2)) + dummy
    aux4 = (aux0 * X1) + (aux1 * Y1)
    aux5 = (aux0 * X2) + (aux1 * Y2)
    d1 = ((aux4 / aux2)) + dummy
    d2 = ((aux5 / aux2)) + dummy
    aux6 = (X1 * X1) + (Y1 * Y1)
    aux7 = (X2 * X2) + (Y2 * Y2)
    aux8 = Z1 * Z1
    aux9 = Z2 * Z2
    R11 = sqrt(aux6 + aux8)
    R12 = sqrt(aux6 + aux9)
    R21 = sqrt(aux7 + aux8)
    R22 = sqrt(aux7 + aux9)
    aux10 = arctan2((Z2 * d2), (p * R22))
    aux11 = arctan2((Z1 * d2), (p * R21))
    aux12 = aux10 - aux11
    res = aux12
    aux10 = arctan2((Z2 * d1), (p * R12))
    aux11 = arctan2((Z1 * d1), (p * R11))
    aux12 = aux10 - aux11
    res -= aux12
    return res


def kernelxx(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v1(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res


def kernelxy(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v2(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res


def kernelxz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v3(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res


def kernelyy(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v4(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res


def kernelyz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v5(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res


def kernelzz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    nverts = prism.nverts
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        res += _integral_v6(x[k] - xp, x[(k + 1) % nverts] - xp,
                            y[k] - yp, y[(k + 1) % nverts] - yp, Z1, Z2)
    return res
