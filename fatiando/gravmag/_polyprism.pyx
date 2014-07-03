#cython: embedsignature=True
"""
This is a Cython implementation of the potential fields of a polygonal prism.
A pure python implementation is in _polyprism_numpy.py
"""
import numpy

from libc.math cimport log, atan2, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

cdef inline double kernelz(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, Z1_sqr, Z2_sqr, Xk1, Yk1, Xk2, Yk2, p, p_sqr, \
                Qk1, Qk2, Ak1, Ak2, R1k1, R1k2, R2k1, R2k2, Bk1, Bk2, E1k1, \
                E1k2, E2k1, E2k2, Ck1, Ck2
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1**2
    Z2_sqr = Z2**2
    for k in range(nverts):
        Xk1 = x[k] - xp
        Yk1 = y[k] - yp
        Xk2 = x[(k + 1) % nverts] - xp
        Yk2 = y[(k + 1) % nverts] - yp
        p = Xk1*Yk2 - Xk2*Yk1
        p_sqr = p**2
        Qk1 = (Yk2 - Yk1)*Yk1 + (Xk2 - Xk1)*Xk1
        Qk2 = (Yk2 - Yk1)*Yk2 + (Xk2 - Xk1)*Xk2
        Ak1 = Xk1**2 + Yk1**2
        Ak2 = Xk2**2 + Yk2**2
        R1k1 = sqrt(Ak1 + Z1_sqr)
        R1k2 = sqrt(Ak2 + Z1_sqr)
        R2k1 = sqrt(Ak1 + Z2_sqr)
        R2k2 = sqrt(Ak2 + Z2_sqr)
        Ak1 = sqrt(Ak1)
        Ak2 = sqrt(Ak2)
        Bk1 = sqrt(Qk1**2 + p_sqr)
        Bk2 = sqrt(Qk2**2 + p_sqr)
        E1k1 = R1k1*Bk1
        E1k2 = R1k2*Bk2
        E2k1 = R2k1*Bk1
        E2k2 = R2k2*Bk2
        kernel += (Z2 - Z1)*(atan2(Qk2, p) - atan2(Qk1, p))
        kernel += Z2*(atan2(Z2*Qk1, R2k1*p) - atan2(Z2*Qk2, R2k2*p))
        kernel += Z1*(atan2(Z1*Qk2, R1k2*p) - atan2(Z1*Qk1, R1k1*p))
        Ck1 = Qk1*Ak1
        Ck2 = Qk2*Ak2
        # dummy helps prevent zero division errors
        kernel += 0.5*p*(Ak1/(Bk1 + dummy))*(
            log((E1k1 - Ck1)/(E1k1 + Ck1 + dummy) + dummy) -
            log((E2k1 - Ck1)/(E2k1 + Ck1 + dummy) + dummy))
        kernel += 0.5*p*(Ak2/(Bk2 + dummy))*(
            log((E2k2 - Ck2)/(E2k2 + Ck2 + dummy) + dummy) -
            log((E1k2 - Ck2)/(E1k2 + Ck2 + dummy) + dummy))
    return kernel

cdef inline double kernelxx(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux3, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, aux13, \
                aux14, aux15, aux16, n, g, p, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        n = (aux0/aux1)
        g = X1 - (Y1*n)
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux3 = (X1*Y2) - (X2*Y1)
        p = ((aux3/aux2)) + dummy
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = atan2((Z2*d2), (p*R22))
        aux11 = atan2((Z1*d2), (p*R21))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d2))
        aux14 = ((p*aux12)/d2)
        res = (g*Y2*aux13) + (n*aux14)
        aux10 = atan2((Z2*d1), (p*R12))
        aux11 = atan2((Z1*d1), (p*R11))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d1))
        aux14 = ((p*aux12)/d1)
        res -= (g*Y1*aux13) + (n*aux14)
        aux10 = log(((Z2 + R22) + dummy))
        aux11 = log(((Z1 + R21) + dummy))
        aux12 = log(((Z2 + R12) + dummy))
        aux13 = log(((Z1 + R11) + dummy))
        aux14 = aux10 - aux11
        aux15 = aux12 - aux13
        res += (n*(aux15 - aux14))
        aux0 = (1.0/(1.0 + (n*n)))
        res *= -aux0
        kernel += res
    return kernel

cdef inline double kernelxy(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux3, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, aux13, \
                aux14, aux15, aux16, n, g, p, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        n = (aux0/aux1)
        g = X1 - (Y1*n)
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux3 = (X1*Y2) - (X2*Y1)
        p = ((aux3/aux2)) + dummy
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = atan2((Z2*d2), (p*R22))
        aux11 = atan2((Z1*d2), (p*R21))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d2))
        aux14 = ((p*aux12)/d2)
        res = (((g*g) + (g*n*Y2))*aux13) - aux14
        aux10 = atan2((Z2*d1), (p*R12))
        aux11 = atan2((Z1*d1), (p*R11))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d1))
        aux14 = ((p*aux12)/d1)
        res -= (((g*g) + (g*n*Y1))*aux13) - aux14
        aux10 = log(((Z2 + R22) + dummy))
        aux11 = log(((Z1 + R21) + dummy))
        aux12 = log(((Z2 + R12) + dummy))
        aux13 = log(((Z1 + R11) + dummy))
        aux14 = aux10 - aux11
        aux15 = aux12 - aux13
        res += (aux14 - aux15)
        aux0 = (1.0/(1.0 + (n*n)))
        res *= aux0
        kernel += res
    return kernel

cdef inline double kernelxz(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux3, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, aux13, \
                aux14, aux15, aux16, n, g, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        n = (aux0/aux1)
        g = X1 - (Y1*n)
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = log((((R11 - d1)/(R11 + d1)) + dummy))
        aux11 = log((((R12 - d1)/(R12 + d1)) + dummy))
        aux12 = log((((R21 - d2)/(R21 + d2)) + dummy))
        aux13 = log((((R22 - d2)/(R22 + d2)) + dummy))
        aux14 = (1.0/(2*d1))
        aux15 = (1.0/(2*d2))
        aux16 = aux15*(aux13 - aux12)
        res = (Y2*(1.0 + (n*n)) + g*n)*aux16
        aux16 = aux14*(aux11 - aux10)
        res -= (Y1*(1.0 + (n*n)) + g*n)*aux16
        aux0 = (1.0/(1.0 + (n*n)))
        res *= -aux0
        kernel += res
    return kernel

cdef inline double kernelyy(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux3, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, aux13, \
                aux14, aux15, p, m, c, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        m = (aux1/aux0)
        c = Y1 - (X1*m)
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux3 = (X1*Y2) - (X2*Y1)
        p = ((aux3/aux2)) + dummy
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = atan2((Z2*d2), (p*R22))
        aux11 = atan2((Z1*d2), (p*R21))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d2))
        aux14 = ((p*aux12)/d2)
        res = (c*X2*aux13) + (m*aux14)
        aux10 = atan2((Z2*d1), (p*R12))
        aux11 = atan2((Z1*d1), (p*R11))
        aux12 = aux10 - aux11
        aux13 = (aux12/(p*d1))
        aux14 = ((p*aux12)/d1)
        res -= (c*X1*aux13) + (m*aux14)
        aux10 = log(((Z2 + R22) + dummy))
        aux11 = log(((Z1 + R21) + dummy))
        aux12 = log(((Z2 + R12) + dummy))
        aux13 = log(((Z1 + R11) + dummy))
        aux14 = aux10 - aux11
        aux15 = aux12 - aux13
        res += (m*(aux15 - aux14))
        aux1 = (1.0/(1.0 + (m*m)))
        res *= aux1
        kernel += res
    return kernel

cdef inline double kernelyz(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, aux13, \
                aux14, aux15, aux16, m, c, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        m = (aux1/aux0)
        c = Y1 - (X1*m)
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = log((((R11 - d1)/(R11 + d1)) + dummy))
        aux11 = log((((R12 - d1)/(R12 + d1)) + dummy))
        aux12 = log((((R21 - d2)/(R21 + d2)) + dummy))
        aux13 = log((((R22 - d2)/(R22 + d2)) + dummy))
        aux14 = (1.0/(2*d1))
        aux15 = (1.0/(2*d2))
        aux16 = aux15*(aux13 - aux12)
        res = (X2*(1.0 + (m*m)) + c*m)*aux16
        aux16 = aux14*(aux11 - aux10)
        res -= (X1*(1.0 + (m*m)) + c*m)*aux16
        aux1 = (1.0/(1.0 + (m*m)))
        res *= aux1
        kernel += res
    return kernel

cdef inline double kernelzz(
    double xp, double yp, double zp, numpy.ndarray[DTYPE_T, ndim=1] x,
    numpy.ndarray[DTYPE_T, ndim=1] y, double z1, double z2,
    unsigned int nverts):
    cdef:
        unsigned int k
        DTYPE_T kernel, Z1, Z2, X1, X2, Y1, Y2, aux0, aux1, aux2, aux3, aux4, \
                aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12, p, d1, d2, \
                R11, R12, R21, R22, res
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    kernel = 0
    Z1 = z1 - zp
    Z2 = z2 - zp
    for k in range(nverts):
        X1 = x[k] - xp
        X2 = x[(k + 1) % nverts] - xp
        Y1 = y[k] - yp
        Y2 = y[(k + 1) % nverts] - yp
        aux0 = X2 - X1 + dummy
        aux1 = Y2 - Y1 + dummy
        aux2 = sqrt((aux0*aux0) + (aux1*aux1))
        aux3 = (X1*Y2) - (X2*Y1)
        p = ((aux3/aux2)) + dummy
        aux4 = (aux0*X1) + (aux1*Y1)
        aux5 = (aux0*X2) + (aux1*Y2)
        d1 = ((aux4/aux2)) + dummy
        d2 = ((aux5/aux2)) + dummy
        aux6 = (X1*X1) + (Y1*Y1)
        aux7 = (X2*X2) + (Y2*Y2)
        aux8 = Z1*Z1
        aux9 = Z2*Z2
        R11 = sqrt(aux6 + aux8)
        R12 = sqrt(aux6 + aux9)
        R21 = sqrt(aux7 + aux8)
        R22 = sqrt(aux7 + aux9)
        aux10 = atan2((Z2*d2), (p*R22))
        aux11 = atan2((Z1*d2), (p*R21))
        aux12 = aux10 - aux11
        res = aux12
        aux10 = atan2((Z2*d1), (p*R12))
        aux11 = atan2((Z1*d1), (p*R11))
        aux12 = aux10 - aux11
        res -= aux12
        kernel += res
    return kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def gz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       numpy.ndarray[DTYPE_T, ndim=1] x not None,
       numpy.ndarray[DTYPE_T, ndim=1] y not None,
       double z1, double z2, double density,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelxx(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelxy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelxz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gyy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelyy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gyz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelyz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gzz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        numpy.ndarray[DTYPE_T, ndim=1] x not None,
        numpy.ndarray[DTYPE_T, ndim=1] y not None,
        double z1, double z2, double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T kernel
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        kernel = kernelzz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def tf(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       numpy.ndarray[DTYPE_T, ndim=1] x not None,
       numpy.ndarray[DTYPE_T, ndim=1] y not None,
       double z1, double z2,
       double mx, double my, double mz, double fx, double fy, double fz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T v1, v2, v3, v4, v5, v6, bx, by, bz
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        v1 = kernelxx(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v2 = kernelxy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v3 = kernelxz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v4 = kernelyy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v5 = kernelyz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v6 = kernelzz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        bx = (v1*mx + v2*my + v3*mz)
        by = (v2*mx + v4*my + v5*mz)
        bz = (v3*mx + v5*my + v6*mz)
        res[i] += fx*bx + fy*by + fz*bz

@cython.wraparound(False)
@cython.boundscheck(False)
def bx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       numpy.ndarray[DTYPE_T, ndim=1] x not None,
       numpy.ndarray[DTYPE_T, ndim=1] y not None,
       double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T v1, v2, v3
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        v1 = kernelxx(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v2 = kernelxy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v3 = kernelxz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += (v1*mx + v2*my + v3*mz)

@cython.wraparound(False)
@cython.boundscheck(False)
def by(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       numpy.ndarray[DTYPE_T, ndim=1] x not None,
       numpy.ndarray[DTYPE_T, ndim=1] y not None,
       double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T v2, v4, v5
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        v2 = kernelxy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v4 = kernelyy(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v5 = kernelyz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += (v2*mx + v4*my + v5*mz)

@cython.wraparound(False)
@cython.boundscheck(False)
def bz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       numpy.ndarray[DTYPE_T, ndim=1] x not None,
       numpy.ndarray[DTYPE_T, ndim=1] y not None,
       double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef:
        unsigned int nverts, size, i
        DTYPE_T v3, v5, v6
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        v3 = kernelxz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v5 = kernelyz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        v6 = kernelzz(xp[i], yp[i], zp[i], x, y, z1, z2, nverts)
        res[i] += (v3*mx + v5*my + v6*mz)
