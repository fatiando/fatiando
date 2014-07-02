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
        unsigned int nverts, size, i, k
        DTYPE_T kernel, Z1, Z2, Z1_sqr, Z2_sqr, Xk1, Yk1, Xk2, Yk2, p, p_sqr, \
                Qk1, Qk2, Ak1, Ak2, R1k1, R1k2, R2k1, R2k2, Bk1, Bk2, E1k1, \
                E1k2, E2k1, E2k2, Ck1, Ck2
        DTYPE_T dummy = 1e-10 # Used to avoid singularities
    nverts = len(x)
    size = len(xp)
    for i in range(size):
        Z1 = z1 - zp[i]
        Z2 = z2 - zp[i]
        Z1_sqr = Z1**2
        Z2_sqr = Z2**2
        kernel = 0
        for k in range(nverts):
            Xk1 = x[k] - xp[i]
            Yk1 = y[k] - yp[i]
            Xk2 = x[(k + 1) % nverts] - xp[i]
            Yk2 = y[(k + 1) % nverts] - yp[i]
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
        res[i] += kernel*density
