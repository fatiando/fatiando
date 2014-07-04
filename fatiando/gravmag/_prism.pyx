#cython: embedsignature=True
"""
Cython implementation of the gravity and magnetic fields of right rectangular
prisms.
"""
import numpy

from libc.math cimport log, atan2, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython
cimport openmp
from cython.parallel cimport prange, parallel

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T


cdef inline double kernelpot(double x, double y, double z, double r) nogil:
    return (x*y*log(z + r) + y*z*log(x + r) + x*z*log(y + r)
            - 0.5*x**2*atan2(z*y, x*r) - 0.5*y**2*atan2(z*x, y*r)
            - 0.5*z**2*atan2(x*y, z*r))

# Minus in gravity because Nagy et al (2000) give the formula for the gradient
# of the potential. Gravity is -grad(V).
cdef inline double kernelx(double x, double y, double z, double r) nogil:
    return -(y*log(z + r) + z*log(y + r) - x*atan2(z*y, x*r))

cdef inline double kernely(double x, double y, double z, double r) nogil:
    return -(z*log(x + r) + x*log(z + r) - y*atan2(x*z, y*r))

cdef inline double kernelz(double x, double y, double z, double r) nogil:
    return -(x*log(y + r) + y*log(x + r) - z*atan2(x*y, z*r))

cdef inline double kernelxx(double x, double y, double z, double r) nogil:
    return -atan2(z*y, x*r)

cdef inline double kernelxy(double x, double y, double z, double r) nogil:
    return log(z + r)

cdef inline double kernelxz(double x, double y, double z, double r) nogil:
    return log(y + r)

cdef inline double kernelyy(double x, double y, double z, double r) nogil:
    return -atan2(z*x, y*r)

cdef inline double kernelyz(double x, double y, double z, double r) nogil:
    return log(x + r)

cdef inline double kernelzz(double x, double y, double z, double r) nogil:
    return -atan2(x*y, z*r)

@cython.wraparound(False)
@cython.boundscheck(False)
def tf(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double mx, double my, double mz, double fx, double fy, double fz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, v1, v2, v3, v4, v5, v6, bx, by, bz, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        v1 = kernelxx(dx, dy, dz, r)
                        v2 = kernelxy(dx, dy, dz, r)
                        v3 = kernelxz(dx, dy, dz, r)
                        v4 = kernelyy(dx, dy, dz, r)
                        v5 = kernelyz(dx, dy, dz, r)
                        v6 = kernelzz(dx, dy, dz, r)
                        bx = (v1*mx + v2*my + v3*mz)
                        by = (v2*mx + v4*my + v5*mz)
                        bz = (v3*mx + v5*my + v6*mz)
                        kernel = fx*bx + fy*by + fz*bz
                        res[l] += ((-1.)**(i + j + k))*kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def bx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, v1, v2, v3, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        v1 = kernelxx(dx, dy, dz, r)
                        v2 = kernelxy(dx, dy, dz, r)
                        v3 = kernelxz(dx, dy, dz, r)
                        kernel = (v1*mx + v2*my + v3*mz)
                        res[l] += ((-1.)**(i + j + k))*kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def by(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, v2, v4, v5, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        v2 = kernelxy(dx, dy, dz, r)
                        v4 = kernelyy(dx, dy, dz, r)
                        v5 = kernelyz(dx, dy, dz, r)
                        kernel = (v2*mx + v4*my + v5*mz)
                        res[l] += ((-1.)**(i + j + k))*kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def bz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, v3, v5, v6, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        v3 = kernelxz(dx, dy, dz, r)
                        v5 = kernelyz(dx, dy, dz, r)
                        v6 = kernelzz(dx, dy, dz, r)
                        kernel = (v3*mx + v5*my + v6*mz)
                        res[l] += ((-1.)**(i + j + k))*kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def gx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double density,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelx(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double density,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernely(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double x1, double x2, double y1, double y2, double z1, double z2,
       double density,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelz(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelxx(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelxy(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gxz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelxz(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gyy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelyy(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gyz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelyz(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def gzz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double x1, double x2, double y1, double y2, double z1, double z2,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelzz(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density

@cython.wraparound(False)
@cython.boundscheck(False)
def potential(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
              numpy.ndarray[DTYPE_T, ndim=1] yp not None,
              numpy.ndarray[DTYPE_T, ndim=1] zp not None,
              double x1, double x2, double y1, double y2, double z1, double z2,
              double density,
              numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, dx, dy, dz
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    with nogil:
        for l in prange(size):
            # Evaluate the integration limits
            for k in range(2):
                dz = z[k] - zp[l]
                for j in range(2):
                    dy = y[j] - yp[l]
                    for i in range(2):
                        dx = x[i] - xp[l]
                        r = sqrt(dx**2 + dy**2 + dz**2)
                        kernel = kernelpot(dx, dy, dz, r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
