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

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

cdef inline double safe_atan2(double y, double x) nogil:
    cdef double res
    if y == 0:
        res = 0
    elif (y > 0) and (x < 0):
        res = atan2(y, x) - 3.1415926535897931159979634685441851615906
    elif (y < 0) and (x < 0):
        res = atan2(y, x) + 3.1415926535897931159979634685441851615906
    else:
        res = atan2(y, x)
    return res

cdef inline double safe_log(double x) nogil:
    cdef double res
    if x == 0:
        res = 0
    else:
        res = log(x)
    return res

cdef inline double kernelpot(double x, double y, double z, double r) nogil:
    return (x*y*safe_log(z + r) + y*z*safe_log(x + r) + x*z*safe_log(y + r)
            - 0.5*x**2*safe_atan2(z*y, x*r) - 0.5*y**2*safe_atan2(z*x, y*r)
            - 0.5*z**2*safe_atan2(x*y, z*r))

# Minus in gravity because Nagy et al (2000) give the formula for the gradient
# of the potential. Gravity is -grad(V).
cdef inline double kernelx(double x, double y, double z, double r) nogil:
    return -(y*safe_log(z + r) + z*safe_log(y + r) - x*safe_atan2(z*y, x*r))

cdef inline double kernely(double x, double y, double z, double r) nogil:
    return -(z*safe_log(x + r) + x*safe_log(z + r) - y*safe_atan2(x*z, y*r))

cdef inline double kernelz(double x, double y, double z, double r) nogil:
    return -(x*safe_log(y + r) + y*safe_log(x + r) - z*safe_atan2(x*y, z*r))

cdef inline double kernelxx(double x, double y, double z, double r) nogil:
    return -safe_atan2(z*y, x*r)

cdef inline double kernelxy(double x, double y, double z, double r) nogil:
    return safe_log(z + r)

cdef inline double kernelxz(double x, double y, double z, double r) nogil:
    return safe_log(y + r)

cdef inline double kernelyy(double x, double y, double z, double r) nogil:
    return -safe_atan2(z*x, y*r)

cdef inline double kernelyz(double x, double y, double z, double r) nogil:
    return safe_log(x + r)

cdef inline double kernelzz(double x, double y, double z, double r) nogil:
    return -safe_atan2(x*y, z*r)

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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    for l in range(size):
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
    cdef DTYPE_T kernel, r, dx, dy, dz, tmp1, tmp2
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    for l in range(size):
        # Evaluate the integration limits
        for k in range(2):
            dz = z[k] - zp[l]
            for j in range(2):
                dy = y[j] - yp[l]
                for i in range(2):
                    dx = x[i] - xp[l]
                    if dx == 0 and dy == 0 and dz < 0:
                        tmp1 = 0.00001*(x2 - x1)
                        tmp2 = 0.00001*(y2 - y1)
                        r = sqrt(tmp1**2 + tmp2**2 + dz**2)
                    else:
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
    cdef DTYPE_T kernel, r, dx, dy, dz, tmp1, tmp2
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    for l in range(size):
        # Evaluate the integration limits
        for k in range(2):
            dz = z[k] - zp[l]
            for j in range(2):
                dy = y[j] - yp[l]
                for i in range(2):
                    dx = x[i] - xp[l]
                    if dx == 0 and dz == 0 and dy < 0:
                        tmp1 = 0.00001*(x2 - x1)
                        tmp2 = 0.00001*(z2 - z1)
                        r = sqrt(tmp1**2 + tmp2**2 + dy**2)
                    else:
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
    for l in range(size):
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
    cdef DTYPE_T kernel, r, dx, dy, dz, tmp1, tmp2
    size = len(xp)
    x = numpy.array([x2, x1], dtype=DTYPE)
    y = numpy.array([y2, y1], dtype=DTYPE)
    z = numpy.array([z2, z1], dtype=DTYPE)
    for l in range(size):
        # Evaluate the integration limits
        for k in range(2):
            dz = z[k] - zp[l]
            for j in range(2):
                dy = y[j] - yp[l]
                for i in range(2):
                    dx = x[i] - xp[l]
                    if dy == 0 and dz == 0 and dx < 0:
                        tmp1 = 0.00001*(y2 - y1)
                        tmp2 = 0.00001*(z2 - z1)
                        r = sqrt(tmp1**2 + tmp2**2 + dx**2)
                    else:
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
    for l in range(size):
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
    for l in range(size):
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
