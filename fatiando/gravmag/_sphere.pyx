#cython: embedsignature=True
"""
Cython implementation of the gravity and magnetic fields of spheres.
"""
from __future__ import division
import numpy

# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

cdef inline double kernelz(double x, double y, double z, double r_cb) nogil:
    return z/r_cb
cdef inline double kernelxx(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return ((3*x**2) - r_sqr)/r_5
cdef inline double kernelxy(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return (3*x*y)/r_5
cdef inline double kernelxz(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return (3*x*z)/r_5
cdef inline double kernelyy(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return ((3*y**2) - r_sqr)/r_5
cdef inline double kernelyz(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return (3*y*z)/r_5
cdef inline double kernelzz(double x, double y, double z, double r_sqr,
                            double r_5) nogil:
    return ((3*z**2) - r_sqr)/r_5

@cython.wraparound(False)
@cython.boundscheck(False)
def tf(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double xc, double yc, double zc, double radius,
       double mx, double my, double mz, double fx, double fy, double fz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T x, y, z
    cdef DTYPE_T volume, r_sqr, r_5, v1, v2, v3, v4, v5, v6, bx, by, bz
    size = len(xp)
    volume = 4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        r_5 = r_sqr**(2.5)
        v1 = kernelxx(x, y, z, r_sqr, r_5)
        v2 = kernelxy(x, y, z, r_sqr, r_5)
        v3 = kernelxz(x, y, z, r_sqr, r_5)
        v4 = kernelyy(x, y, z, r_sqr, r_5)
        v5 = kernelyz(x, y, z, r_sqr, r_5)
        v6 = kernelzz(x, y, z, r_sqr, r_5)
        bx = (v1*mx + v2*my + v3*mz)
        by = (v2*mx + v4*my + v5*mz)
        bz = (v3*mx + v5*my + v6*mz)
        res[l] += volume*(fx*bx + fy*by + fz*bz)

@cython.wraparound(False)
@cython.boundscheck(False)
def bx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double xc, double yc, double zc, double radius,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T x, y, z
    cdef DTYPE_T volume, r_sqr, r_5, v1, v2, v3
    size = len(xp)
    volume = 4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        r_5 = r_sqr**(2.5)
        v1 = kernelxx(x, y, z, r_sqr, r_5)
        v2 = kernelxy(x, y, z, r_sqr, r_5)
        v3 = kernelxz(x, y, z, r_sqr, r_5)
        res[l] += volume*(v1*mx + v2*my + v3*mz)

@cython.wraparound(False)
@cython.boundscheck(False)
def by(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double xc, double yc, double zc, double radius,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T x, y, z
    cdef DTYPE_T volume, r_sqr, r_5, v2, v4, v5
    size = len(xp)
    volume = 4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        r_5 = r_sqr**(2.5)
        v2 = kernelxy(x, y, z, r_sqr, r_5)
        v4 = kernelyy(x, y, z, r_sqr, r_5)
        v5 = kernelyz(x, y, z, r_sqr, r_5)
        res[l] += volume*(v2*mx + v4*my + v5*mz)

@cython.wraparound(False)
@cython.boundscheck(False)
def bz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double xc, double yc, double zc, double radius,
       double mx, double my, double mz,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T x, y, z
    cdef DTYPE_T volume, r_sqr, r_5, v3, v5, v6
    size = len(xp)
    volume = 4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        r_5 = r_sqr**(2.5)
        v3 = kernelxz(x, y, z, r_sqr, r_5)
        v5 = kernelyz(x, y, z, r_sqr, r_5)
        v6 = kernelzz(x, y, z, r_sqr, r_5)
        res[l] += volume*(v3*mx + v5*my + v6*mz)

@cython.wraparound(False)
@cython.boundscheck(False)
def gz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None,
       double xc, double yc, double zc, double radius,
       double density,
       numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_cb, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_cb = (x**2 + y**2 + z**2)**(1.5)
        res[l] += mass*kernelz(x, y, z, r_cb)

@cython.wraparound(False)
@cython.boundscheck(False)
def gxx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelxx(x, y, z, r_sqr, r_sqr**(2.5))

@cython.wraparound(False)
@cython.boundscheck(False)
def gxy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelxy(x, y, z, r_sqr, r_sqr**(2.5))

@cython.wraparound(False)
@cython.boundscheck(False)
def gxz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelxz(x, y, z, r_sqr, r_sqr**(2.5))

@cython.wraparound(False)
@cython.boundscheck(False)
def gyy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelyy(x, y, z, r_sqr, r_sqr**(2.5))

@cython.wraparound(False)
@cython.boundscheck(False)
def gyz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelyz(x, y, z, r_sqr, r_sqr**(2.5))

@cython.wraparound(False)
@cython.boundscheck(False)
def gzz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None,
        double xc, double yc, double zc, double radius,
        double density,
        numpy.ndarray[DTYPE_T, ndim=1] res not None):
    cdef unsigned int l, size
    cdef DTYPE_T mass, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in range(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        res[l] += mass*kernelzz(x, y, z, r_sqr, r_sqr**(2.5))
