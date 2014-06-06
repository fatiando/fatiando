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


cdef inline double kernelxx(double x, double y, double z, double r_sqr,
                            double r_5):
    return ((3*x**2) - r_sqr)/r_5
cdef inline double kernelxy(double x, double y, double z, double r_sqr,
                            double r_5):
    return (3*x*y)/r_5
cdef inline double kernelxz(double x, double y, double z, double r_sqr,
                            double r_5):
    return (3*x*z)/r_5
cdef inline double kernelyy(double x, double y, double z, double r_sqr,
                            double r_5):
    return ((3*y**2) - r_sqr)/r_5
cdef inline double kernelyz(double x, double y, double z, double r_sqr,
                            double r_5):
    return (3*y*z)/r_5
cdef inline double kernelzz(double x, double y, double z, double r_sqr,
                            double r_5):
    return ((3*z**2) - r_sqr)/r_5

@cython.wraparound(False)
@cython.boundscheck(False)
def magnetic_kernels(field,
    numpy.ndarray[DTYPE_T, ndim=1] xp not None,
    numpy.ndarray[DTYPE_T, ndim=1] yp not None,
    numpy.ndarray[DTYPE_T, ndim=1] zp not None,
    double xc, double yc, double zc, double radius,
    double mx, double my, double mz, double fx, double fy, double fz,
    numpy.ndarray[DTYPE_T, ndim=1] res not None):
    """
    Calculate a given magnetic field 'kernel' for a single sphere.

    'field' is a string that defines the field that will be calculated.

    Results are returned in 'res' (should be initialized with zeros!).
    """
    cdef unsigned int l, size
    cdef DTYPE_T x, y, z
    cdef DTYPE_T volume, kernel, r_sqr, r_5, v1, v2, v3, v4, v5, v6, bx, by, bz
    size = len(xp)
    volume = 4.*numpy.pi*(radius**3)/3.
    for l in xrange(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        r_5 = r_sqr**(2.5)
        if field == 'tf':
            v1 = kernelxx(x, y, z, r_sqr, r_5)
            v2 = kernelxy(x, y, z, r_sqr, r_5)
            v3 = kernelxz(x, y, z, r_sqr, r_5)
            v4 = kernelyy(x, y, z, r_sqr, r_5)
            v5 = kernelyz(x, y, z, r_sqr, r_5)
            v6 = kernelzz(x, y, z, r_sqr, r_5)
            bx = (v1*mx + v2*my + v3*mz)
            by = (v2*mx + v4*my + v5*mz)
            bz = (v3*mx + v5*my + v6*mz)
            kernel = fx*bx + fy*by + fz*bz
        elif field == 'bx':
            v1 = kernelxx(x, y, z, r_sqr, r_5)
            v2 = kernelxy(x, y, z, r_sqr, r_5)
            v3 = kernelxz(x, y, z, r_sqr, r_5)
            kernel = (v1*mx + v2*my + v3*mz)
        elif field == 'by':
            v2 = kernelxy(x, y, z, r_sqr, r_5)
            v4 = kernelyy(x, y, z, r_sqr, r_5)
            v5 = kernelyz(x, y, z, r_sqr, r_5)
            kernel = (v2*mx + v4*my + v5*mz)
        elif field == 'bz':
            v3 = kernelxz(x, y, z, r_sqr, r_5)
            v5 = kernelyz(x, y, z, r_sqr, r_5)
            v6 = kernelzz(x, y, z, r_sqr, r_5)
            kernel = (v3*mx + v5*my + v6*mz)
        res[l] += volume*kernel

@cython.wraparound(False)
@cython.boundscheck(False)
def gravity_kernels(field,
    numpy.ndarray[DTYPE_T, ndim=1] xp not None,
    numpy.ndarray[DTYPE_T, ndim=1] yp not None,
    numpy.ndarray[DTYPE_T, ndim=1] zp not None,
    double xc, double yc, double zc, double radius,
    double density,
    numpy.ndarray[DTYPE_T, ndim=1] res not None):
    """
    Calculate a given gravity field 'kernel' for a single sphere.

    The 'kernel' has density multiplied so that the sum can be done inplace in
    'res' and I don't need to allocate memory inside this function.

    'field' is a string that defines the field that will be calculated.

    Results are returned in 'res' (should be initialized with zeros!).
    """
    cdef unsigned int l, size
    cdef DTYPE_T mass, kernel, r_sqr, x, y, z
    size = len(xp)
    mass = density*4.*numpy.pi*(radius**3)/3.
    for l in xrange(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x = xc - xp[l]
        y = yc - yp[l]
        z = zc - zp[l]
        r_sqr = x**2 + y**2 + z**2
        if field == 'gz':
            kernel = z/r_sqr**(1.5)
        elif field == 'gzz':
            kernel = kernelzz(x, y, z, r_sqr, r_sqr**(2.5))
        elif field == 'gxx':
            kernel = kernelxx(x, y, z, r_sqr, r_sqr**(2.5))
        elif field == 'gxy':
            kernel = kernelxy(x, y, z, r_sqr, r_sqr**(2.5))
        elif field == 'gxz':
            kernel = kernelxz(x, y, z, r_sqr, r_sqr**(2.5))
        elif field == 'gyy':
            kernel = kernelyy(x, y, z, r_sqr, r_sqr**(2.5))
        elif field == 'gyz':
            kernel = kernelyz(x, y, z, r_sqr, r_sqr**(2.5))
        res[l] += kernel*mass
