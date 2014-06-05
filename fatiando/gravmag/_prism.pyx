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

from ..constants import CM, T2NT
from .. import utils


cdef inline double kernelxx(double x, double y, double z, double r):
    return -atan2(z*y, x*r)
cdef inline double kernelxy(double x, double y, double z, double r):
    return log(z + r)
cdef inline double kernelxz(double x, double y, double z, double r):
    return log(y + r)
cdef inline double kernelyy(double x, double y, double z, double r):
    return -atan2(z*x, y*r)
cdef inline double kernelyz(double x, double y, double z, double r):
    return log(x + r)
cdef inline double kernelzz(double x, double y, double z, double r):
    return -atan2(x*y, z*r)

@cython.wraparound(False)
@cython.boundscheck(False)
def magnetic_kernels(field,
    numpy.ndarray[DTYPE_T, ndim=1] xp not None,
    numpy.ndarray[DTYPE_T, ndim=1] yp not None,
    numpy.ndarray[DTYPE_T, ndim=1] zp not None,
    double x1, double x2, double y1, double y2, double z1, double z2,
    double intensity, double mx, double my, double mz,
    double fx, double fy, double fz,
    numpy.ndarray[DTYPE_T, ndim=1] res not None):
    """
    Calculate a given magnetic field 'kernel' for a single prism.

    'field' is a string that defines the field that will be calculated.

    Results are returned in 'res' (should be initialized with zeros!).
    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r, v1, v2, v3, v4, v5, v6, bx, by, bz
    size = len(xp)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for l in xrange(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x[0] = x2 - xp[l]
        x[1] = x1 - xp[l]
        y[0] = y2 - yp[l]
        y[1] = y1 - yp[l]
        z[0] = z2 - zp[l]
        z[1] = z1 - zp[l]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    if field == 'tf':
                        v1 = kernelxx(x[i], y[j], z[k], r)
                        v2 = kernelxy(x[i], y[j], z[k], r)
                        v3 = kernelxz(x[i], y[j], z[k], r)
                        v4 = kernelyy(x[i], y[j], z[k], r)
                        v5 = kernelyz(x[i], y[j], z[k], r)
                        v6 = kernelzz(x[i], y[j], z[k], r)
                        bx = (v1*mx + v2*my + v3*mz)
                        by = (v2*mx + v4*my + v5*mz)
                        bz = (v3*mx + v5*my + v6*mz)
                        kernel = fx*bx + fy*by + fz*bz
                    res[l] += ((-1.)**(i + j + k))*kernel*intensity

@cython.wraparound(False)
@cython.boundscheck(False)
def gravity_kernels(field,
    numpy.ndarray[DTYPE_T, ndim=1] xp not None,
    numpy.ndarray[DTYPE_T, ndim=1] yp not None,
    numpy.ndarray[DTYPE_T, ndim=1] zp not None,
    double x1, double x2, double y1, double y2, double z1, double z2,
    double density,
    numpy.ndarray[DTYPE_T, ndim=1] res not None):
    """
    Calculate a given gravity field 'kernel' for a single prism.

    The 'kernel' has density multiplied so that the sum can be done inplace in
    'res' and I don't need to allocate memory inside this function.

    'field' is a string that defines the field that will be calculated.

    Results are returned in 'res' (should be initialized with zeros!).
    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] x, y, z
    cdef DTYPE_T kernel, r
    size = len(xp)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for l in xrange(size):
        # First thing to do is make the computation point P the origin of
        # the coordinate system
        x[0] = x2 - xp[l]
        x[1] = x1 - xp[l]
        y[0] = y2 - yp[l]
        y[1] = y1 - yp[l]
        z[0] = z2 - zp[l]
        z[1] = z1 - zp[l]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    # Minus in gravity because Nagy et al (2000) give the
                    # formula for the gradient of the potential. Gravity is
                    # -grad(V)
                    if field == 'gz':
                        kernel = -(x[i]*log(y[j] + r)
                                   + y[j]*log(x[i] + r)
                                   - z[k]*atan2(x[i]*y[j], z[k]*r))
                    elif field == 'gzz':
                        kernel = kernelzz(x[i], y[j], z[k], r)
                    elif field == 'gxx':
                        kernel = kernelxx(x[i], y[j], z[k], r)
                    elif field == 'gxy':
                        kernel = kernelxy(x[i], y[j], z[k], r)
                    elif field == 'gxz':
                        kernel = kernelxz(x[i], y[j], z[k], r)
                    elif field == 'gyy':
                        kernel = kernelyy(x[i], y[j], z[k], r)
                    elif field == 'gyz':
                        kernel = kernelyz(x[i], y[j], z[k], r)
                    elif field == 'potential':
                        kernel = (x[i]*y[j]*log(z[k] + r)
                                  + y[j]*z[k]*log(x[i] + r)
                                  + x[i]*z[k]*log(y[j] + r)
                                  - 0.5*x[i]**2*atan2(z[k]*y[j], x[i]*r)
                                  - 0.5*y[j]**2*atan2(z[k]*x[i], y[j]*r)
                                  - 0.5*z[k]**2*atan2(x[i]*y[j], z[k]*r))
                    elif field == 'gx':
                        kernel = -(y[j]*log(z[k] + r)
                                   + z[k]*log(y[j] + r)
                                   - x[i]*atan2(z[k]*y[j], x[i]*r))
                    elif field == 'gy':
                        kernel = -(z[k]*log(x[i] + r)
                                   + x[i]*log(z[k] + r)
                                   - y[j]*atan2(x[i]*z[k], y[j]*r))
                    res[l] += ((-1.)**(i + j + k))*kernel*density

