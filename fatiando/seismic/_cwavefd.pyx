"""
Cython implementation of the time stepping functions for fatiando.seismic.wavefd
"""
import numpy

from libc.math cimport exp
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T


def step_elastic_sh(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] u_t,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1,
    int nx, int nz, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] svel,
    int pad, double decay):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (svel[i,j]**2)*(dt**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/(12.*dx**2) +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                u_tp1[i,j] *= exp(-in_pad**2/decay**2)

def step_elastic_psv_x(
    numpy.ndarray[DTYPE_T, ndim=2] ux_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t,
    numpy.ndarray[DTYPE_T, ndim=2] ux_tm1,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t,
    int nx, int nz, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel,
    numpy.ndarray[DTYPE_T, ndim=2] svel,
    int pad, double decay):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
            ux_tp1[i,j] = (2.*ux_t[i,j] - ux_tm1[i,j]
                + (pvel[i,j]**2)*(dt**2)*(
                    -ux_t[i,j + 2] + 16.*ux_t[i,j + 1] - 30.*ux_t[i,j] +
                     16.*ux_t[i,j - 1] - ux_t[i,j - 2])/(12.*dx**2)
                + (svel[i,j]**2)*(dt**2)*(
                    -ux_t[i + 2,j] + 16.*ux_t[i + 1,j] - 30.*ux_t[i,j] +
                     16.*ux_t[i - 1,j] - ux_t[i - 2,j])/(12.*dz**2)
                + (pvel[i,j]**2 - svel[i,j]**2)*(dt**2)*(
                    uz_t[i + 2,j + 2] - 8.*uz_t[i + 1,j + 2]
                    + 8.*uz_t[i - 1,j + 2] - uz_t[i - 2,j + 2]
                    - 8.*uz_t[i + 2,j + 1] + 64.*uz_t[i + 1,j + 1]
                    - 64.*uz_t[i - 1,j + 1] + 8.*uz_t[i - 2,j + 1]
                    + 8.*uz_t[i + 2,j - 1] - 64*uz_t[i + 1,j - 1]
                    + 64.*uz_t[i - 1,j - 1] - 8*uz_t[i - 2,j - 1]
                    - uz_t[i + 2,j - 2] + 8.*uz_t[i + 1,j - 2]
                    - 8.*uz_t[i - 1,j - 2] + uz_t[i - 2,j - 2])/(144*dx*dz)
                )
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                ux_tp1[i,j] *= exp(-in_pad**2/decay**2)

def step_elastic_psv_z(
    numpy.ndarray[DTYPE_T, ndim=2] uz_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t,
    numpy.ndarray[DTYPE_T, ndim=2] uz_tm1,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t,
    int nx, int nz, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel,
    numpy.ndarray[DTYPE_T, ndim=2] svel,
    int pad, double decay):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
            uz_tp1[i,j] = (2.*uz_t[i,j] - uz_tm1[i,j]
                + (pvel[i,j]**2)*(dt**2)*(
                    -uz_t[i + 2,j] + 16.*uz_t[i + 1,j] - 30.*uz_t[i,j] +
                     16.*uz_t[i - 1,j] - uz_t[i - 2,j])/(12.*dz**2)
                + (svel[i,j]**2)*(dt**2)*(
                    -uz_t[i,j + 2] + 16.*uz_t[i,j + 1] - 30.*uz_t[i,j] +
                     16.*uz_t[i,j - 1] - uz_t[i,j - 2])/(12.*dx**2)
                + (pvel[i,j]**2 - svel[i,j]**2)*(dt**2)*(
                    ux_t[i + 2,j + 2] - 8.*ux_t[i + 1,j + 2]
                    + 8.*ux_t[i - 1,j + 2] - ux_t[i - 2,j + 2]
                    - 8.*ux_t[i + 2,j + 1] + 64.*ux_t[i + 1,j + 1]
                    - 64.*ux_t[i - 1,j + 1] + 8.*ux_t[i - 2,j + 1]
                    + 8.*ux_t[i + 2,j - 1] - 64*ux_t[i + 1,j - 1]
                    + 64.*ux_t[i - 1,j - 1] - 8*ux_t[i - 2,j - 1]
                    - ux_t[i + 2,j - 2] + 8.*ux_t[i + 1,j - 2]
                    - 8.*ux_t[i - 1,j - 2] + ux_t[i - 2,j - 2])/(144*dx*dz)
                )
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                uz_tp1[i,j] *= exp(-in_pad**2/decay**2)
