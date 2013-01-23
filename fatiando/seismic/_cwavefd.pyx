"""
Cython implementation of the time stepping functions for fatiando.seismic.wavefd
"""
import numpy

from libc.math cimport exp
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

__all__ = ['_apply_damping', '_boundary_conditions', '_step_elastic_sh', 
    '_step_elastic_psv_x', '_step_elastic_psv_z']


def _apply_damping(numpy.ndarray[DTYPE_T, ndim=2] array, int nx, int nz, 
    int pad, double decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    """
    cdef int i, j, in_pad
    for i in xrange(nz):
        for j in xrange(nx):
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                array[i,j] *= exp(-in_pad**2/decay**2)

def _boundary_conditions(numpy.ndarray[DTYPE_T, ndim=2] u, int nx, int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    """
    cdef unsigned int i
    for i in xrange(nx):
        u[1,i] = u[2,i]
        u[0,i] = u[1,i]
        u[-1,i] *= 0
        u[-2,i] *= 0
    for i in xrange(nz):
        u[i,0] *= 0
        u[i,1] *= 0
        u[i,-1] *= 0
        u[i,-2] *= 0

def _step_elastic_sh(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] u_t,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1,
    int x1, int x2, int z1, int z2, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] svel):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef int i, j
    for i in xrange(z1 + 2, z2 - 2):
        for j in xrange(x1 + 2, x2 - 2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (svel[i,j]**2)*(dt**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/(12.*dx**2) +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))

def _step_elastic_psv_x(
    numpy.ndarray[DTYPE_T, ndim=2] ux_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t,
    numpy.ndarray[DTYPE_T, ndim=2] ux_tm1,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t,
    int x1, int x2, int z1, int z2, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel,
    numpy.ndarray[DTYPE_T, ndim=2] svel):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(z1 + 2, z2 - 2):
        for j in xrange(x1 + 2, x2 - 2):
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

def _step_elastic_psv_z(
    numpy.ndarray[DTYPE_T, ndim=2] uz_tp1,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t,
    numpy.ndarray[DTYPE_T, ndim=2] uz_tm1,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t,
    int x1, int x2, int z1, int z2, double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel,
    numpy.ndarray[DTYPE_T, ndim=2] svel):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    cdef int i, j
    cdef double in_pad
    for i in xrange(z1 + 2, z2 - 2):
        for j in xrange(x1 + 2, x2 - 2):
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
