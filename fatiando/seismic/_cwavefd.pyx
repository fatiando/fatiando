"""
Cython implementation of the time stepping functions for fatiando.seismic.wavefd
"""
import numpy

from libc.math cimport exp
# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

__all__ = ['_apply_damping', '_boundary_conditions', '_step_elastic_sh',
    '_step_elastic_psv_x', '_step_elastic_psv_z', '_step_scalar']


@cython.boundscheck(False)
@cython.wraparound(False)
def _apply_damping(numpy.ndarray[DTYPE_T, ndim=2] array not None,
    unsigned int nx, unsigned int nz, unsigned int pad, double decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    """
    cdef unsigned int i, j
    # Damping on the left
    for i in xrange(nz):
        for j in xrange(pad):
            array[i,j] *= exp(-(pad - j)**2/decay**2)
    # Damping on the right
    for i in xrange(nz):
        for j in xrange(nx - pad, nx):
            array[i,j] *= exp(-(j - (nx - pad) + 1)**2/decay**2)
    # Damping on the bottom
    for i in xrange(nz - pad, nz):
        for j in xrange(pad, nx - pad):
            array[i,j] *= exp(-(i - (nz - pad) + 1)**2/decay**2)

@cython.boundscheck(False)
@cython.wraparound(False)
def _boundary_conditions(numpy.ndarray[DTYPE_T, ndim=2] u not None,
    unsigned int nx, unsigned int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    """
    cdef unsigned int i
    for i in xrange(nx):
        u[1, i] = u[2, i]
        u[0, i] = u[1, i]
        u[nz - 1, i] *= 0
        u[nz - 2, i] *= 0
    for i in xrange(nz):
        u[i, 0] *= 0
        u[i, 1] *= 0
        u[i, nx - 1] *= 0
        u[i, nx - 2] *= 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_scalar(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1 not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] vel not None):
    """
    Perform a single time step in the Finite Difference solution for scalar
    waves.
    """
    cdef unsigned int i, j
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (vel[i,j]**2)*(dt**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/(12.*dx**2) +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_sh(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1 not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] svel not None):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef unsigned int i, j
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (svel[i,j]**2)*(dt**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/(12.*dx**2) +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_psv_x(
    numpy.ndarray[DTYPE_T, ndim=2] ux_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux_tm1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel not None,
    numpy.ndarray[DTYPE_T, ndim=2] svel not None):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    cdef unsigned int i, j
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
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

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_psv_z(
    numpy.ndarray[DTYPE_T, ndim=2] uz_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz_tm1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] pvel not None,
    numpy.ndarray[DTYPE_T, ndim=2] svel not None):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    cdef unsigned int i, j
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
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
