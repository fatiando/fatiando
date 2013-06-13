"""
Cython implementation of the time stepping functions for fatiando.seismic.wavefd
"""
import numpy

from libc.math cimport exp, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

__all__ = ['_apply_damping', '_boundary_conditions', '_step_elastic_sh',
    '_step_elastic_psv_x', '_step_elastic_psv_z',
    '_nonreflexive_sh_boundary_conditions',
    '_nonreflexive_psv_boundary_conditions']


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
            array[i,j] *= exp(-((decay*(pad - j))**2))
    # Damping on the right
    for i in xrange(nz):
        for j in xrange(nx - pad, nx):
            array[i,j] *= exp(-((decay*(j - nx + pad))**2))
    # Damping on the bottom
    for i in xrange(nz - pad, nz):
        for j in xrange(nx):
            array[i,j] *= exp(-((decay*(i - nz + pad))**2))

@cython.boundscheck(False)
@cython.wraparound(False)
def _boundary_conditions(numpy.ndarray[DTYPE_T, ndim=2] u not None,
    unsigned int nx, unsigned int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    """
    cdef unsigned int i
    for i in xrange(nx):
        u[2, i] = u[3, i]
        u[1, i] = u[2, i]
        u[0, i] = u[1, i]
        u[nz - 1, i] *= 0
        u[nz - 2, i] *= 0
        u[nz - 3, i] *= 0
    for i in xrange(nz):
        u[i, 0] *= 0
        u[i, 1] *= 0
        u[i, 2] *= 0
        u[i, nx - 1] *= 0
        u[i, nx - 2] *= 0
        u[i, nx - 3] *= 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _nonreflexive_psv_boundary_conditions(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_t not None,
    unsigned int nx, unsigned int nz,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] mu not None,
    numpy.ndarray[DTYPE_T, ndim=2] lamb not None,
    numpy.ndarray[DTYPE_T, ndim=2] dens not None):
    """
    Apply nonreflexive boundary contitions to elastic P-SV waves.
    """
    cdef unsigned int i, j
    # Left
    for i in xrange(nz):
        for j in xrange(1):
            u_tp1[i,j] = u_t[i,j] + \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    u_t[i,j+1] - u_t[i,j])/dx
    # Right
    for i in xrange(nz):
        for j in xrange(nx - 1, nx):
            u_tp1[i,j] = u_t[i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    u_t[i,j] - u_t[i,j-1])/dx
    # Bottom
    for i in xrange(nz - 1, nz):
        for j in xrange(nx):
            u_tp1[i,j] = u_t[i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    u_t[i,j] - u_t[i-1,j])/dz
    # Top
    for j in xrange(nx):
        u_tp1[0,j] = u_tp1[1,j]

@cython.boundscheck(False)
@cython.wraparound(False)
def _nonreflexive_sh_boundary_conditions(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_t not None,
    unsigned int nx, unsigned int nz,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] mu not None,
    numpy.ndarray[DTYPE_T, ndim=2] dens not None):
    """
    Apply nonreflexive boundary contitions to elastic SH waves.
    """
    cdef unsigned int i, j
    # Left
    for i in xrange(nz):
        for j in xrange(3):
            u_tp1[i,j] = u_t[i,j] + dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j+1] - u_t[i,j])/dx
    # Right
    for i in xrange(nz):
        for j in xrange(nx - 3, nx):
            u_tp1[i,j] = u_t[i,j] - dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j] - u_t[i,j-1])/dx
    # Bottom
    for i in xrange(nz - 3, nz):
        for j in xrange(nx):
            u_tp1[i,j] = u_t[i,j] - dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j] - u_t[i-1,j])/dz
    # Top
    for j in xrange(nx):
        u_tp1[2,j] = u_tp1[3,j]
        u_tp1[1,j] = u_tp1[2,j]
        u_tp1[0,j] = u_tp1[1,j]

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_sh(
    numpy.ndarray[DTYPE_T, ndim=2] u_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] u_tm1 not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] mu not None,
    numpy.ndarray[DTYPE_T, ndim=2] dens not None):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef unsigned int i, j
    cdef DTYPE_T dt2, dx2, dz2
    dt2 = dt**2
    dx2 = dx**2
    dz2 = dz**2
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            u_tp1[i,j] = 2*u_t[i,j] - u_tm1[i,j] + (dt2/dens[i,j])*(
                (1.125/dz2)*(
                    0.5*(mu[i+1,j] + mu[i,j])*(
                        1.125*(u_t[i+1,j] - u_t[i,j])
                        - (u_t[i+2,j] - u_t[i-1,j])/24.)
                    - 0.5*(mu[i,j] + mu[i-1,j])*(
                        1.125*(u_t[i,j] - u_t[i-1,j])
                        - (u_t[i+1,j] - u_t[i-2,j])/24.))
                - (1./(24.*dz2))*(
                    0.5*(mu[i+2,j] + mu[i+1,j])*(
                        1.125*(u_t[i+2,j] - u_t[i+1,j])
                        - (u_t[i+3,j] - u_t[i,j])/24.)
                    - 0.5*(mu[i-1,j] + mu[i-2,j])*(
                        1.125*(u_t[i-1,j] - u_t[i-2,j])
                        - (u_t[i,j] - u_t[i-3,j])/24.))
                + (1.125/dx2)*(
                    0.5*(mu[i,j+1] + mu[i,j])*(
                        1.125*(u_t[i,j+1] - u_t[i,j])
                        - (u_t[i,j+2] - u_t[i,j-1])/24.)
                    - 0.5*(mu[i,j] + mu[i,j-1])*(
                        1.125*(u_t[i,j] - u_t[i,j-1])
                        - (u_t[i,j+1] - u_t[i,j-2])/24.))
                - (1./(24.*dx2))*(
                    0.5*(mu[i,j+2] + mu[i,j+1])*(
                        1.125*(u_t[i,j+2] - u_t[i,j+1])
                        - (u_t[i,j+3] - u_t[i,j])/24.)
                    - 0.5*(mu[i,j-1] + mu[i,j-2])*(
                        1.125*(u_t[i,j-1] - u_t[i,j-2])
                        - (u_t[i,j] - u_t[i,j-3])/24.)))

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_psv_x(
    numpy.ndarray[DTYPE_T, ndim=2] ux_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux_tm1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] mu not None,
    numpy.ndarray[DTYPE_T, ndim=2] lamb not None,
    numpy.ndarray[DTYPE_T, ndim=2] dens not None):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    cdef unsigned int i, j
    cdef DTYPE_T dt2, dx2, dz2, tauxx_p, tauxx_m, tauxz_p, tauxz_m, l, m
    dt2 = dt**2
    dx2 = dx**2
    dz2 = dz**2
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            l = 0.5*(lamb[i,j+1] + lamb[i,j])
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxx_p = (l + 2*m)*(ux_t[i,j+1] - ux_t[i,j])/dx + \
                (l/dz)*(
                    0.5*(uz[i+1,j+1] + uz[i,j]) - 0.5*(uz[i-1,j+1] + uz[i,j]))
            l = 0.5*(lamb[i,j-1] + lamb[i,j])
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxx_m = (l + 2*m)*(ux_t[i,j] - ux_t[i,j-1])/dx + \
                (l/dz)*(
                    0.5*(uz[i+1,j-1] + uz[i,j]) - 0.5*(uz[i-1,j-1] + uz[i,j]))
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauxz_p = 0.5*(mu[i+1,j] + mu[i,j])*(
                (ux_t[i+1,j] - ux_t[i,j])/dz +
                (0.5*(uz[i+1,j+1] + uz[i,j]) - 0.5*(uz[i+1,j-1] + uz[i,j]))/dx)
            tauxz_m = 0.5*(mu[i-1,j] + mu[i,j])*(
                (ux_t[i,j] - ux_t[i-1,j])/dz +
                (0.5*(uz[i-1,j+1] + uz[i,j]) - 0.5*(uz[i-1,j-1] + uz[i,j]))/dx)
            ux_tp1[i,j] = 2*ux_t[i,j] - ux_tm1[i,j] + (dt2/dens[i,j])*(
                (tauxx_p - tauxx_m)/dx + (tauxz_p - tauxz_m)/dz)

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_psv_z(
    numpy.ndarray[DTYPE_T, ndim=2] uz_tp1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz_t not None,
    numpy.ndarray[DTYPE_T, ndim=2] uz_tm1 not None,
    numpy.ndarray[DTYPE_T, ndim=2] ux not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    numpy.ndarray[DTYPE_T, ndim=2] mu not None,
    numpy.ndarray[DTYPE_T, ndim=2] lamb not None,
    numpy.ndarray[DTYPE_T, ndim=2] dens not None):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    cdef unsigned int i, j
    cdef DTYPE_T dt2, dx2, dz2, tauzz_p, tauzz_m, tauxz_p, tauxz_m, l, m
    dt2 = dt**2
    dx2 = dx**2
    dz2 = dz**2
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            l = 0.5*(lamb[i+1,j] + lamb[i,j])
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauzz_p = (l + 2*m)*(uz_t[i+1,j] - uz_t[i,j])/dz + \
                (l/dx)*(
                    0.5*(ux[i+1,j+1] + ux[i,j]) - 0.5*(ux[i+1,j-1] + ux[i,j]))
            l = 0.5*(lamb[i-1,j] + lamb[i,j])
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauzz_m = (l + 2*m)*(uz_t[i,j] - uz_t[i-1,j])/dz + \
                (l/dx)*(
                    0.5*(ux[i-1,j+1] + ux[i,j]) - 0.5*(ux[i-1,j-1] + ux[i,j]))
            tauxz_p = 0.5*(mu[i,j+1] + mu[i,j])*(
                (uz_t[i,j+1] - uz_t[i,j])/dx +
                (0.5*(ux[i+1,j+1] + ux[i,j]) - 0.5*(ux[i-1,j+1] + ux[i,j]))/dz)
            tauxz_m = 0.5*(mu[i,j-1] + mu[i,j])*(
                (uz_t[i,j] - uz_t[i,j-1])/dx +
                (0.5*(ux[i+1,j-1] + ux[i,j]) - 0.5*(ux[i-1,j-1] + ux[i,j]))/dz)
            uz_tp1[i,j] = 2*uz_t[i,j] - uz_tm1[i,j] + (dt2/dens[i,j])*(
                (tauzz_p - tauzz_m)/dz + (tauxz_p - tauxz_m)/dx)
