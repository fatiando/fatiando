"""
Cython implementation of the time stepping functions for
fatiando.seismic.wavefd
"""
import numpy

from libc.math cimport exp, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t double

__all__ = ['_apply_damping', '_boundary_conditions', '_step_elastic_sh',
    '_step_elastic_psv_x', '_step_elastic_psv_z',
    '_nonreflexive_sh_boundary_conditions',
    '_nonreflexive_psv_boundary_conditions']


@cython.boundscheck(False)
@cython.wraparound(False)
def _apply_damping(double[:,::1] array not None,
    unsigned int nx, unsigned int nz, unsigned int pad, double decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    """
    cdef:
        unsigned int i, j
    # Damping on the left
    for i in range(nz):
        for j in range(pad):
            array[i,j] *= exp(-((decay*(pad - j))**2))
    # Damping on the right
    for i in range(nz):
        for j in range(nx - pad, nx):
            array[i,j] *= exp(-((decay*(j - nx + pad))**2))
    # Damping on the bottom
    for i in range(nz - pad, nz):
        for j in range(nx):
            array[i,j] *= exp(-((decay*(i - nz + pad))**2))

@cython.boundscheck(False)
@cython.wraparound(False)
def _boundary_conditions(double[:,::1] u not None,
    unsigned int nx, unsigned int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    """
    cdef:
        unsigned int i
    for i in range(nx):
        u[2, i] = u[3, i]
        u[1, i] = u[2, i]
        u[0, i] = u[1, i]
        u[nz - 1, i] *= 0
        u[nz - 2, i] *= 0
        u[nz - 3, i] *= 0
    for i in range(nz):
        u[i, 0] *= 0
        u[i, 1] *= 0
        u[i, 2] *= 0
        u[i, nx - 1] *= 0
        u[i, nx - 2] *= 0
        u[i, nx - 3] *= 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _nonreflexive_psv_boundary_conditions(
    double[:,:,::1] ux not None,
    double[:,:,::1] uz not None,
    int tp1, int t, int tm1,
    unsigned int nx, unsigned int nz,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] lamb not None,
    double[:,::1] dens not None):
    """
    Apply nonreflexive boundary contitions to elastic P-SV waves.
    """
    cdef:
        unsigned int i, j
    # Left
    for i in range(nz):
        for j in range(1):
            ux[tp1,i,j] = ux[t,i,j] + \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    ux[t,i,j+1] - ux[t,i,j])/dx
            uz[tp1,i,j] = uz[t,i,j] + \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    uz[t,i,j+1] - uz[t,i,j])/dx
    # Right
    for i in range(nz):
        for j in range(nx - 1, nx):
            ux[tp1,i,j] = ux[t,i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    ux[t,i,j] - ux[t,i,j-1])/dx
            uz[tp1,i,j] = uz[t,i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    uz[t,i,j] - uz[t,i,j-1])/dx
    # Bottom
    for i in range(nz - 1, nz):
        for j in range(nx):
            ux[tp1,i,j] = ux[t,i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    ux[t,i,j] - ux[t,i-1,j])/dz
            uz[tp1,i,j] = uz[t,i,j] - \
                dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
                    uz[t,i,j] - uz[t,i-1,j])/dz
    # Top
    for j in range(1, nx - 1):
        ux[tp1,0,j] = ux[tp1,1,j] + (0.5*dz/dx)*(uz[tp1,1,j+1] - uz[tp1,1,j-1])
        uz[tp1,0,j] = uz[tp1,1,j] + (0.5*dz/dx)*(lamb[1,j]/(lamb[1,j] + 2*mu[1,j]))*(
                ux[tp1,1,j+1] - ux[tp1,1,j-1])

@cython.boundscheck(False)
@cython.wraparound(False)
def _nonreflexive_sh_boundary_conditions(
    double[:,::1] u_tp1 not None,
    double[:,::1] u_t not None,
    unsigned int nx, unsigned int nz,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] dens not None):
    """
    Apply nonreflexive boundary contitions to elastic SH waves.
    """
    cdef:
        unsigned int i, j
    # Left
    for i in range(nz):
        for j in range(3):
            u_tp1[i,j] = u_t[i,j] + dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j+1] - u_t[i,j])/dx
    # Right
    for i in range(nz):
        for j in range(nx - 3, nx):
            u_tp1[i,j] = u_t[i,j] - dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j] - u_t[i,j-1])/dx
    # Bottom
    for i in range(nz - 3, nz):
        for j in range(nx):
            u_tp1[i,j] = u_t[i,j] - dt*sqrt(mu[i,j]/dens[i,j])*(
                u_t[i,j] - u_t[i-1,j])/dz
    # Top
    for j in range(nx):
        u_tp1[2,j] = u_tp1[3,j]
        u_tp1[1,j] = u_tp1[2,j]
        u_tp1[0,j] = u_tp1[1,j]

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_sh(
    double[:,::1] u_tp1 not None,
    double[:,::1] u_t not None,
    double[:,::1] u_tm1 not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] dens not None):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    cdef:
        unsigned int i, j
        double dt2, dx2, dz2
    dt2 = dt**2
    dx2 = dx**2
    dz2 = dz**2
    for i in range(z1, z2):
        for j in range(x1, x2):
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
    double[:,::1] ux_tp1 not None,
    double[:,::1] ux_t not None,
    double[:,::1] ux_tm1 not None,
    double[:,::1] uz not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] lamb not None,
    double[:,::1] dens not None):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    cdef:
        unsigned int i, j
        double dt2, tauxx_p, tauxx_m, tauxz_p, tauxz_m, l, m
    dt2 = dt**2
    for i in range(z1, z2):
        for j in range(x1, x2):
            #l, m = lamb[i,j], mu[i,j]
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
            tauxz_p = m*((ux_t[i+1,j] - ux_t[i,j])/dz +
                (0.5*(uz[i+1,j+1] + uz[i,j]) - 0.5*(uz[i+1,j-1] + uz[i,j]))/dx)
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauxz_m = m*((ux_t[i,j] - ux_t[i-1,j])/dz +
                (0.5*(uz[i-1,j+1] + uz[i,j]) - 0.5*(uz[i-1,j-1] + uz[i,j]))/dx)
            ux_tp1[i,j] = 2*ux_t[i,j] - ux_tm1[i,j] + (dt2/dens[i,j])*(
                (tauxx_p - tauxx_m)/dx + (tauxz_p - tauxz_m)/dz)

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_elastic_psv_z(
    double[:,::1] uz_tp1 not None,
    double[:,::1] uz_t not None,
    double[:,::1] uz_tm1 not None,
    double[:,::1] ux not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] lamb not None,
    double[:,::1] dens not None):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    cdef:
        unsigned int i, j
        double dt2, tauzz_p, tauzz_m, tauxz_p, tauxz_m, l, m
    dt2 = dt**2
    for i in range(z1, z2):
        for j in range(x1, x2):
            #l, m = lamb[i,j], mu[i,j]
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
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxz_p = m*(
                (uz_t[i,j+1] - uz_t[i,j])/dx +
                (0.5*(ux[i+1,j+1] + ux[i,j]) - 0.5*(ux[i-1,j+1] + ux[i,j]))/dz)
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxz_m = m*((uz_t[i,j] - uz_t[i,j-1])/dx +
                (0.5*(ux[i+1,j-1] + ux[i,j]) - 0.5*(ux[i-1,j-1] + ux[i,j]))/dz)
            uz_tp1[i,j] = 2*uz_t[i,j] - uz_tm1[i,j] + (dt2/dens[i,j])*(
                (tauzz_p - tauzz_m)/dz + (tauxz_p - tauxz_m)/dx)
