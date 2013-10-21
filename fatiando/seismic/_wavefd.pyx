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

__all__ = [
    '_apply_damping',
    '_apply_damping3',
    '_step_elastic_sh',
    '_step_elastic_psv',
    '_xz2ps',
    '_nonreflexive_sh_boundary_conditions',
    '_nonreflexive_psv_boundary_conditions',
    '_reflexive_scalar_boundary_conditions',
    '_reflexive_scalar3_boundary_conditions',
    '_step_scalar',
    '_step_scalar3_esg']

@cython.boundscheck(False)
@cython.wraparound(False)
def _xz2ps(
    numpy.ndarray[double, ndim=2] ux not None,
    numpy.ndarray[double, ndim=2] uz not None,
    double[:,::1] p not None,
    double[:,::1] s not None,
    unsigned int nx, unsigned int nz,
    double dx, double dz):
    """
    Convert ux and uz to P and S waves.
    """
    cdef:
        unsigned int i, j
        double tmpx, tmpz
    tmpx = dx*12.
    tmpz = dz*12.
    for i in range(2, nz - 2):
        for j in range(2, nx - 2):
            p[i,j] = (
                (-uz[i+2,j] + 8*uz[i+1,j] - 8*uz[i-1,j] + uz[i-2,j])/tmpz
                + (-ux[i,j+2] + 8*ux[i,j+1] - 8*ux[i,j-1] + ux[i,j-2])/tmpx)
            s[i,j] = (
                (-ux[i+2,j] + 8*ux[i+1,j] - 8*ux[i-1,j] + ux[i-2,j])/tmpz
                - (-uz[i,j+2] + 8*uz[i,j+1] - 8*uz[i,j-1] + uz[i,j-2])/tmpx)
    # Fill in the borders with the same values
    for i in range(nz):
        p[i,nx-2] = p[i,nx-3]
        p[i,nx-1] = p[i,nx-2]
        p[i,1] = p[i,2]
        p[i,0] = p[i,1]
        s[i,nx-2] = s[i,nx-3]
        s[i,nx-1] = s[i,nx-2]
        s[i,1] = s[i,2]
        s[i,0] = s[i,1]
    for j in range(nx):
        p[nz-2,j] = p[nz-3,j]
        p[nz-1,j] = p[nz-2,j]
        p[1,j] = p[2,j]
        p[0,j] = p[1,j]
        s[nz-2,j] = s[nz-3,j]
        s[nz-1,j] = s[nz-2,j]
        s[1,j] = s[2,j]
        s[0,j] = s[1,j]

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
def _apply_damping3(double[:,:,::1] array not None,
    unsigned int nx, unsigned int ny, unsigned int nz,
    unsigned int pad, double decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    Damping 3D padding regions. x (north), y (east), z (down)
    """
    cdef:
        unsigned int i, j, k
    # Damping on the north/south
    for k in range(nz-pad):
        for j in range(pad,ny-pad):  # avoid double damping
            for i in range(pad):
                array[k, j, i] *= exp(-(decay*(pad - i))**2) # south
                array[k, j, (nx-pad) + i] *= exp(-(decay*i)**2) # north
    # Damping on the east/west
    for k in range(nz-pad):
        for i in range(nx):
            for j in range(pad):
                array[k, j, i] *= exp(-(decay*(pad - j))**2) # west
                array[k, (ny-pad) + j, i] *= exp(-(decay*j)**2) # east
    # Damping on the bottom
    for i in range(nx):
        for j in range(ny):
            for k in range(pad):
                array[(nz-pad) + k, i, j] *= exp(-(decay*k)**2)


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
    for i in range(nz):
        # Left
        j = 0
        ux[tp1,i,j] = ux[t,i,j] + dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j+1] - ux[t,i,j])/dx
        uz[tp1,i,j] = uz[t,i,j] + dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j+1] - uz[t,i,j])/dx
        # Right
        j = nx - 1
        ux[tp1,i,j] = ux[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j] - ux[t,i,j-1])/dx
        uz[tp1,i,j] = uz[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j] - uz[t,i,j-1])/dx
    # Bottom
    i = nz - 1
    for j in range(nx):
        ux[tp1,i,j] = ux[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j] - ux[t,i-1,j])/dz
        uz[tp1,i,j] = uz[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j] - uz[t,i-1,j])/dz

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
def _step_elastic_psv(
    double[:,:,::1] ux not None,
    double[:,:,::1] uz not None,
    unsigned int tp1, unsigned int t, unsigned int tm1,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double dx, double dz,
    double[:,::1] mu not None,
    double[:,::1] lamb not None,
    double[:,::1] dens not None):
    """
    Perform a single time step in the Finite Difference solution for P-SV
    elastic waves.
    """
    cdef:
        unsigned int i, j
        double dt2, tauzz_p, tauzz_m, tauxx_p, tauxx_m, tauxz_p, tauxz_m, l, m
    dt2 = dt**2
    for i in range(z1, z2):
        for j in range(x1, x2):
            # Step the ux component
            l = 0.5*(lamb[i,j+1] + lamb[i,j])
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxx_p = (l + 2*m)*(ux[t,i,j+1] - ux[t,i,j])/dx + l*0.25*(
                uz[t,i+1,j+1] + uz[t,i+1,j] - uz[t,i-1,j+1] - uz[t,i-1,j])/dz
            l = 0.5*(lamb[i,j-1] + lamb[i,j])
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxx_m = (l + 2*m)*(ux[t,i,j] - ux[t,i,j-1])/dx + l*0.25*(
                uz[t,i+1,j] + uz[t,i+1,j-1] - uz[t,i-1,j] - uz[t,i-1,j-1])/dz
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauxz_p = m*((ux[t,i+1,j] - ux[t,i,j])/dz + 0.25*(
                uz[t,i+1,j+1] + uz[t,i,j+1]- uz[t,i+1,j-1] - uz[t,i,j-1])/dx)
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauxz_m = m*((ux[t,i,j] - ux[t,i-1,j])/dz + 0.25*(
                uz[t,i,j+1] + uz[t,i-1,j+1]- uz[t,i,j-1]  - uz[t,i-1,j-1])/dx)
            ux[tp1,i,j] = 2*ux[t,i,j] - ux[tm1,i,j] + (dt2/dens[i,j])*(
                (tauxx_p - tauxx_m)/dx + (tauxz_p - tauxz_m)/dz)
            # Step the uz component
            l = 0.5*(lamb[i+1,j] + lamb[i,j])
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauzz_p = (l + 2*m)*(uz[t,i+1,j] - uz[t,i,j])/dz + l*0.25*(
                ux[t,i+1,j+1] + ux[t,i,j+1] - ux[t,i+1,j-1] - ux[t,i,j-1])/dx
            l = 0.5*(lamb[i-1,j] + lamb[i,j])
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauzz_m = (l + 2*m)*(uz[t,i,j] - uz[t,i-1,j])/dz + l*0.25*(
                ux[t,i,j+1] + ux[t,i-1,j+1] - ux[t,i,j-1] - ux[t,i-1,j-1])/dx
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxz_p = m*((uz[t,i,j+1] - uz[t,i,j])/dx + 0.25*(
                ux[t,i+1,j+1] + ux[t,i+1,j] - ux[t,i-1,j+1] - ux[t,i-1,j])/dz)
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxz_m = m*((uz[t,i,j] - uz[t,i,j-1])/dx + 0.25*(
                ux[t,i+1,j] + ux[t,i+1,j-1]- ux[t,i-1,j]  - ux[t,i-1,j-1])/dz)
            uz[tp1,i,j] = 2*uz[t,i,j] - uz[tm1,i,j] + (dt2/dens[i,j])*(
                (tauzz_p - tauzz_m)/dz + (tauxz_p - tauxz_m)/dx)

@cython.boundscheck(False)
@cython.wraparound(False)
def _reflexive_scalar_boundary_conditions(
    double[:,::1] u not None,
    unsigned int nx, unsigned int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    4th order (+2-2) indexes
    """
    cdef unsigned int i
    # Top
    for i in xrange(nx):
        u[1, i] = u[2, i] #up
        u[0, i] = u[1, i]
        u[nz - 1, i] *= 0 #down
        u[nz - 2, i] *= 0
    # Sides
    for i in xrange(nz):
        u[i, 0] *= 0 #left
        u[i, 1] *= 0
        u[i, nx - 1] *= 0 #right
        u[i, nx - 2] *= 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_scalar(
    double[:,::1] u_tp1 not None,
    double[:,::1] u_t not None,
    double[:,::1] u_tm1 not None,
    unsigned int x1, unsigned int x2, unsigned int z1, unsigned int z2,
    double dt, double ds,
    double[:,::1] vel not None):
    """
    Perform a single time step in the Finite Difference solution for scalar
    waves 4th order in space
    """
    cdef unsigned int i, j
    for i in xrange(z1, z2):
        for j in xrange(x1, x2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + ((vel[i,j]*dt/ds)**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/12. +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/12.))

@cython.boundscheck(False)
@cython.wraparound(False)
def _reflexive_scalar3_boundary_conditions(
    double[:,:,::1] u not None,
    unsigned int nx, unsigned int ny, unsigned int nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others. 3D
    4th order Staggered (+3-3) indexes, x (north), y (east), z (down)
    """
    cdef unsigned int i, j, k
    # Top
    for j in range(ny):
        for i in range(nx):
            u[0, j, i] = u[1, j, i]
            u[1, j, i] = u[2, j, i]  # up reflexive
            u[2, j, i] = u[3, j, i]
            u[nz - 1, j, i] *= 0  # down fixed
            u[nz - 2, j, i] *= 0
            u[nz - 3, j, i] *= 0
    # Sides
    for k in range(nz):
        for j in range(ny):
            u[k, j, 0] *= 0 #south reflexive
            u[k, j, 1] *= 0
            u[k, j, 2] *= 0
            u[k, j, nx-1] *= 0 #north reflexive
            u[k, j, nx-2] *= 0
            u[k, j, nx-3] *= 0
    for k in range(nz):
        for i in range(nx):
            u[k, 0, i] *= 0 #west reflexive
            u[k, 1, i] *= 0
            u[k, 2, i] *= 0
            u[k, ny-1, i] *= 0 #east reflexive
            u[k, ny-2, i] *= 0
            u[k, ny-3, i] *= 0

@cython.boundscheck(False)
@cython.wraparound(False)
def _step_scalar3_esg(
    double[:,:,::1] p_tp1 not None,
    double[:,:,::1] p_t not None,
    double[:,:,::1] p_tm1 not None,
    unsigned int x1, unsigned int x2,
    unsigned int y1, unsigned int y2,
    unsigned int z1, unsigned int z2,
    double dt, double dx, double dy, double dz,
    double[:,:,::1] b not None,
    double[:,:,::1] kmod not None):
    """
    Perform a single time step in Equivalent Staggered FD 3D solution for
    scalar waves 4th order in space
    """

    cdef:
        unsigned int i, j, k
        double dt2, dx2, dy2, dz2
    dt2 = dt**2
    dx2 = 1./(dx**2)
    dy2 = 1./(dy**2)
    dz2 = 1./(dz**2)
    c1 = (3./64)
    c2 = (1./576)
    for k in range(z1, z2):
        for j in range(y1, y2):
            for i in range(x1, x2):
                p_tp1[k, j, i] = ( 2.*p_t[k, j, i] - p_tm1[k, j, i] +
                    kmod[k, j, i] * dt2 * (
                    +dx2*c1*( 0.5*(b[k, j, i+1]+b[k, j, i])*(27*
                    (p_t[k, j, i+1]-p_t[k, j, i])-p_t[k, j, i+2]+p_t[k, j, i-1])-
                    0.5*(b[k, j, i-1]+b[k, j, i])*(27*(p_t[k, j, i]-
                    p_t[k, j, i-1])-p_t[k, j, i+1]+p_t[k, j, i-2]) )
                    -dx2*c2*( 0.5*(b[k, j, i+2]+b[k, j, i+1])*(27*
                    (p_t[k, j, i+2]-p_t[k, j, i+1])-p_t[k, j, i+3]+p_t[k, j, i])-
                    0.5*(b[k, j, i-2]+b[k, j, i-1])*(27*(p_t[k, j, i-1]-
                    p_t[k, j, i-2])-p_t[k, j, i]+p_t[k, j, i-3]) )
                    +dy2*c1*( 0.5*(b[k, j+1, i]+b[k, j, i])*(27*
                    (p_t[k, j+1, i]-p_t[k, j, i])-p_t[k, j+2, i]+p_t[k, j-1, i])-
                    0.5*(b[k, j-1, i]+b[k, j, i])*(27*(p_t[k, j, i]-
                    p_t[k, j-1, i])-p_t[k, j+1, i]+p_t[k, j-2, i]) )
                    -dy2*c2*( 0.5*(b[k, j+2, i]+b[k, j+1, i])*(27*
                    (p_t[k, j+2, i]-p_t[k, j+1, i])-p_t[k, j+3, i]+p_t[k, j, i])-
                    0.5*(b[k, j-2, i]+b[k, j-1, i])*(27*(p_t[k, j-1, i]-
                    p_t[k, j-2, i])-p_t[k, j, i]+p_t[k, j-3, i]) )
                    +dz2*c1*( 0.5*(b[k+1, j, i]+b[k, j, i])*(27*
                    (p_t[k+1, j, i]-p_t[k, j, i])-p_t[k+2, j, i]+p_t[k-1, j, i])-
                    0.5*(b[k-1, j, i]+b[k, j, i])*(27*(p_t[k, j, i]-
                    p_t[k-1, j, i])-p_t[k+1, j, i]+p_t[k-2, j, i]) )
                    -dz2*c2*( 0.5*(b[k+2, j, i]+b[k+1, j, i])*(27*
                    (p_t[k+2, j, i]-p_t[k+1, j, i])-p_t[k+3, j, i]+p_t[k, j, i])-
                    0.5*(b[k-2, j, i]+b[k-1, j, i])*(27*(p_t[k-1, j, i]-
                    p_t[k-2, j, i])-p_t[k, j, i]+p_t[k-3, j, i]) )
                    )
                )
