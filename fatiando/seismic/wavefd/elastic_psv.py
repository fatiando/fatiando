from __future__ import division
from numpy import sqrt
import numba



@numba.jit(nopython=True, nogil=True)
def nonreflexive_psv_bc(ux, uz, tp1, t, tm1, nx, nz, dt, dx, dz, mu, lamb, dens):
    """
    Apply nonreflexive boundary contitions to elastic P-SV waves.
    """
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


@numba.jit(nopython=True, nogil=True)
def timestep_psv_psg(ux, uz, tp1, t, tm1, x1, x2, z1, z2, dt, dx, dz, mu, lamb, dens):
    """
    Perform a single time step in the Finite Difference solution for P-SV
    elastic waves.
    """
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
