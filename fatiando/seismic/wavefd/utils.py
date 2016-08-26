from __future__ import division
from numpy import exp
import numba


@numba.jit(nopython=True, nogil=True)
def apply_damping(array, nx, nz, pad, decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    """
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


@numba.jit(nopython=True, nogil=True)
def xz2ps(ux, uz, p, s, nx, nz, dx, dz):
    """
    Convert ux and uz to P and S waves.
    """
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
