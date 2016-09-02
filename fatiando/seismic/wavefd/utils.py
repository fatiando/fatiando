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


def lame_lamb(pvel, svel, dens):
    r"""
    Calculate the Lame parameter :math:`\lambda` P and S wave velocities
    (:math:`\alpha` and :math:`\beta`) and the density (:math:`\rho`).

    .. math::

        \lambda = \alpha^2 \rho - 2\beta^2 \rho

    Parameters:

    * pvel : float or array
        The P wave velocity
    * svel : float or array
        The S wave velocity
    * dens : float or array
        The density

    Returns:

    * lambda : float or array
        The Lame parameter

    Examples::

        >>> print lame_lamb(2000, 1000, 2700)
        5400000000
        >>> import numpy as np
        >>> pv = np.array([2000, 3000])
        >>> sv = np.array([1000, 1700])
        >>> dens = np.array([2700, 3100])
        >>> print lame_lamb(pv, sv, dens)
        [5400000000 9982000000]

    """
    lamb = dens * pvel ** 2 - 2 * dens * svel ** 2
    return lamb


def lame_mu(svel, dens):
    r"""
    Calculate the Lame parameter :math:`\mu` from S wave velocity
    (:math:`\beta`) and the density (:math:`\rho`).

    .. math::

        \mu = \beta^2 \rho

    Parameters:

    * svel : float or array
        The S wave velocity
    * dens : float or array
        The density

    Returns:

    * mu : float or array
        The Lame parameter

    Examples::

        >>> print lame_mu(1000, 2700)
        2700000000
        >>> import numpy as np
        >>> sv = np.array([1000, 1700])
        >>> dens = np.array([2700, 3100])
        >>> print lame_mu(sv, dens)
        [2700000000 8959000000]

    """
    mu = dens * svel ** 2
    return mu
