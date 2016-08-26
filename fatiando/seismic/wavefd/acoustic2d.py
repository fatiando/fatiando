from __future__ import division
import numba


@numba.jit(nopython=True, nogil=True)
def timestep_scalar(u_tp1, u_t, u_tm1, x1, x2, z1, z2, dt, ds, vel):
    """
    Perform a single time step in the Finite Difference solution for scalar
    waves 4th order in space
    """
    for i in range(z1, z2):
        for j in range(x1, x2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + ((vel[i,j]*dt/ds)**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/12. +
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/12.))


@numba.jit(nopython=True, nogil=True)
def reflexive_scalar_bc(u, nx, nz):
    """
    Apply the boundary conditions: free-surface at top, fixed on the others.
    4th order (+2-2) indexes
    """
    # Top
    for i in range(nx):
        u[1, i] = u[2, i] #up
        u[0, i] = u[1, i]
        u[nz - 1, i] *= 0 #down
        u[nz - 2, i] *= 0
    # Sides
    for i in range(nz):
        u[i, 0] *= 0 #left
        u[i, 1] *= 0
        u[i, nx - 1] *= 0 #right
        u[i, nx - 2] *= 0


def scalar_maxdt(area, shape, maxvel):
    r"""
    Calculate the maximum time step that can be used in the
    FD scalar simulation with 4th order space 1st time backward.

    References

    Alford R.M., Kelly K.R., Boore D.M. (1974) Accuracy of finite-difference
    modeling of the acoustic wave equation Geophysics, 39 (6), P. 834-842

    Chen, Jing-Bo (2011) A stability formula for Lax-Wendroff methods
    with fourth-order in time and general-order in space for
    the scalar wave equation Geophysics, v. 76, p. T37-T42

    Convergence

    .. math::

         \Delta t \leq \frac{2 \Delta s}{ V \sqrt{\sum_{a=-N}^{N} (|w_a^1| +
         |w_a^2|)}}
         = \frac{ \Delta s \sqrt{3}}{ V_{max} \sqrt{8}}

    Where w_a are the centered differences weights

    Parameters:

    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.
    * shape : (nz, nx)
        The number of nodes in the finite difference grid
    * maxvel : float
        The maximum velocity in the medium

    Returns:

    * maxdt : float
        The maximum time step

    """
    x1, x2, z1, z2 = area
    nz, nx = shape
    spacing = min([(x2 - x1) / (nx - 1), (z2 - z1) / (nz - 1)])
    factor = numpy.sqrt(3. / 8.)
    factor -= factor / 100.  # 1% smaller to guarantee criteria
    # the closer to stability criteria the better the convergence
    return factor * spacing / maxvel
