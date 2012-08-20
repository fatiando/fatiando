"""
.. topic:: Pure Python implementation.

    Module :mod:`fatiando.seis.wavefd2d` loads the time stepping functions from
    ``fatiando.seis._wavefd2d``, which contain the pure Python
    implementations. There is also a faster Cython module
    ``fatiando.seis._cwavefd2d``. If it is available, then will substitude
    the pure Python functions with its functions.

"""
__all__ = ['_step_elastic_sh']

import numpy


def _step_elastic_sh(u_tp1, u_t, u_tm1, nx, nz, dt, dx, dz, mu, dens):
    for i in xrange(1, nz - 1):
        for j in xrange(1, nx - 1):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + mu[i,j]*dt**2/dens[i,j]*(
                    (u_t[i + 1,j] - 2.*u_t[i,j] + u_t[i - 1,j])/dz**2 +
                    (u_t[i,j + 1] - 2.*u_t[i,j] + u_t[i,j - 1])/dx**2))
