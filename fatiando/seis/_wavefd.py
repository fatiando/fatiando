"""
.. topic:: Pure Python implementation.

    Module :mod:`fatiando.seis.wavefd` loads the time stepping functions from
    ``fatiando.seis._wavefd``, which contain the pure Python
    implementations. There is also a faster Cython module
    ``fatiando.seis._cwavefd``. If it is available, then will substitude
    the pure Python functions with its functions.

"""
__all__ = ['_step_elastic_sh']

import numpy


def _step_elastic_sh(u_tp1, u_t, u_tm1, nx, nz, dt, dx, dz, svel, pad, decay):
    """
    Perform a single time step in the Finite Difference solution for elastic
    SH waves.
    """
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
            u_tp1[i,j] = (2.*u_t[i,j] - u_tm1[i,j]
                + (svel[i,j]**2)*(dt**2)*(
                    (-u_t[i,j + 2] + 16.*u_t[i,j + 1] - 30.*u_t[i,j] +
                     16.*u_t[i,j - 1] - u_t[i,j - 2])/(12.*dx**2) +
                    (-u_t[i + 2,j] +16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                u_tp1[i,j] *= numpy.exp(-in_pad**2/decay**2)
