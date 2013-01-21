"""
Pure Python implementation of the time stepping functions for
fatiando.seismic.wavefd
"""
import numpy


def _apply_damping(array, nx, nz, pad, decay):
    """
    Apply a decay factor to the values of the array in the padding region.
    """
    for i in xrange(nz):
        for j in xrange(nx):
            in_pad = -1
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                array[i,j] *= numpy.exp(-in_pad**2/decay**2)

def step_elastic_sh(u_tp1, u_t, u_tm1, nx, nz, dt, dx, dz, svel):
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
                    (-u_t[i + 2,j] + 16.*u_t[i + 1,j] - 30.*u_t[i,j] +
                     16.*u_t[i - 1,j] - u_t[i - 2,j])/(12.*dz**2)))

def step_elastic_psv_x(ux_tp1, ux_t, ux_tm1, uz_t, nx, nz, dt, dx, dz, pvel,
    svel, pad, decay):
    """
    Perform a single time step in the Finite Difference solution for ux elastic
    P and SV waves.
    """
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
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
                    - 8.*uz_t[i - 1,j - 2] + uz_t[i - 2,j - 2])/(144.*dx*dz)
                )
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1.
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                ux_tp1[i,j] *= numpy.exp(-in_pad**2/decay**2)

def step_elastic_psv_z(uz_tp1, uz_t, uz_tm1, ux_t, nx, nz, dt, dx, dz, pvel,
    svel, pad, decay):
    """
    Perform a single time step in the Finite Difference solution for uz elastic
    P and SV waves.
    """
    for i in xrange(2, nz - 2):
        for j in xrange(2, nx - 2):
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
                    - 8.*ux_t[i - 1,j - 2] + ux_t[i - 2,j - 2])/(144.*dx*dz)
                )
            # Damp the amplitudes after the paddings to avoid reflections
            in_pad = -1.
            if j < pad:
                in_pad = pad - j
            if j >= nx - pad:
                in_pad = j - nx + pad + 1
            if i >= nz - pad:
                in_pad = i - nz + pad + 1
            if in_pad != -1:
                uz_tp1[i,j] *= numpy.exp(-in_pad**2/decay**2)
