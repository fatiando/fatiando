r"""
Finite difference solution of the 2D wave equation for isotropic media.

Simulates both elastic and acoustic waves:

* :func:`~fatiando.seis.wavefd.elastic_psv`: Simulates the coupled P and SV
  elastic waves
* :func:`~fatiando.seis.wavefd.elastic_sh`: Simulates SH elastic waves

**Sources**

* :class:`~fatiando.seis.wavefd.MexHatSource`: Mexican hat wavelet source
* :class:`~fatiando.seis.wavefd.SinSqrSource`: Sine squared source

**Auxiliary function**

* :func:`~fatiando.seis.wavefd.lame`: Calculate the Lame constants from P and
  S wave velocities and density

**Theory**

We start with the wave equation for elastic isotropic media

.. math::

    (\lambda + \mu)\nabla(\nabla\cdot\mathbf{u})
    +
    \mu\nabla^2\mathbf{u}
    - \rho \partial_t^2 \mathbf{u} = - \mathbf{f}

where :math:`\mathbf{u} = (u_x, u_y, y_z)` is the particle movement vector,
:math:`\rho` is the density,
:math:`\lambda` and :math:`\mu` are the Lame constants,
and :math:`\mathbf{f}` is the source vector.

In the 2D approximation, we assume all derivatives in the y direction
are zero and consider only x and z coordinates
(though :math:`u_y` remains).
The three equations in the vector equation above can be separated into
two groups:

.. math::

    \mu\left(\partial_x^2 u_y + \partial_z^2 u_y\right)
    - \rho \partial_t^2 u_y = -f_y

and

.. math::

    (\lambda + 2\mu)\partial_x^2 u_x + \mu\partial_z^2 u_x
    + (\lambda + \mu)\partial_x\partial_z u_z
    - \rho \partial_t^2 u_x = -f_x

.. math::

    (\lambda + 2\mu)\partial_z^2 u_z + \mu\partial_x^2 u_z
    + (\lambda + \mu)\partial_x\partial_z u_x
    - \rho \partial_t^2 u_z = -f_z

The first equation depends only on :math:`u_y` and represents SH waves.
The other two depend on :math:`u_x` and :math:`u_z` and are coupled.
They represent P and SV waves.

I'll use an explicit finite difference solution for these equations.
I'll use a second order approximation for the time derivative
and a fourth order approximation for the spacial derivatives.

The finite difference solution for SH waves is:

.. math::
   :nowrap:

    \begin{align*}
        u_y[i,j]_{t+1} =& 2u_y[i,j]_{t} - u_y[i,j]_{t-1}
        + \frac{\Delta t^2}{\rho[i,j]}
        \Biggl[
            f_y[i,j]_t
        \\[0.3cm] &
            + \mu[i,j]
                \left(
                    \frac{-u_y[i,j+2]_{t} + 16u_y[i,j+1]_{t} -30u_y[i,j]_{t}
                        + 16u_y[i,j-1]_{t} - u_y[i,j-2]_{t}}{
                        12\Delta x^2}
                \right)
        \\[0.3cm] &
            + \mu[i,j]
                \left(
                    \frac{-u_y[i+2,j]_{t} + 16u_y[i+1,j]_{t} -30u_y[i,j]_{t}
                        + 16u_y[i-1,j]_{t} - u_y[i-2,j]_{t}}{
                        12\Delta z^2}
                \right)
        \Biggr]
    \end{align*}


where :math:`[i,j]_t` is the quantity at the grid node i,j at a
time t. In this formulation, i denotes z coordinates and j x coordinates.

The solution for P and SV waves is:

.. math::
   :nowrap:

    \begin{align*}
    u_x[i,j]_{t+1} =& 2u_x[i,j]_{t} - u_x[i,j]_{t-1}
    \\&
    + \frac{\Delta t^2}{\rho[i,j]}
    \left\lbrace
        f_x[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_x[i,j+1]_{t} - 2u_x[i,j]_{t} + u_x[i,j-1]_{t}}{
                \Delta x^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_x[i+1,j]_{t} - 2u_x[i,j]_{t} + u_x[i-1,j]_{t}}{
                \Delta z^2}
        \right)
    \\[0.3cm]&
    \left.
        +
        (\lambda[i,j] + \mu[i,j])
        \left(
            \frac{u_z[i,j]_{t} - u_z[i-1,j]_{t} - u_z[i,j-1]_{t} +
            u_z[i-1,j-1]_{t}
            }{\Delta x\Delta z}
        \right)
    \right\rbrace
    \end{align*}

.. math::
   :nowrap:

    \begin{align*}
    u_z[i,j]_{t+1} =& 2u_z[i,j]_{t} - u_z[i,j]_{t-1}
    \\&
    + \frac{\Delta t^2}{\rho[i,j]}
    \left\lbrace
        f_z[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_z[i+1,j]_{t} - 2u_z[i,j]_{t} + u_z[i-1,j]_{t}}{
                \Delta z^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_z[i,j+1]_{t} - 2u_z[i,j]_{t} + u_z[i,j-1]_{t}}{
                \Delta x^2}
        \right)
    \\[0.3cm]&
    \left.
        +
        (\lambda[i,j] + \mu[i,j])
        \left(
            \frac{u_x[i,j]_{t} - u_x[i-1,j]_{t} - u_x[i,j-1]_{t} +
            u_x[i-1,j-1]_{t}
            }{\Delta x\Delta z}
        \right)
    \right\rbrace
    \end{align*}


----

"""
import numpy

import fatiando.log

try:
    from fatiando.seis._cwavefd import *
except ImportError:
    from fatiando.seis._wavefd import *


log = fatiando.log.dummy('fatiando.seis.wavefd')


class MexHatSource(object):
    r"""
    A wave source that vibrates as a mexicam hat (Ricker) wavelet.

    .. math::

        \psi(t) = A\frac{2}{\sqrt{3\sigma}\pi^{\frac{1}{4}}}
        \left( 1 - \frac{t^2}{\sigma^2} \right)
        \exp\left(\frac{-t^2}{2\sigma^2}\right)

    Parameters:

    * i, j : int
        The i,j coordinates of the source in the target finite difference grid.
        i is the index for z, j for x
    * amp : float
        The amplitude of the source (:math:`A`)
    * wlength : float
        The "wave length" (:math:`\sigma`)
    * delay : float
        The delay before the source starts

        .. note:: If you want the source to start with amplitude close to 0, use
            ``delay = 3.5*wlength``.

    """

    def __init__(self, i, j, amp, wlength, delay=0):
        self.i = i
        self.j = j
        self.amp = amp
        self.wlength = wlength
        self.delay = delay

    def __call__(self, time):
        t = time - self.delay
        psi = (self.amp*
            (2./(numpy.sqrt(3.*self.wlength)*(numpy.pi**0.25)))*
            (1. - (t**2)/(self.wlength**2))*
            numpy.exp(-(t**2)/(2.*self.wlength**2)))
        return psi

    def coords(self):
        """
        Get the i,j coordinates of the source in the finite difference grid.

        Returns:

        * (i,j) : tuple
            The i,j coordinates

        """
        return (self.i, self.j)

class SinSqrSource(MexHatSource):
    r"""
    A wave source that vibrates as a sine squared function.

    .. math::

        \psi(t) = A\sin\left(t\frac{2\pi}{T}\right)^2

    Parameters:

    * i, j : int
        The i,j coordinates of the source in the target finite difference grid.
        i is the index for z, j for x
    * amp : float
        The amplitude of the source (:math:`A`)
    * wlength : float
        The wave length (:math:`T`)
    * delay : float
        The delay before the source starts

        .. note:: If you want the source to start with amplitude close to 0, use
            ``delay = 3.5*wlength``.

    """

    def __init__(self, i, j, amp, wlength, delay=0):
        MexHatSource.__init__(self, i, j, amp, wlength, delay)

    def __call__(self, time):
        t = time - self.delay
        if t > self.wlength:
            return 0
        psi = self.amp*numpy.sin(2.*numpy.pi*t/float(self.wlength))**2
        return psi

def lame(pvel, svel, dens):
    r"""
    Calculate the Lame constants :math:`\lambda` and :math:`\mu` from the
    P and S wave velocities (:math:`\alpha` and :math:`\beta`) and the density
    (:math:`\rho`).

    .. math::

        \mu = \beta^2 \rho

    .. math::

        \lambda = \alpha^2 \rho - 2\mu

    Parameters:

    * pvel : float or array
        The P wave velocity
    * svel : float or array
        The S wave velocity
    * dens : float or array
        The density

    Returns:

    * [lambda, mu] : floats or arrays
        The Lame constants

    Examples::

        >>> print lame(2000, 1000, 2700)
        (5400000000, 2700000000)
        >>> import numpy as np
        >>> pv = np.array([2000, 3000])
        >>> sv = np.array([1000, 1700])
        >>> dens = np.array([2700, 3100])
        >>> lamb, mu = lame(pv, sv, dens)
        >>> print lamb
        [5400000000 9982000000]
        >>> print mu
        [2700000000 8959000000]

    """
    mu = dens*svel**2
    lamb = dens*pvel**2 - 2*mu
    return lamb, mu

def _add_pad(array, pad, shape):
    """
    Pad the array with the values of the borders
    """
    array_pad = numpy.zeros(shape, dtype=numpy.float)
    array_pad[:-pad, pad:-pad] = array
    for k in xrange(pad):
        array_pad[:-pad,k] = array[:,0]
        array_pad[:-pad,-(k + 1)] = array[:,-1]
    for k in xrange(pad):
        array_pad[-(pad - k),:] = array_pad[-(pad + 1),:]
    return array_pad

def elastic_sh(spacing, shape, svel, dens, deltat, iterations, sources,
    padding=1.0):
    """
    Simulate SH waves using an explicit finite differences scheme.

    Parameters:

    * spacing : (dz, dx)
        The node spacing of the finite differences grid
    * shape : (nz, nx)
        The number of nodes in the grid in the z and x directions
    * svel : 2D-array (shape = *shape*)
        The S wave velocity at all the grid nodes
    * dens : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * deltat : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * sources : list
        A list of the sources of waves
        (see :class:`~fatiando.seis.wavefd.MexHatSource` for an example
        source)
    * padding : float
        The decimal percentage of padding to use in the grid to avoid
        reflections at the borders

    Yields:

    * uy : 2D-array
        The particle movement in the y direction at each time step

    """
    nz, nx = shape
    dz, dx = spacing
    # Add some nodes in x and z for padding to avoid reflections
    pad = int(padding*max(shape))
    decay = 20*pad
    nx += 2*pad
    nz += pad
    # Pad the velocity as well
    svel_pad = _add_pad(svel, pad, (nz, nx))
    # Compute and yield the initial solutions
    u_tm1 = numpy.zeros((nz, nx), dtype=numpy.float)
    u_t = numpy.zeros((nz, nx), dtype=numpy.float)
    u_tp1 = numpy.zeros((nz, nx), dtype=numpy.float)
    for src in sources:
        i, j = src.coords()
        u_t[i, j + pad] += (deltat**2/dens[i, j])*src(0)
    yield u_tm1[:-pad, pad:-pad]
    yield u_t[:-pad, pad:-pad]
    # Time steps
    for t in xrange(1, iterations):
        _step_elastic_sh(u_tp1, u_t, u_tm1, nx, nz, deltat, dx, dz, svel_pad,
            pad, decay)
        # Set the boundary conditions
        u_tp1[1,:] = u_tp1[2,:]
        u_tp1[0,:] = u_tp1[1,:]
        u_tp1[-1,:] *= 0
        u_tp1[-2,:] *= 0
        u_tp1[:,0] *= 0
        u_tp1[:,1] *= 0
        u_tp1[:,-1] *= 0
        u_tp1[:,-2] *= 0
        # Update the sources
        for src in sources:
            i, j = src.coords()
            u_tp1[i, j + pad] += (deltat**2/dens[i, j])*src(t*deltat)
        u_tm1 = numpy.copy(u_t)
        u_t = numpy.copy(u_tp1)
        yield u_t[:-pad, pad:-pad]

def elastic_psv(spacing, shape, pvel, svel, dens, deltat, iterations, xsources,
    zsources, padding=1.0):
    """
    Simulate SH waves using an explicit finite differences scheme.

    Parameters:

    * spacing : (dz, dx)
        The node spacing of the finite differences grid
    * shape : (nz, nx)
        The number of nodes in the grid in the z and x directions
    * svel : 2D-array (shape = *shape*)
        The S wave velocity at all the grid nodes
    * dens : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * deltat : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * xsources : list
        A list of the sources of waves for the particle movement in the x
        direction
        (see :class:`~fatiando.seis.wavefd.MexHatSource` for an example
        source)
    * zsources : list
        A list of the sources of waves for the particle movement in the z
        direction
    * padding : float
        The decimal percentage of padding to use in the grid to avoid
        reflections at the borders

    Yields:

    * ux, uz : 2D-arrays
        The particle movement in the x and z direction at each time step

    """
    nz, nx = shape
    dz, dx = spacing
    # Add some nodes in x and z for padding to avoid reflections
    pad = int(padding*max(shape))
    decay = 20*pad
    nx += 2*pad
    nz += pad
    # Pad the velocity as well
    pvel_pad = _add_pad(pvel, pad, (nz, nx))
    svel_pad = _add_pad(svel, pad, (nz, nx))
    # Compute and yield the initial solutions
    ux_tm1 = numpy.zeros((nz, nx), dtype=numpy.float)
    ux_t = numpy.zeros((nz, nx), dtype=numpy.float)
    ux_tp1 = numpy.zeros((nz, nx), dtype=numpy.float)
    uz_tm1 = numpy.zeros((nz, nx), dtype=numpy.float)
    uz_t = numpy.zeros((nz, nx), dtype=numpy.float)
    uz_tp1 = numpy.zeros((nz, nx), dtype=numpy.float)
    for src in xsources:
        i, j = src.coords()
        ux_t[i, j + pad] += (deltat**2/dens[i, j])*src(0)
    for src in zsources:
        i, j = src.coords()
        uz_t[i, j + pad] += (deltat**2/dens[i, j])*src(0)
    yield ux_tm1[:-pad, pad:-pad], uz_tm1[:-pad, pad:-pad]
    yield ux_t[:-pad, pad:-pad], uz_t[:-pad, pad:-pad]
    # Time steps
    for t in xrange(1, iterations):
        _step_elastic_psv_x(ux_tp1, ux_t, ux_tm1, uz_t, nx, nz, deltat, dx, dz,
            pvel_pad, svel_pad, pad, decay)
        _step_elastic_psv_z(uz_tp1, uz_t, uz_tm1, ux_t, nx, nz, deltat, dx, dz,
            pvel_pad, svel_pad, pad, decay)
        # Set the boundary conditions
        ux_tp1[0,:] = ux_tp1[1,:]
        ux_tp1[-1,:] *= 0
        ux_tp1[:,0] *= 0
        ux_tp1[:,-1] *= 0
        uz_tp1[0,:] = uz_tp1[1,:]
        uz_tp1[-1,:] *= 0
        uz_tp1[:,0] *= 0
        uz_tp1[:,-1] *= 0
        # Update the sources
        for src in xsources:
            i, j = src.coords()
            ux_tp1[i, j + pad] += (deltat**2/dens[i, j])*src(t*deltat)
        for src in zsources:
            i, j = src.coords()
            uz_tp1[i, j + pad] += (deltat**2/dens[i, j])*src(t*deltat)
        ux_tm1 = numpy.copy(ux_t)
        ux_t = numpy.copy(ux_tp1)
        uz_tm1 = numpy.copy(uz_t)
        uz_t = numpy.copy(uz_tp1)
        yield ux_t[:-pad, pad:-pad], uz_t[:-pad, pad:-pad]


