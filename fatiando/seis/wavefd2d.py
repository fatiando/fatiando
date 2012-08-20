r"""
Finite difference solution of the 2D wave equation for isotropic media.

Simulates both elastic and acoustic waves:

* :func:`~fatiando.seis._wavefd2d.elastic_psv`: Simulates the coupled P and SV
  elastic waves
* :func:`~fatiando.seis._wavefd2d.elastic_sh`: Simulates SH elastic waves

**Auxiliary function**

* :func:`~fatiando.seis.wavefd2d.lame`: Calculate the Lame constants from P and
  S wave velocities and density

**Sources**

* :class:`~fatiando.seis.wavefd2d.MexHatSource`: Mexican hat wavelet source

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
    - \rho \partial_t^2 u_y = f_y

and

.. math::

    (\lambda + 2\mu)\partial_x^2 u_x + \mu\partial_z^2 u_x
    + (\lambda + \mu)\partial_x\partial_z u_z
    - \rho \partial_t^2 u_x = f_x

.. math::

    (\lambda + 2\mu)\partial_z^2 u_z + \mu\partial_x^2 u_z
    + (\lambda + \mu)\partial_x\partial_z u_x
    - \rho \partial_t^2 u_z = f_z

The first equation depends only on :math:`u_y` and represents SH waves.
The other two depend on :math:`u_x` and :math:`u_z` and are coupled.
They represent P and SV waves.

The explicit finite difference solution for the SH waves is:

.. math::
   :nowrap:

    \begin{align*}
        u_y[i,j]_{t+1} =& 2u_y[i,j]_{t} - u_y[i,j]_{t-1}
        + \frac{\Delta t^2}{\rho[i,j]}
        \left[
            -f_y[i,j]_t +
            \mu[i,j]
                \left(
                    \frac{u_y[i+1,j]_{t} - 2u_y[i,j]_{t} + u_y[i-1,j]_{t}}{
                        \Delta x^2}
                \right.
        \right.
        \\[0.3cm] &
        \left.
            \left.
                +
                \frac{u_y[i,j+1]_{t} - 2u_y[i,j]_{t} + u_y[i,j-1]_{t}}{\Delta z^2}
            \right)
        \right]
    \end{align*}


where :math:`[i,j]_t` is the quantity at the grid node i,j at a
time t.

The solution for P and SV waves is:

.. math::
   :nowrap:

    \begin{align*}
    u_x[i,j]_{t+1} =& 2u_x[i,j]_{t} - u_x[i,j]_{t-1}
    \\&
    + \frac{\Delta t^2}{\rho[i,j]}
    \left\lbrace
        -f_x[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_x[i+1,j]_{t} - 2u_x[i,j]_{t} + u_x[i-1,j]_{t}}{
                \Delta x^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_x[i,j+1]_{t} - 2u_x[i,j]_{t} + u_x[i,j-1]_{t}}{
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
        -f_z[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_z[i,j+1]_{t} - 2u_z[i,j]_{t} + u_z[i,j-1]_{t}}{
                \Delta z^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_z[i+1,j]_{t} - 2u_z[i,j]_{t} + u_z[i-1,j]_{t}}{
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
    from fatiando.seis._cwavefd2d import *
except ImportError:
    from fatiando.seis._wavefd2d import *


log = fatiando.log.dummy('fatiando.seis.wavefd2d')


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

        .. warning:: Don't put sources in the boundaries of the grid!

    * amp : float
        The amplitude of the source (:math:`A`)
    * std : float
        The "wave length" (:math:`\sigma`)
    * delay : float
        The delay before the source starts

        .. note:: If you want the source to start with amplitude close to 0, use
            ``delay = 3.5*std``.

    """

    def __init__(self, i, j, amp, std, delay=0):
        self.i = i
        self.j = j
        self.amp = amp
        self.std = std
        self.delay = delay

    def __call__(self, time):
        t = time - self.delay
        psi = (self.amp*
            (2./(numpy.sqrt(3.*self.std)*(numpy.pi**0.25)))*
            (1. - (t**2)/(self.std**2))*
            numpy.exp(-(t**2)/(2.*self.std**2)))
        return psi

    def coords(self):
        """
        Get the i,j coordinates of the source in the finite difference grid.

        Returns:

        * (i,j) : tuple
            The i,j coordinates

        """
        return (self.i, self.j)

class SinSQRSource(MexHatSource):

    def __init__(self, i, j, amp, std, delay=0):
        MexHatSource.__init__(self, i, j, amp, std, delay)

    def __call__(self, time):
        t = time - self.delay
        if t > self.std:
            return 0
        psi = self.amp*numpy.sin(2.*numpy.pi*t/self.std)**2
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

def elastic_sh(spacing, shape, mu, dens, deltat, iterations, sources):
    """
    Simulate SH waves using an explicit finite differences scheme.

    Parameters:

    * spacing : (dz, dx)
        The node spacing of the finite differences grid
    * shape : (nz, nx)
        The number of nodes in the grid in the z and x directions
    * mu : 2D-array (shape = *shape*)
        The value of the mu Lame constant at all the grid nodes
    * dens : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * deltat : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * sources : list
        A list of the sources of waves
        (see :class:`~fatiando.seis.wavefd2d.MexHatSource` for an example
        source)

    Yields:

    * uy : 2D-array
        The particle movement in the y direction at each time step

    """
    nz, nx = shape
    dz, dx = spacing
    u_tm1 = numpy.zeros(shape, dtype=numpy.float)
    u_t = numpy.zeros(shape, dtype=numpy.float)
    u_tp1 = numpy.zeros(shape, dtype=numpy.float)
    for src in sources:
        u_t[src.coords()] = src(0)
    for t in xrange(1, iterations):
        _step_elastic_sh(u_tp1, u_t, u_tm1, nx, nz, deltat, dx, dz, mu, dens)
        # Update the sources
        for src in sources:
            i, j = src.coords()
            u_tp1[i, j] -= (deltat**2/dens[i, j])*src(t*deltat)
            print u_tp1[i, j]
            #u_tp1[i, j] = src(t*deltat)
        # Set the boundary conditions
        u_tp1[0,:] = u_tp1[1,:]
        u_tp1[-1,:] = 0
        u_tp1[:,0] = 0
        u_tp1[:,-1] = 0

        u_tm1 = u_t
        u_t = u_tp1[:,:]
        yield u_t

