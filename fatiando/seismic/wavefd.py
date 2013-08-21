r"""
Finite difference solution of the 2D wave equation for isotropic media.

.. warning::

    Due to the high computational demmand of these simulations,
    the pure Python time stepping functions are **very** slow!
    I strongly recommend using the optimized Cython time stepping module.

Simulates both elastic and acoustic waves:

* :func:`~fatiando.seismic.wavefd.elastic_psv`: Simulates the coupled P and SV
  elastic waves
* :func:`~fatiando.seismic.wavefd.elastic_sh`: Simulates SH elastic waves

**Sources**

* :class:`~fatiando.seismic.wavefd.MexHatSource`: Mexican hat wavelet source
* :class:`~fatiando.seismic.wavefd.SinSqrSource`: Sine squared source

**Auxiliary function**

* :func:`~fatiando.seismic.wavefd.lame`: Calculate the Lame constants from P and
  S wave velocities and density

**References**

Di Bartolo, L., C. Dors, and W. J. Mansur (2012), A new family of
finite-difference schemes to solve the heterogeneous acoustic wave equation,
Geophysics, 77(5), T187-T199, doi:10.1190/geo2011-0345.1.

Luo, Y., and G. Schuster (1990), Parsimonious staggered grid
finite-differencing of the wave equation, Geophysical Research Letters, 17(2),
155-158, doi:10.1029/GL017i002p00155.

----

"""
from __future__ import division

import numpy

from fatiando.seismic._wavefd import *

class MexHatSource(object):
    r"""
    A wave source that vibrates as a mexicam hat (Ricker) wavelet.

    .. math::

        \psi(t) = A(1 - 2 \pi^2 f^2 t^2)exp(-\pi^2 f^2 t^2)

    Parameters:

    * x, z : float
        The x, z coordinates of the source
    * area : [xmin, xmax, zmin, zmax]
        The area bounding the finite difference simulation
    * shape : (nz, nx)
        The number of nodes in the finite difference grid
    * amp : float
        The amplitude of the source (:math:`A`)
    * frequency : float
        The peak frequency of the wavelet
    * delay : float
        The delay before the source starts

    """

    def __init__(self, x, z, area, shape, amp, frequency, delay=0):
        nz, nx = shape
        dz, dx = sum(area[2:])/(nz - 1), sum(area[:2])/(nx - 1)
        self.i = int(round((z - area[2])/dz))
        self.j = int(round((x - area[0])/dx))
        self.x, self.z = x, z
        self.amp = amp
        self.frequency = frequency
        self.f2 = frequency**2
        self.delay = delay

    def __call__(self, time):
        t2 = (time - self.delay)**2
        pi2 = numpy.pi**2
        psi = self.amp*(1 - 2*pi2*self.f2*t2)*numpy.exp(-pi2*self.f2*t2)
        return psi

    def coords(self):
        """
        Get the x, z coordinates of the source.

        Returns:

        * (x, z) : tuple
            The x, z coordinates

        """
        return (self.x, self.z)

    def indexes(self):
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

    * x, z : float
        The x, z coordinates of the source
    * area : [xmin, xmax, zmin, zmax]
        The area bounding the finite difference simulation
    * shape : (nz, nx)
        The number of nodes in the finite difference grid
    * amp : float
        The amplitude of the source (:math:`A`)
    * wlength : float
        The wave length (:math:`T`)
    * delay : float
        The delay before the source starts

        .. note:: If you want the source to start with amplitude close to 0, use
            ``delay = 3.5*wlength``.

    """

    def __init__(self, x, z, area, shape, amp, wlength, delay=0):
        super(SinSqrSource, self).__init__(self, x, z, area, shape, amp,
                                           1./wlength, delay)
        self.wlength = wlength

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

def elastic_sh(mu, density, area, dt, iterations, sources, stations=None,
    snapshot=None, padding=50, taper=0.005):
    """
    Simulate SH waves using the Equivalent Staggered Grid (ESG) finite
    differences scheme of Di Bartolo et al. (2012).

    This is an iterator. It yields a panel of $u_y$ displacements and a list
    of arrays with recorded displacements in a time series.
    Parameter *snapshot* controls how often the iterator yields. The default
    is only at the end, so only the final panel and full time series are
    yielded.

    Parameters:

    * mu : 2D-array (shape = *shape*)
        The :math:`\mu` Lame parameter at all the grid nodes
    * density : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the swallowest point is
        at zmin, the deepest at zmax.
    * dt : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * sources : list
        A list of the sources of waves
        (see :class:`~fatiando.seismic.wavefd.MexHatSource` for an example
        source)
    * stations : None or list
        If not None, then a list of [x, z] pairs with the x and z coordinates
        of the recording stations. These are physical coordinates, not the
        indexes!
    * snapshot : None or int
        If not None, than yield a snapshot of the displacement at every
        *snapshot* iterations.
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the gaussian taper function used for the absorbing
        boundary conditions

    Yields:

    * t, uy, seismograms : int, 2D-array and list of 1D-arrays
        The current iteration, the particle displacement in the y direction
        and a list of the displacements recorded at each station until the
        current iteration.

    """
    if mu.shape != density.shape:
        raise ValueError('Density and mu grids should have same shape')
    x1, x2, z1, z2 = area
    nz, nx = mu.shape
    dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)
    # Get the index of the closest point to the stations and start the
    # seismograms
    if stations is not None:
        stations = [[int(round((z - z1)/dz)), int(round((x - x1)/dx))]
                    for x, z in stations]
        seismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
    else:
        stations, seismograms = [], []
    # Add some padding to x and z. The padding region is where the wave is
    # absorbed
    pad = int(padding)
    nx += 2*pad
    nz += pad
    mu_pad = _add_pad(mu, pad, (nz, nx))
    dens_pad = _add_pad(density, pad, (nz, nx))
    # Pack the particle position u at 2 different times in one 3d array
    # u[0] = u(t-1)
    # u[1] = u(t)
    # The next time step overwrites the t-1 panel
    u = numpy.zeros((2, nz, nx), dtype=numpy.float)
    # Compute and yield the initial solutions
    for src in sources:
        i, j = src.indexes()
        u[1, i, j + pad] += (dt**2/density[i, j])*src(0)
    # Update seismograms
    for station, seismogram in zip(stations, seismograms):
        i, j = station
        seismogram[0] = u[1, i, j + pad]
    if snapshot is not None:
        yield 0, u[1, :-pad, pad:-pad], seismograms
    for iteration in xrange(1, iterations):
        t, tm1 = iteration%2, (iteration + 1)%2
        tp1 = tm1
        _step_elastic_sh(u[tp1], u[t], u[tm1], 3, nx - 3, 3, nz - 3, dt, dx,
            dz, mu_pad, dens_pad)
        _apply_damping(u[t], nx, nz, pad, taper)
        _nonreflexive_sh_boundary_conditions(u[tp1], u[t], nx, nz, dt, dx, dz,
            mu_pad, dens_pad)
        _apply_damping(u[tp1], nx, nz, pad, taper)
        for src in sources:
            i, j = src.indexes()
            u[tp1, i, j + pad] += (dt**2/density[i, j])*src(iteration*dt)
        # Update seismograms
        for station, seismogram in zip(stations, seismograms):
            i, j = station
            seismogram[iteration] = u[tp1, i, j + pad]
        if snapshot is not None and iteration%snapshot == 0:
            yield iteration, u[tp1, :-pad, pad:-pad], seismograms
    yield iteration, u[tp1, :-pad, pad:-pad], seismograms

def elastic_psv(spacing, shape, mu, lamb, density, dt, iterations, xsources,
    zsources, padding=50, taper=0.002):
    """
    Simulate P and SV waves using the Parsimoneous Staggered Grid (PSG) finite
    differences scheme of Luo and Schuster (1990).

    Parameters:

    * spacing : (dz, dx)
        The node spacing of the finite differences grid
    * shape : (nz, nx)
        The number of nodes in the grid in the z and x directions
    * mu : 2D-array (shape = *shape*)
        The :math:`\mu` Lame parameter at all the grid nodes
    * lamb : 2D-array (shape = *shape*)
        The :math:`\lambda` Lame parameter at all the grid nodes
    * density : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * dt : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * xsources : list
        A list of the sources of waves for the particle movement in the x
        direction
        (see :class:`~fatiando.seismic.wavefd.MexHatSource` for an example
        source)
    * zsources : list
        A list of the sources of waves for the particle movement in the z
        direction
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the gaussian taper function used for the absorbing
        boundary conditions

    Yields:

    * ux, uz : 2D-arrays
        The particle movement in the x and z direction at each time step

    """
    nz, nx = shape
    dz, dx = (float(i) for i in spacing)
    pad = int(padding)
    nx += 2*pad
    nz += pad
    mu_pad = _add_pad(mu, pad, (nz, nx))
    lamb_pad = _add_pad(lamb, pad, (nz, nx))
    dens_pad = _add_pad(density, pad, (nz, nx))
    # Compute and yield the initial solutions
    ux = numpy.zeros((2, nz, nx), dtype=numpy.float)
    uz = numpy.zeros((2, nz, nx), dtype=numpy.float)
    for src in xsources:
        i, j = src.coords()
        ux[1, i, j + pad] += (dt**2/density[i, j])*src(0)
    for src in zsources:
        i, j = src.coords()
        uz[1, i, j + pad] += (dt**2/density[i, j])*src(0)
    yield ux[1, :-pad, pad:-pad], uz[1, :-pad, pad:-pad]
    for iteration in xrange(1, iterations):
        t, tm1 = iteration%2, (iteration + 1)%2
        tp1 = tm1
        _step_elastic_psv_x(ux[tp1], ux[t], ux[tm1], uz[t], 1, nx - 1,
            1, nz - 1, dt, dx, dz, mu_pad, lamb_pad, dens_pad)
        _step_elastic_psv_z(uz[tp1], uz[t], uz[tm1], ux[t], 1, nx - 1,
            1, nz - 1, dt, dx, dz, mu_pad, lamb_pad, dens_pad)
        _apply_damping(ux[t], nx, nz, pad, taper)
        _apply_damping(uz[t], nx, nz, pad, taper)
        _nonreflexive_psv_boundary_conditions(ux, uz, tp1, t, tm1, nx, nz, dt,
            dx, dz, mu_pad, lamb_pad, dens_pad)
        _apply_damping(ux[tp1], nx, nz, pad, taper)
        _apply_damping(uz[tp1], nx, nz, pad, taper)
        for src in xsources:
            i, j = src.coords()
            ux[tp1, i, j + pad] += (dt**2/density[i, j])*src(iteration*dt)
        for src in zsources:
            i, j = src.coords()
            uz[tp1, i, j + pad] += (dt**2/density[i, j])*src(iteration*dt)
        yield ux[tp1, :-pad, pad:-pad], uz[tp1, :-pad, pad:-pad]
