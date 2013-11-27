r"""
Finite difference solution of the 2D wave equation for isotropic media.

* :func:`~fatiando.seismic.wavefd.elastic_psv`: Simulates the coupled P and SV
  elastic waves using the Parsimonious Staggered Grid method of Luo and
  Schuster (1990)
* :func:`~fatiando.seismic.wavefd.elastic_sh`: Simulates SH elastic waves using
  the Equivalent Staggered Grid method of Di Bartolo et al. (2012)
* :func:`~fatiando.seismic.wavefd.scalar`: Simulates scalar waves using simple
  explicit finite differences scheme

**Sources**

* :class:`~fatiando.seismic.wavefd.MexHatSource`: Mexican hat wavelet source
* :class:`~fatiando.seismic.wavefd.SinSqrSource`: Sine squared source
* :class:`~fatiando.seismic.wavefd.GaussSource`: Gauss derivative source
* :func:`~fatiando.seismic.wavefd.blast_source`: A source blasting in all
  directions

**Auxiliary functions**

* :func:`~fatiando.seismic.wavefd.lame_lamb`: Calculate the lambda Lame
  parameter
* :func:`~fatiando.seismic.wavefd.lame_mu`: Calculate the mu Lame parameter
* :func:`~fatiando.seismic.wavefd.xz2ps`: Convert x and z displacements to
  representations of P and S waves
* :func:`~fatiando.seismic.wavefd.maxdt`: Calculate the maximum time step for
  elastic wave simulations
* :func:`~fatiando.seismic.wavefd.scalar_maxdt`: Calculate the maximum time
  step for a scalar wave simulation

**Theory**

A good place to start is the equation of motion for elastic isotropic materials

.. math::

    \partial_j \tau_{ij} - \rho \partial_t^2 u_i = -f_i

where :math:`\tau_{ij}` is the stress tensor, :math:`\rho` the density,
:math:`u_i` the displacement (particle motion) and :math:`f_i` is the source
term.
But what I'm interested in modeling are the displacements in x, y and z.
They are what is recorded by the seismometers.
Luckily, I can use Hooke's law to write the stress tensor as a function of the
displacements

.. math::

    \tau_{ij} = \lambda\delta_{ij}\partial_k u_k +
    \mu(\partial_i u_j + \partial_j u_i)

where :math:`\lambda` and :math:`\mu` are the Lame parameters and
:math:`\delta_{ij}` is the Kronecker delta.
Just as a reminder, in tensor notation, :math:`\partial_k u_k` is the divergent
of :math:`\mathbf{u}`. Free indices (not :math:`i,j`) represent a summation.

In a 2D medium, there is no variation in one of the directions.
So I'll choose the y direction for this purpose, so all y-derivatives cancel
out.
Looking at the second component of the equation of motion

.. math::

    \partial_x\tau_{xy} + \underbrace{\partial_y\tau_{yy}}_{0} +
    \partial_z\tau_{yz}  -  \rho \partial_t^2 u_y = -f_y

Substituting the stress formula in the above equation yields

.. math::

    \partial_x\mu\partial_x u_y  + \partial_z\mu\partial_z u_y
    - \rho\partial_t^2 u_y = -f_y

which is the wave equation for horizontally polarized S waves, i.e. SH waves.
This equation is solved here using the Equivalent Staggered Grid (ESG) method
of Di Bartolo et al. (2012).
This method was developed for acoustic (pressure) waves but can be applied
without modification to SH waves.

Canceling the y derivatives in the first and third components of the equation
of motion

.. math::

    \partial_x\tau_{xx} + \partial_z\tau_{xz} - \rho \partial_t^2 u_x = -f_x

.. math::

    \partial_x\tau_{xz} + \partial_z\tau_{zz} - \rho \partial_t^2 u_z = -f_z

And the corresponding stress components are

.. math::

    \tau_{xx} = (\lambda + 2\mu)\partial_x u_x + \lambda\partial_z u_z

.. math::

    \tau_{zz} = (\lambda + 2\mu)\partial_z u_z + \lambda\partial_x u_x

.. math::

    \tau_{xz} = \mu( \partial_x u_z + \partial_z u_x)

This means that the displacements in x and z are coupled and must be solved
together.
This equation describes the motion of pressure (P) and vertically polarized S
waves (SV).
The method used here to solve these equations is the Parsimonious Staggered
Grid (PSG) of Luo and Schuster (1990).


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
import scipy.sparse
import scipy.sparse.linalg

try:
    from fatiando.seismic._wavefd import *
except:
    def not_implemented(*args, **kwargs):
        raise NotImplementedError(
        "Couldn't load C coded extension module.")
    _apply_damping = not_implemented
    _step_elastic_sh = not_implemented
    _step_elastic_psv = not_implemented
    _xz2ps = not_implemented
    _nonreflexive_sh_boundary_conditions = not_implemented
    _nonreflexive_psv_boundary_conditions = not_implemented
    _step_scalar = not_implemented
    _reflexive_scalar_boundary_conditions = not_implemented

class MexHatSource(object):
    r"""
    A wave source that vibrates as a Mexican hat (Ricker) wavelet.

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

        .. note:: If you want the source to start with amplitude close to 0, use
            ``delay = 3.5/frequency``.

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

    This source vibrates for a time equal to one period (T).

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
    * frequency : float
        The frequency of the source
    * delay : float
        The delay before the source starts

        .. note:: If you want the source to start with amplitude close to 0,
            use ``delay = 3.5/frequency``.

    """

    def __init__(self, x, z, area, shape, amp, frequency, delay=0):
        super(SinSqrSource, self).__init__(x, z, area, shape, amp,
                                           frequency, delay)
        self.wlength = 1./frequency

    def __call__(self, time):
        t = time - self.delay
        if t + self.delay > self.wlength:
            return 0
        psi = self.amp*numpy.sin(2.*numpy.pi*t/float(self.wlength))**2
        return psi

def blast_source(x, z, area, shape, amp, frequency, delay=0,
    sourcetype=MexHatSource):
    """
    Uses several MexHatSources to create a blast source that pushes in all
    directions.

    Parameters:

    * x, z : float
        The x, z coordinates of the source
    * area : [xmin, xmax, zmin, zmax]
        The area bounding the finite difference simulation
    * shape : (nz, nx)
        The number of nodes in the finite difference grid
    * amp : float
        The amplitude of the source
    * frequency : float
        The frequency of the source
    * delay : float
        The delay before the source starts
    * sourcetype : source class
        The type of source to use, like
        :class:`~fatiando.seismic.wavefd.MexHatSource`.

    Returns:

    * [xsources, zsources]
        Lists of sources for x- and z-displacements

    """
    nz, nx = shape
    xsources, zsources = [], []
    center = sourcetype(x, z, area, shape, amp, frequency, delay)
    i, j = center.indexes()
    tmp = numpy.sqrt(2)
    locations = [[i-1,j-1,-amp,-amp], [i-1,j,0,-tmp*amp], [i-1,j+1,amp,-amp],
                 [i,j-1,-tmp*amp,0], [i,j+1,tmp*amp,0],
                 [i+1,j-1,-amp,amp], [i+1,j,0,tmp*amp], [i+1,j+1,amp,amp]]
    locations = [[i, j, xamp, zamp] for i, j, xamp, zamp in locations
                 if i >= 0 and i < nz and j >= 0 and j < nx]
    for i, j, xamp, zamp in locations:
        xsrc = sourcetype(x, z, area, shape, xamp, frequency, delay)
        xsrc.i, xsrc.j = i, j
        zsrc = sourcetype(x, z, area, shape, zamp, frequency, delay)
        zsrc.i, zsrc.j = i, j
        xsources.append(xsrc)
        zsources.append(zsrc)
    return xsources, zsources

class GaussSource(MexHatSource):
    r"""
    A wave source that vibrates as a Gaussian derivative wavelet.

    .. math::

        \psi(t) = A 2 \sqrt{e}\ f\ t\ e^\left(-2t^2f^2\right)

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
        The approximate frequency of the source
    * delay : float
        The delay before the source starts

    .. note:: If you want the source to start with amplitude close to 0,
        use ``delay = 3.0/frequency``.
    """

    def __init__(self, x, z, area, shape, amp, frequency, delay=None):
        super(GaussSource, self).__init__(x, z, area, shape, amp,
                                           frequency, delay)
        if (delay == None):
            self.delay = 3.0/frequency

    def __call__(self, time):
        t = time - self.delay
        psi = self.amp*((2*numpy.sqrt(numpy.e)*self.frequency)
               *t*numpy.exp(-2*(t**2)*self.f2)
            )
        return psi

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
    lamb = dens*pvel**2 - 2*dens*svel**2
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
    mu = dens*svel**2
    return mu

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

def scalar(vel, area, dt, iterations, sources, stations=None,
    snapshot=None, padding=50, taper=0.005):
    """

    Simulate scalar waves using an explicit finite differences scheme 4th order
    space. Space increment must be equal in x and z.

    Parameters:

    * vel : 2D-array (defines shape simulation)
        The wave velocity at all the grid nodes
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
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
        indexes
    * snapshot : None or int
        If not None, than yield a snapshot of the displacement at every
        *snapshot* iterations.
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the Gaussian taper function used for the absorbing
        boundary conditions

    Yields:

    * i, u, seismograms : int, 2D-array and list of 1D-arrays
        The current iteration, the scalar quantity disturbed
        and a list of the scalar quantity disturbed recorded at each
        station until the current iteration.

    """

    nz, nx = numpy.shape(vel) # get simulation dimensions
    x1, x2, z1, z2 = area
    dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)

    if dz != dx:
        raise ValueError('Space increment must be equal in x and z')

    ds = dz # dz or dx doesn't matter

    # Get the index of the closest point to the stations and start the
    # seismograms
    if stations is not None:
        stations = [[int(round((z - z1)/ds)), int(round((x - x1)/ds))]
                    for x, z in stations]
        seismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
    else:
        stations, seismograms = [], []
    # Add some padding to x and z. The padding region is where the wave is
    # absorbed
    pad = int(padding)
    nx += 2*pad
    nz += pad
    # Pad the velocity as well
    vel_pad = _add_pad(vel, pad, (nz, nx))
    # Pack the particle position u at 2 different times in one 3d array
    # u[0] = u(t-1)
    # u[1] = u(t)
    # The next time step overwrites the t-1 panel
    u = numpy.zeros((2, nz, nx), dtype=numpy.float)
    # Compute and yield the initial solutions
    for src in sources:
        i, j = src.indexes()
        u[1, i, j + pad] += -((vel[i,j]*dt)**2)*src(0)
    # Update seismograms
    for station, seismogram in zip(stations, seismograms):
        i, j = station
        seismogram[0] = u[1, i, j + pad]
    if snapshot is not None:
        yield 0, u[1, :-pad, pad:-pad], seismograms

    for iteration in xrange(1, iterations):
        t, tm1 = iteration%2, (iteration + 1)%2
        tp1 = tm1
        _step_scalar(u[tp1], u[t], u[tm1], 2, nx - 2, 2, nz - 2,
                     dt, ds, vel_pad)
        # forth order +2-2 indexes needed
        # Damp the regions in the padding to make waves go to infinity
        _apply_damping(u[t], nx, nz, pad, taper)
        # not PML yet or anything similar
        _reflexive_scalar_boundary_conditions(u[tp1], nx, nz)
        # Damp the regions in the padding to make waves go to infinity
        _apply_damping(u[tp1], nx, nz, pad, taper)
        for src in sources:
            i, j = src.indexes()
            u[tp1, i, j + pad] += -((vel[i,j]*dt)**2)*src(iteration*dt)
        # Update seismograms
        for station, seismogram in zip(stations, seismograms):
            i, j = station
            seismogram[iteration] = u[tp1, i, j + pad]
        if snapshot is not None and iteration%snapshot == 0:
            yield iteration, u[tp1, :-pad, pad:-pad], seismograms
    yield iteration, u[tp1, :-pad, pad:-pad], seismograms

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

    Uses absorbing boundary conditions (Gaussian taper) in the lower, left and
    right boundaries. The top implements a free-surface boundary condition.

    Parameters:

    * mu : 2D-array (shape = *shape*)
        The :math:`\mu` Lame parameter at all the grid nodes
    * density : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
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
        indexes
    * snapshot : None or int
        If not None, than yield a snapshot of the displacement at every
        *snapshot* iterations.
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the Gaussian taper function used for the absorbing
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

def elastic_psv(mu, lamb, density, area, dt, iterations, sources,
    stations=None, snapshot=None, padding=50, taper=0.002, xz2ps=False):
    """
    Simulate P and SV waves using the Parsimonious Staggered Grid (PSG) finite
    differences scheme of Luo and Schuster (1990).

    This is an iterator. It yields panels of $u_x$ and $u_z$ displacements
    and a list of arrays with recorded displacements in a time series.
    Parameter *snapshot* controls how often the iterator yields. The default
    is only at the end, so only the final panel and full time series are
    yielded.

    Uses absorbing boundary conditions (Gaussian taper) in the lower, left and
    right boundaries. The top implements the free-surface boundary condition
    of Vidale and Clayton (1986).

    Parameters:

    * mu : 2D-array (shape = *shape*)
        The :math:`\mu` Lame parameter at all the grid nodes
    * lamb : 2D-array (shape = *shape*)
        The :math:`\lambda` Lame parameter at all the grid nodes
    * density : 2D-array (shape = *shape*)
        The value of the density at all the grid nodes
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.
    * dt : float
        The time interval between iterations
    * iterations : int
        Number of time steps to take
    * sources : [xsources, zsources] : lists
        A lists of the sources of waves for the particle movement in the x and
        z directions
        (see :class:`~fatiando.seismic.wavefd.MexHatSource` for an example
        source)
    * stations : None or list
        If not None, then a list of [x, z] pairs with the x and z coordinates
        of the recording stations. These are physical coordinates, not the
        indexes!
    * snapshot : None or int
        If not None, than yield a snapshot of the displacements at every
        *snapshot* iterations.
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the Gaussian taper function used for the absorbing
        boundary conditions
    * xz2ps : True or False
        If True, will yield P and S wave panels instead of ux, uz. See
        :func:`~fatiando.seismic.wavefd.xz2ps`.

    Yields:

    * [t, ux, uz, xseismograms, zseismograms]
        The current iteration, the particle displacements in the x and z
        directions, lists of arrays containing the displacements recorded at
        each station until the current iteration.

    References:

    Vidale, J. E., and R. W. Clayton (1986), A stable free-surface boundary
    condition for two-dimensional elastic finite-difference wave simulation,
    Geophysics, 51(12), 2247-2249.

    """
    if mu.shape != lamb.shape != density.shape:
        raise ValueError('Density lambda, and mu grids should have same shape')
    x1, x2, z1, z2 = area
    nz, nx = mu.shape
    dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)
    xsources, zsources = sources
    # Get the index of the closest point to the stations and start the
    # seismograms
    if stations is not None:
        stations = [[int(round((z - z1)/dz)), int(round((x - x1)/dx))]
                    for x, z in stations]
        xseismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
        zseismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
    else:
        stations, xseismograms, zseismograms = [], [], []
    # Add padding to have an absorbing region to simulate an infinite medium
    pad = int(padding)
    nx += 2*pad
    nz += pad
    mu_pad = _add_pad(mu, pad, (nz, nx))
    lamb_pad = _add_pad(lamb, pad, (nz, nx))
    dens_pad = _add_pad(density, pad, (nz, nx))
    # Pre-compute the matrices required for the free-surface boundary
    dzdx = dz/dx
    identity = scipy.sparse.identity(nx)
    B = scipy.sparse.eye(nx, nx, k=1) - scipy.sparse.eye(nx, nx, k=-1)
    gamma = scipy.sparse.spdiags(lamb_pad[0]/(lamb_pad[0] + 2*mu_pad[0]), [0],
                                 nx, nx)
    Mx1 = identity - 0.0625*(dzdx**2)*B*gamma*B
    Mx2 = identity + 0.0625*(dzdx**2)*B*gamma*B
    Mx3 = 0.5*dzdx*B
    Mz1 = identity - 0.0625*(dzdx**2)*gamma*B*B
    Mz2 = identity + 0.0625*(dzdx**2)*gamma*B*B
    Mz3 = 0.5*dzdx*gamma*B
    # Compute and yield the initial solutions
    ux = numpy.zeros((2, nz, nx), dtype=numpy.float)
    uz = numpy.zeros((2, nz, nx), dtype=numpy.float)
    if xz2ps:
        p, s = numpy.empty_like(mu), numpy.empty_like(mu)
    for src in xsources:
        i, j = src.indexes()
        ux[1, i, j + pad] += (dt**2/density[i, j])*src(0)
    for src in zsources:
        i, j = src.indexes()
        uz[1, i, j + pad] += (dt**2/density[i, j])*src(0)
    # Update seismograms
    for station, xseis, zseis in zip(stations, xseismograms, zseismograms):
        i, j = station
        xseis[0] = ux[1, i, j + pad]
        zseis[0] = uz[1, i, j + pad]
    if snapshot is not None:
        if xz2ps:
            _xz2ps(ux[1, :-pad, pad:-pad], uz[1, :-pad, pad:-pad], p, s,
                   p.shape[1], p.shape[0], dx, dz)
            yield [0, p, s, xseismograms, zseismograms]
        else:
            yield [0, ux[1, :-pad, pad:-pad], uz[1, :-pad, pad:-pad],
                   xseismograms, zseismograms]
    for iteration in xrange(1, iterations):
        t, tm1 = iteration%2, (iteration + 1)%2
        tp1 = tm1
        _step_elastic_psv(ux, uz, tp1, t, tm1, 1, nx - 1,  1, nz - 1, dt, dx,
            dz, mu_pad, lamb_pad, dens_pad)
        _apply_damping(ux[t], nx, nz, pad, taper)
        _apply_damping(uz[t], nx, nz, pad, taper)
        # Free-surface boundary conditions
        ux[tp1,0,:] = scipy.sparse.linalg.spsolve(Mx1,
                      Mx2*ux[tp1,1,:] + Mx3*uz[tp1,1,:])
        uz[tp1,0,:] = scipy.sparse.linalg.spsolve(Mz1,
                      Mz2*uz[tp1,1,:] + Mz3*ux[tp1,1,:])
        _nonreflexive_psv_boundary_conditions(ux, uz, tp1, t, tm1, nx, nz, dt,
            dx, dz, mu_pad, lamb_pad, dens_pad)
        _apply_damping(ux[tp1], nx, nz, pad, taper)
        _apply_damping(uz[tp1], nx, nz, pad, taper)
        for src in xsources:
            i, j = src.indexes()
            ux[tp1, i, j + pad] += (dt**2/density[i, j])*src(iteration*dt)
        for src in zsources:
            i, j = src.indexes()
            uz[tp1, i, j + pad] += (dt**2/density[i, j])*src(iteration*dt)
        for station, xseis, zseis in zip(stations, xseismograms, zseismograms):
            i, j = station
            xseis[iteration] = ux[tp1, i, j + pad]
            zseis[iteration] = uz[tp1, i, j + pad]
        if snapshot is not None and iteration%snapshot == 0:
            if xz2ps:
                _xz2ps(ux[tp1, :-pad, pad:-pad], uz[tp1, :-pad, pad:-pad], p,
                       s, p.shape[1], p.shape[0], dx, dz)
                yield [iteration, p, s, xseismograms, zseismograms]
            else:
                yield [iteration, ux[tp1, :-pad, pad:-pad],
                       uz[tp1, :-pad, pad:-pad], xseismograms, zseismograms]
    if xz2ps:
        _xz2ps(ux[tp1, :-pad, pad:-pad], uz[tp1, :-pad, pad:-pad], p,
               s, p.shape[1], p.shape[0], dx, dz)
        yield [iteration, p, s, xseismograms, zseismograms]
    else:
        yield [iteration, ux[tp1, :-pad, pad:-pad], uz[tp1, :-pad, pad:-pad],
               xseismograms, zseismograms]

def xz2ps(ux, uz, area):
    r"""
    Convert the x and z displacements into representations of P and S waves
    using the divergence and curl, respectively, after Kennett (2002, pp 57)

    .. math::

        P: \frac{\partial u_x}{\partial x} + \frac{\partial u_z}{\partial z}

    .. math::

        S: \frac{\partial u_x}{\partial z} - \frac{\partial u_z}{\partial x}

    Parameters:

    * ux, uz : 2D-arrays
        The x and z displacement panels
    * area : [xmin, xmax, zmin, zmax]
        The x, z limits of the simulation area, e.g., the shallowest point is
        at zmin, the deepest at zmax.

    Returns:

    * p, s : 2D-arrays
        Panels corresponding to P and S wave components


    References:

    Kennett, B. L. N. (2002), The Seismic Wavefield: Volume 2, Interpretation
    of Seismograms on Regional and Global Scales, Cambridge University Press.

    """
    if ux.shape != uz.shape:
        raise ValueError('ux and uz grids should have same shape')
    x1, x2, z1, z2 = area
    nz, nx = ux.shape
    dz, dx = (z2 - z1)/(nz - 1), (x2 - x1)/(nx - 1)
    p, s = numpy.empty_like(ux), numpy.empty_like(ux)
    _xz2ps(ux, uz, p, s, nx, nz, dx, dz)
    return p, s

def maxdt(area, shape, maxvel):
    """
    Calculate the maximum time step that can be used in the simulation.

    Uses the result of the Von Neumann type analysis of Di Bartolo et al.
    (2012).

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
    spacing = min([(x2 - x1)/(nx - 1), (z2 - z1)/(nz - 1)])
    return 0.606*spacing/maxvel


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
    spacing = min([(x2 - x1)/(nx - 1), (z2 - z1)/(nz - 1)])
    factor = numpy.sqrt(3./8.)
    factor -= factor/100. # 1% smaller to guarantee criteria
    # the closer to stability criteria the better the convergence
    return factor*spacing/maxvel
