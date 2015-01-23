r"""
Finite difference solution of the 2D wave equation for isotropic media.

Solutions are implemented as simulation classes taking advantage of Object
Orientation. They have Rich Display features of IPython Notebook and use
the HDF5 file format to store and persist the simulation objects.

**Sources**

* :class:`~fatiando.seismic.wavefd.Ricker`: Mexican hat (Ricker) wavelet source.
* :class:`~fatiando.seismic.wavefd.Gauss`: Gauss derivative wavelet source

**Simulation Classes**

* :class:`~fatiando.seismic.wavefd.ElasticSH`: Simulates SH elastic waves using
  the Equivalent Staggered Grid method of Di Bartolo et al. (2012)
* :class:`~fatiando.seismic.wavefd.ElasticPSV`: Simulates the coupled P and SV
  elastic waves using the Parsimonious Staggered Grid method of Luo and
  Schuster (1990)
* :class:`~fatiando.seismic.wavefd.Scalar`: Simulates scalar waves using simple
  explicit finite differences scheme

**Simulation Storage Access**

The simulation classes store each time frame in an HDF5 file (the cache file).
All data needed to resume the simulation later are stored. So when you need the
simulation result, the class can dig into the cache file and return the data
requested using python slice.

All simulation classes have the following methods:

* :func:`from_cache`: rebuilds a simulation object from the cache file and
  computations can be resumed right away, Source functions are also stored
  in the cache file as pickles.

* slicing : ... to finish ...

**Simulation Rich display**

Simulation and  wavelet classes have rich display capabilities
through IPython notebook features. There is an option to
convert the animation to a video and embed it in the IPython notebook.

All simulation classes have the following methods:
(They work on IPython notebook only!)

* :func:`animate`: ... to finish...

* :func:`explore`:

* :func:`snapshot`: loads a time step frame from the cache file and plots it.

* :func:`_repr_png`:

**Auxiliary functions** what to do with this? ... cleaning up

* :func:`~fatiando.seismic.wavefd.lame_lamb`: Calculate the lambda Lame
  parameter
* :func:`~fatiando.seismic.wavefd.lame_mu`: Calculate the mu Lame parameter
* :func:`~fatiando.seismic.wavefd.xz2ps`: Convert x and z displacements to
  representations of P and S waves
* :func:`~fatiando.seismic.wavefd.maxdt`: Calculate the maximum time step for
  elastic wave simulations

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
from tempfile import NamedTemporaryFile
import time
import sys
import os
import cPickle as pickle
from abc import ABCMeta, abstractmethod
import six
import numpy
import scipy.sparse
import scipy.sparse.linalg
from IPython.display import Image, HTML, display
from IPython.html import widgets
from IPython.core.pylabtools import print_figure
from matplotlib import animation
from matplotlib import pyplot as plt
import h5py

try:
    from ._wavefd import *
except:
    def not_implemented():
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


class Source(six.with_metaclass(ABCMeta)):
    """
    Base class for describing seismic sources.

    Call an instance as a function with a given time to get back the source
    function at that time.

    Implements a `_repr_png_` method that plots the source function around the
    delay time.

    Overloads multiplication by a scalar to multiply the amplitude of the
    source and return a new source.
    """
    def __init__(self, amp, cf, delay):
        self.amp = amp
        self.cf = cf
        self.delay = delay

    def __mul__(self, scalar):
        return self.__class__(self.amp*scalar, self.cf, self.delay)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def _repr_png_(self):
        t = self.delay + numpy.linspace(-2/self.cf, 2/self.cf, 200)
        fig = plt.figure(figsize=(6, 4), facecolor='white')
        fig.set_figheight(0.5*fig.get_figwidth())
        plt.title('{} wavelet'.format(self.__class__.__name__))
        plt.plot(t, self(t), '-k')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.xlim(t.min(), t.max())
        plt.grid(True)
        plt.tight_layout()
        data = print_figure(fig, dpi=70)
        plt.close(fig)
        return data

    def __call__(self, t):
        """
        Return the source function value at a given time

        Parameters:

        * t: float
            time value where evaluate the source function
        """
        return self.value(t)

    @abstractmethod
    def value(self, t):
        """
        Return the source function value at a given time

        Parameters:

        * t: float
            Time value where evaluate the source function

        Returns:

        * value : float or numpy array
            The value of Source(t)

        """
        pass


class Ricker(Source):
    r"""
    A wave source that vibrates as a Mexican hat (Ricker) wavelet.

    .. math::

        \psi(t) = A(1 - 2 \pi^2 f^2 t^2)exp(-\pi^2 f^2 t^2)

    .. note:: If you want the source to start with amplitude close to 0,
                use ``delay = 3.5/frequency``

    Parameters:

    * amp : float
        The amplitude of the source (:math:`A`)
    * cf : float
        The peak frequency of the wavelet
    * delay : float
        The delay before the source starts
    """

    def __init__(self, amp, cf, delay=0):
        super(Ricker, self).__init__(amp, cf, delay)

    def value(self, t):
        """
        Return the source function value at a given time

        Parameters:

        * t: float
            Time value where evaluate the source function

        Returns:

        * value : float or numpy array
            The value of Source(t)

        """
        t = (t - self.delay)
        aux = self.amp*(1 - 2*(numpy.pi*self.cf*t)**2)
        return aux*numpy.exp(-(numpy.pi*self.cf*t)**2)


class Gauss(Source):
    r"""
    A wave source that vibrates as a Gaussian derivative wavelet.

    .. math::

        \psi(t) = A 2 \sqrt{e}\ f\ t\ e^\left(-2t^2f^2\right)

    .. note:: If you want the source to start with amplitude close to 0,
            use ``delay = 3.0/frequency``.

    Parameters:

    * amp : float
        The amplitude of the source (:math:`A`)
    * cf : float
        The peak frequency of the wavelet
    * delay : float
        The delay before the source starts
    """

    def __init__(self, amp, cf, delay=None):
        super(Gauss, self).__init__(amp, cf, delay)
        self.f2 = self.cf**2
        if delay is None:
            self.delay = 3.0/self.cf


    def value(self, t):
        """
        Return the source function value at a given time

        Parameters:

        * t: float
            Time value where evaluate the source function

        Returns:

        * value : float or numpy array
            The value of Source(t)

        """
        t = (t - self.delay)
        psi = self.amp * ((2 * numpy.sqrt(numpy.e) * self.cf)
                          * t * numpy.exp(-2 * (t ** 2) * self.f2)
                          )
        return psi


class WaveFD2D(six.with_metaclass(ABCMeta)):
    """
    Base class for 2D simulations.

    Implements the ``run`` method and delegates actual _timestepping to the
    abstract ``_timestep`` method.

    Handles creating an HDF5 cache file, plotting snapshots of the simulation,
    printing a progress bar to stderr, and creating an IPython widget to
    explore the snapshots.

    Overloads ``__getitem__``. Indexing the simulation object is like
    indexing the HDF5 cache file. This way you can treat the simulation
    object as a numpy array.

    """

    def __init__(self, cachefile, spacing, shape, dt=None,
                 padding=50, taper=0.007, verbose=True):
        if numpy.size(spacing) == 1:  # equal space increment in x and z
            self.dx, self.dz = spacing, spacing
        else:
            self.dx, self.dz = spacing
        self.shape = shape  # grid shape without padding
        self.set_verbose(verbose)
        self.sources = []
        # simsize stores the total size of this simulation
        # after some or many runs
        self.simsize = 0  # simulation number of interations already ran
        # it is the `run` iteration time step indexer
        self.it = -1  # iteration time step index (where we are)
        # `it` and `simsize` together allows indefinite simulation runs
        if cachefile is None:
            cachefile = self._create_tmp_cache()
        self.cachefile = cachefile
        self.padding = padding # padding region size
        self.taper = taper
        self.dt = dt

    def _create_tmp_cache(self):
        """
        Creates the temporary file used
        to store data in hdf5 format

        Returns:

        * _create_tmp_cache: str
            returns the name of the file created

        """
        tmpfile = NamedTemporaryFile(
            suffix='.h5',
            prefix='{}-'.format(self.__class__.__name__),
            dir=os.path.curdir,
            delete=False)
        fname = tmpfile.name
        tmpfile.close()
        return fname

    @abstractmethod
    def from_cache(fname, verbose=True):
        pass

    @abstractmethod
    def _init_panels(self):
        pass

    @abstractmethod
    def _init_cache(self, npanels, chunks=None, compression='lzf', shuffle=True):
        pass

    @abstractmethod
    def _expand_cache(self, npanels):
        pass

    @abstractmethod
    def _cache_panels(self, npanels, tp1, iteration, simul_size):
        pass

    def _get_cache(self, mode='r'):
        """
        Get the cache file as h5py file object

        Parameters:

        * mode: str
            'r' or 'w'
            for reading or writing

        Returns:

        * cache : h5py file object

        """
        return h5py.File(self.cachefile, mode)

    def set_verbose(self, verbose):
        """
        Whether to show or not progress bar

        Parameters:

        * verbose : bool
            True shows progress bar
        """
        self.verbose = verbose
        # Need an option to get rid of the sys.stderr reference because it
        # can't be pickled.
        if verbose:
            self.stream = sys.stderr
        else:
            self.stream = None

    def __getitem__(self, index):
        """
        Get an iteration of the panels object from the hdf5 cache file.
        """
        pass


    @abstractmethod
    def _plot_snapshot(self, frame, **kwargs):
        pass

    def snapshot(self, frame, embed=False, raw=False, ax=None, **kwargs):
        """
        Returns an image (snapshot) of the 2D wavefield simulation
        at a desired iteration number (frame).

        Parameters:

        * frame : int
            The time step iteration number
        * embed : bool
            True to plot it inline
        * raw: bool
            True for raw byte image

        Returns:

        * image:
            raw byte image if raw=True
            jpeg picture if embed=True

        """
        # calls `_plot_snapshot`
        if ax is None:
            fig = plt.figure(facecolor='white')
            ax = plt.subplot(111)
        if frame < 0:
            title = self.simsize + frame
        else:
            title = frame
        plt.sca(ax)
        fig = ax.get_figure()
        plt.title('Time frame {:d}'.format(title))
        self._plot_snapshot(frame, **kwargs)
        nz, nx = self.shape
        mx, mz = nx*self.dx, nz*self.dz
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mz)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.invert_yaxis()
        # Check the aspect ratio of the plot and adjust figure size to match
        aspect = min(self.shape)/max(self.shape)
        try:
            aspect /= ax.get_aspect()
        except TypeError:
            pass
        if nx > nz:
            width = 10
            height = width*aspect*0.8
        else:
            height = 8
            width = height*aspect*1.5
        fig.set_size_inches(width, height)
        plt.tight_layout()
        if raw or embed:
            png = print_figure(fig, dpi=70)
            plt.close(fig)
        if raw:
            return png
        elif embed:
            return Image(png)

    def _repr_png_(self):
        """
        Display one time frame of this simulation
        """
        return self.snapshot(-1, raw=True)

    def explore(self, **kwargs):
        """
        Interactive visualization of simulation results.
        Allows to move back and forth on simulation frames.
        """
        plotargs = kwargs
        def plot(Frame):
            image = Image(self.snapshot(Frame, raw=True, **plotargs))
            display(image)
            return image
        slider = widgets.IntSliderWidget(min=0, max=self.it, step=1,
                                         value=self.it, description="Frame")
        widget = widgets.interact(plot, Frame=slider)
        return widget

    @abstractmethod
    def _timestep(self, panels, tm1, t, tp1, iteration):
        pass

    def run(self, iterations):
        """
        Run this simulation given the number of iterations.

        * iterations: int
            number of time step iterations to run
        """
        # calls `_init_cache`, `_expand_cache`, `_init_panels`
        # and `_cache_panels` and  `_time_step`

        nz, nx = self.shape
        dz, dx = self.dz, self.dx
        u = self._init_panels()  # panels must be created first

        # Initialize the cache on the first run
        if self.simsize == 0:
            self._init_cache(iterations)
        else:   # increase cache size by iterations
            self._expand_cache(iterations)

        if self.verbose:
            # The size of the progress status bar
            places = 50
            self.stream.write(''.join(['|', '-'*places, '|', '  0%']))
            self.stream.flush()
            nprinted = 0
            start_time = time.clock()
        for iteration in xrange(iterations):
            t, tm1 = iteration%2, (iteration + 1)%2
            tp1 = tm1
            self.it += 1
            self._timestep(u, tm1, t, tp1, self.it)
            self.simsize += 1
            #  won't this make it slower than it should? I/O
            self._cache_panels(u, tp1, self.it, self.simsize)
            # Update the status bar
            if self.verbose:
                percent = int(round(100*(iteration + 1)/iterations))
                n = int(round(0.01*percent*places))
                if n > nprinted:
                    self.stream.write(''.join(['\r|', '#'*n, '-'*(places - n),
                                               '|', '%3d%s' % (percent, '%')]))
                    self.stream.flush()
                    nprinted = n
        # Make sure the progress bar ends in 100 percent
        if self.verbose:
            self.stream.write(''.join(
                ['\r|', '#'*places, '|', '100%',
                 ' Ran {:d} iterations in {:g} seconds.'.format(
                     iterations, time.clock() - start_time)]))
            self.stream.flush()

    def animate(self, every=1, cutoff=None, ax=None, cmap=plt.cm.seismic,
                embed=False, fps=10, dpi=70, writer='avconv', **kwargs):
        """
        Creates a 2D animation from all the simulation iterations
        that has been run.

        * every : int

        * cutoff : int

        * ax : int

        * cmap : int

        * embed:

        """
        if ax is None:
            plt.figure(facecolor='white')
            ax = plt.subplot(111)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        fig = ax.get_figure()
        nz, nx = self.shape
        # Check the aspect ratio of the plot and adjust figure size to match
        aspect = min(self.shape)/max(self.shape)
        try:
            aspect /= ax.get_aspect()
        except TypeError:
            pass
        if nx > nz:
            width = 10
            height = width*aspect*0.8
        else:
            height = 10
            width = height*aspect*1.5
        fig.set_size_inches(width, height)
        # Separate the arguments for imshow
        imshow_args = dict(cmap=cmap)
        if cutoff is not None:
            imshow_args['vmin'] = -cutoff
            imshow_args['vmax'] = cutoff
        wavefield = ax.imshow(numpy.zeros(self.shape), **imshow_args)
        fig.colorbar(wavefield, pad=0, aspect=30).set_label('Displacement')
        ax.set_title('iteration: 0')
        frames = self.simsize//every
        def plot(i):
            ax.set_title('iteration: {:d}'.format(i*every))
            u = self[i*every]
            wavefield.set_array(u)
            return wavefield
        anim = animation.FuncAnimation(fig, plot, frames=frames, **kwargs)
        if embed:
            return anim_to_html(anim, fps=fps, dpi=dpi, writer=writer)
        else:
            plt.show()
            return anim


class ElasticSH(WaveFD2D):
    """
    Simulate SH waves using the Equivalent Staggered Grid (ESG) finite
    differences scheme of Di Bartolo et al. (2012).

    Uses absorbing boundary conditions (Gaussian taper) in the lower, left and
    right boundaries. The top implements a free-surface boundary condition.

    Parameters:

    * velocity: 2D-array (defines shape simulation)
        The wave velocity at all the grid nodes
    * density: 2D-array
        The medium density
    * spacing: (dx, dz)
        space increment for x and z direction
    * cachefile: str
        The hdf5 cachefile file path to store the simulation
    * dt: float
        time increment for simulation
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the Gaussian taper function used for the absorbing
        boundary conditions
    * verbose: bool
        True to show simulation progress bar
    """

    def __init__(self, velocity, density, spacing, cachefile=None, dt=None,
                 padding=50, taper=0.007, verbose=True):
        super(ElasticSH, self).__init__(cachefile, spacing, velocity.shape, dt, padding,
                                        taper, verbose)
        self.density = density
        self.velocity = velocity
        self.mu = lame_mu(velocity, density)
        if self.dt is None:
            self.dt = self.maxdt()

    def __getitem__(self, index):
        """
        Get an iteration of the panels object from the hdf5 cache file.

        .. note:: panels object is a tuple or variable containing all 2D panels needed for this simulation

        * index: index or slicing
            index for slicing hdf5 data set

        Returns:

        * panels object : 2D panels object at index

        """
        with self._get_cache() as f:
            data = f['panels'][index]
        return data

    @staticmethod
    def from_cache(fname, verbose=True):
        """
        Creates a simulation object from a pre-existing HDF5 file

        * fname: str
            HDF5 file path containing a previous simulation stored

        * verbose: bool
            Progress status shown or not
        """
        with h5py.File(fname, 'r') as f:
            vel = f['velocity']
            dens = f['density']
            panels = f['panels']
            dx = panels.attrs['dx']
            dz = panels.attrs['dz']
            dt = panels.attrs['dt']
            padding = panels.attrs['padding']
            taper = panels.attrs['taper']
            sim = ElasticSH(vel[:], dens[:], (dx, dz), dt=dt, padding=padding,
                            taper=taper, cachefile=fname)
            sim.simsize = panels.attrs['simsize']
            sim.it = panels.attrs['iteration']
            sim.sources = pickle.loads(f['sources'].value.tostring())
        sim.set_verbose(verbose)
        return sim

    def _init_cache(self, npanels, chunks=None, compression='lzf', shuffle=True):
        """
        Init the hdf5 cache file with this simulation parameters

        * npanels: int
            number of 2D panels needed for this simulation run
        *  chunks : HDF5 data set option

        * compression: HDF5 data set option

        * shuffle: HDF5 data set option

        """
        nz, nx = self.shape
        if chunks is None:
            chunks = (1, nz//10, nx//10)
        with self._get_cache(mode='w') as f:  # create HDF5 data sets
            nz, nx = self.shape
            dset = f.create_dataset('panels', (npanels, nz, nx),
                                     maxshape=(None, nz, nx),
                                     chunks=chunks,
                                     compression=compression,
                                     shuffle=shuffle,
                                     dtype=numpy.float)
            dset.attrs['shape'] = self.shape
            dset.attrs['simsize'] = self.simsize
            dset.attrs['iteration'] = self.it
            dset.attrs['dx'] = self.dx
            dset.attrs['dz'] = self.dz
            dset.attrs['dt'] = self.dt
            dset.attrs['padding'] = self.padding
            dset.attrs['taper'] = self.taper
            f.create_dataset('velocity', data=self.velocity)
            f.create_dataset('density', data=self.density)
            f.create_dataset(
                'sources', data=numpy.void(pickle.dumps(self.sources)))

    def _expand_cache(self, npanels):
        """
        Expand the hdf5 cache file of this simulation parameters
        for more iterations

        *  npanels: int
            number of 2D panels needed for this simulation run
        """
        with self._get_cache(mode='a') as f:
            cache = f['panels']
            cache.resize(self.simsize + npanels, axis=0)

    def _cache_panels(self, u, tp1, iteration, simul_size):
        """
        Save the last calculated panels and information about it
        in the hdf5 cache file

        Parameters:

        * panels : tuple or variable
            tuple or variable containing all 2D panels needed for this simulation
        * tp1 : int
            panel time index
        * iteration:
            iteration number
        * simul_size:
            number of iterations that has been run
        """
        # Save the panel to disk
        with self._get_cache(mode='a') as f:
            cache = f['panels']
            cache[simul_size - 1] = u[tp1]
            # is this really needed? it's already saved when initiated or expanded??
            cache.attrs['simsize'] = simul_size
            # I need to update the attribute with this iteration number
            cache.attrs['iteration'] = iteration
            # that simulation runs properly after reloaded from file

    def _init_panels(self):
        """
        Start the simulation panels used for finite difference solution.
        Keep consistency of simulations if loaded from file.

        Returns:

        * return:
            panels object

        """
        # If this is the first run, start with zeros, else, get the last two
        # panels from the cache so that the simulation can be resumed
        if self.simsize == 0:
            nz, nx = self.shape
            u = numpy.zeros((2, nz, nx), dtype=numpy.float)
        else:
            with self._get_cache() as f:
                cache = f['panels']
                u = cache[self.simsize - 2 : self.simsize][::-1]
        return u

    def add_point_source(self, position, wavelet):
        """"
        Adds a point source to this simulation

        Parameters:

        * position : tuple
            The (z, x) coordinates of the source
        * source : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an example source)

        """
        self.sources.append([position, wavelet])

    def _timestep(self, u, tm1, t, tp1, iteration):
        nz, nx = self.shape
        _step_elastic_sh(u[tp1], u[t], u[tm1], 3, nx - 3, 3, nz - 3,
                         self.dt, self.dx, self.dz, self.mu, self.density)
        _apply_damping(u[t], nx, nz, self.padding, self.taper)
        _nonreflexive_sh_boundary_conditions(
            u[tp1], u[t], nx, nz, self.dt, self.dx, self.dz, self.mu, self.density)
        _apply_damping(u[tp1], nx, nz, self.padding, self.taper)
        for pos, src in self.sources:
            i, j = pos
            u[tp1, i, j] += src(iteration*self.dt)

    def _plot_snapshot(self, frame, **kwargs):
        with h5py.File(self.cachefile) as f:
            data = f['panels'][frame]
        scale = numpy.abs(data).max()
        nz, nx = self.shape
        dx, dz = nx*self.dx, nz*self.dz
        extent = [0, dx, dz, 0]
        if 'cmap' not in kwargs:
            kwargs['cmap'] = plt.cm.seismic
        plt.imshow(data, extent=extent, vmin=-scale, vmax=scale, **kwargs)
        plt.colorbar(pad=0, aspect=30).set_label('Displacement')

    def maxdt(self):
        nz, nx = self.shape
        return 0.6*maxdt([0, nx*self.dx, 0, nz*self.dz], self.shape, self.velocity.max())


def anim_to_html(anim, fps=6, dpi=30, writer='avconv'):
    """
    Convert a matplotlib animation object to a video embedded in an HTML
    <video> tag.

    Uses avconv (default) or ffmpeg.

    Returns an IPython.display.HTML object for embedding in the notebook.

    Adapted from `the yt project docs
    <http://yt-project.org/doc/cookbook/embedded_webm_animation.html>`__.
    """
    VIDEO_TAG = """
    <video controls>
    <source src="data:video/webm;base64,{0}" type="video/webm">
    Your browser does not support the video tag.
    </video>"""
    plt.close(anim._fig)
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.webm') as f:
            anim.save(f.name, fps=fps, dpi=dpi, writer=writer,
                      extra_args=['-vcodec', 'libvpx'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return HTML(VIDEO_TAG.format(anim._encoded_video))


class ElasticPSV(WaveFD2D):
    """
    Simulate P and SV waves using the Parsimonious Staggered Grid (PSG) finite
    differences scheme of Luo and Schuster (1990).

    Uses absorbing boundary conditions (Gaussian taper) in the lower, left and
    right boundaries. The top implements the free-surface boundary condition
    of Vidale and Clayton (1986).
    """

    def __init__(self, pvel, svel, density, spacing, cachefile=None, dt=None,
                 padding=50, taper=0.007, verbose=True):
        super(ElasticPSV, self).__init__(cachefile, spacing, pvel.shape, dt,
                                         padding, taper, verbose)
        self.pvel = pvel
        self.svel = svel
        self.density = density
        if self.dt is None:
            self.dt = self.maxdt()
        self.mu = lame_mu(svel, density)
        self.lamb = lame_lamb(pvel, svel, density)
        self.padding = padding
        self.taper = taper
        self.make_free_surface_matrices()

    def maxdt(self):
        nz, nx = self.shape
        return 0.6*maxdt([0, nx*self.dx, 0, nz*self.dz], self.shape, self.pvel.max())

    def add_blast_source(self, position, wavelet):
        nz, nx = self.shape
        i, j = position
        amp = 1/(2**0.5)
        locations = [
            [i - 1, j    ,    0,   -1],
            [i + 1, j    ,    0,    1],
            [i    , j - 1,   -1,    0],
            [i    , j + 1,    1,    0],
            [i - 1, j - 1, -amp, -amp],
            [i + 1, j - 1, -amp,  amp],
            [i - 1, j + 1,  amp, -amp],
            [i + 1, j + 1,  amp,  amp],
            ]
        for k, l, xamp, zamp in locations:
            if k >= 0 and k < nz and l >= 0 and l < nx:
                xwav = xamp*wavelet
                zwav = zamp*wavelet
                self.sources.append([[k, l], xwav, zwav])

    def add_point_source(self, position, dip, source):
        """
        Adds a point source to this simulation

        Parameters:

        * position : tuple
            The (x, z) coordinates of the source

        * dip : float
            dip of the source (with respect to the horizontal)
            angle in degrees

        * source : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an example
            source)

        """
        d2r = numpy.pi/180
        xamp = numpy.cos(d2r*dip)
        zamp = numpy.sin(d2r*dip)
        self.sources.append([position, xamp*source, zamp*source])

    def __getitem__(self, args):
        with self._get_cache() as f:
            ux = f['xpanels'][args]
            uz = f['zpanels'][args]
        return [ux, uz]

    def _plot_snapshot(self, frame, **kwargs):
        with h5py.File(self.cachefile) as f:
            ux = f['xpanels'][frame]
            uz = f['zpanels'][frame]
        plottype = kwargs.get('plottype', ['wavefield'])
        nz, nx = self.shape
        mx, mz = nx*self.dx, nz*self.dz
        if 'wavefield' in plottype:
            extent = [0, mx, mz, 0]
            cmap = kwargs.get('cmap', plt.cm.seismic)
            p = numpy.empty(self.shape, dtype=numpy.float)
            s = numpy.empty(self.shape, dtype=numpy.float)
            _xz2ps(ux, uz, p, s, nx, nz, self.dx, self.dz)
            data = p + s
            scale = numpy.abs(data).max()
            vmin = kwargs.get('vmin', -scale)
            vmax = kwargs.get('vmax', scale)
            plt.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            plt.colorbar(pad=0, aspect=30).set_label('Divergence + Curl')
        if 'particles' in plottype:
            every_particle = kwargs.get('every_particle', 5)
            markersize = kwargs.get('markersize', 1)
            scale = kwargs.get('scale', 1)
            xs = numpy.linspace(0, mx, nx)[::every_particle]
            zs = numpy.linspace(0, mz, nz)[::every_particle]
            x, z = numpy.meshgrid(xs, zs)
            x += scale*ux[::every_particle, ::every_particle]
            z += scale*uz[::every_particle, ::every_particle]
            plt.plot(x, z, '.k', markersize=markersize)
        if 'vectors' in plottype:
            every_particle = kwargs.get('every_particle', 5)
            scale = kwargs.get('scale', 1)
            linewidth = kwargs.get('linewidth', 0.1)
            xs = numpy.linspace(0, mx, nx)[::every_particle]
            zs = numpy.linspace(0, mz, nz)[::every_particle]
            x, z = numpy.meshgrid(xs, zs)
            plt.quiver(x, z,
                       ux[::every_particle, ::every_particle],
                       uz[::every_particle, ::every_particle],
                       scale=1/scale, linewidth=linewidth,
                       pivot='tail', angles='xy', scale_units='xy')

    def _init_cache(self, panels, chunks=None, compression='lzf', shuffle=True):
        """
        Init the hdf5 cache file with this simulation parameters

        *  chunks : HDF5 data set option

        * compression: HDF5 data set option

        * shuffle: HDF5 data set option

        """
        nz, nx = self.shape
        if chunks is None:
            chunks = (1, nz//10, nx//10)
        with self._get_cache(mode='w') as f:
            nz, nx = self.shape
            dset = f.create_dataset('xpanels', (panels, nz, nx),
                                     maxshape=(None, nz, nx),
                                     chunks=chunks,
                                     compression=compression,
                                     shuffle=shuffle,
                                     dtype=numpy.float)
            dset.attrs['shape'] = self.shape
            dset.attrs['simsize'] = self.simsize
            dset.attrs['iteration'] = self.it
            dset.attrs['dx'] = self.dx
            dset.attrs['dz'] = self.dz
            dset.attrs['dt'] = self.dt
            dset.attrs['padding'] = self.padding
            dset.attrs['taper'] = self.taper
            f.create_dataset('zpanels', (panels, nz, nx),
                                     maxshape=(None, nz, nx),
                                     chunks=chunks,
                                     compression=compression,
                                     shuffle=shuffle,
                                     dtype=numpy.float)
            f.create_dataset('pvel', data=self.pvel,
                             chunks=chunks[1:],
                             compression=compression,
                             shuffle=shuffle)
            f.create_dataset('svel', data=self.svel,
                             chunks=chunks[1:],
                             compression=compression,
                             shuffle=shuffle)
            f.create_dataset('density', data=self.density,
                             chunks=chunks[1:],
                             compression=compression,
                             shuffle=shuffle)
            f.create_dataset(
                'sources', data=numpy.void(pickle.dumps(self.sources)))
            dset.attrs['len'] = len(self.sources)

    @staticmethod
    def from_cache(fname, verbose=True):
        """
        Creates a simulation object from a pre-existing HDF5 file

        * fname: str
            HDF5 file path containing a previous simulation stored

        * verbose: bool
            Progress status shown or not
        """
        with h5py.File(fname, 'r') as f:
            pvel = f['pvel']
            svel = f['svel']
            dens = f['density']
            panels = f['xpanels']  # the first data set contain all the simulation parameters
            dx = panels.attrs['dx']
            dz = panels.attrs['dz']
            dt = panels.attrs['dt']
            padding = panels.attrs['padding']
            taper = panels.attrs['taper']
            sim = ElasticPSV(pvel[:], svel[:], dens[:], (dx, dz), dt=dt,
                             padding=padding, taper=taper, cachefile=fname)
            sim.simsize = panels.attrs['simsize']
            sim.it = panels.attrs['iteration']
            sim.sources = pickle.loads(f['sources'].value.tostring())
        sim.verbose = verbose
        return sim

    def _cache_panels(self, u, tp1, iteration, simul_size):
        """
        Save the last calculated panels and information about it
        in the hdf5 cache file

        Parameters:

        * panels : tuple or variable
            tuple or variable containing all 2D panels needed for this simulation
        * tp1 : int
            panel time index
        * iteration:
            iteration number
        * simul_size:
            number of iterations that has been run
        """
        # Save the panel to disk
        with self._get_cache(mode='a') as f:
            ux, uz = u
            xpanels = f['xpanels']
            zpanels = f['zpanels']
            xpanels[simul_size - 1] = ux[tp1]
            zpanels[simul_size - 1] = uz[tp1]
            xpanels.attrs['simsize'] = simul_size
            xpanels.attrs['iteration'] = iteration
            zpanels.attrs['simsize'] = simul_size
            zpanels.attrs['iteration'] = iteration


    def _expand_cache(self, iterations):
        with self._get_cache(mode='a') as f:
            f['xpanels'].resize(self.simsize + iterations, axis=0)
            f['zpanels'].resize(self.simsize + iterations, axis=0)

    def _init_panels(self):
        if self.simsize == 0:
            nz, nx = self.shape
            ux = numpy.zeros((2, nz, nx), dtype=numpy.float)
            uz = numpy.zeros((2, nz, nx), dtype=numpy.float)
        else:
            # Get the last two panels from the cache
            with self._get_cache() as f:
                # Reverse the array because the later time comes first in
                # ux, uz
                # Need to copy them and reorder because the _timestep function
                # takes the whole ux array and complains that it isn't C
                # contiguous
                # Could change the _timestep to pass ux[tp1], etc, like in
                # ElasticSH. That would probably be much better.
                ux = numpy.copy(f['xpanels'][self.simsize - 2 : self.simsize][::-1],
                                order='C')
                uz = numpy.copy(f['zpanels'][self.simsize - 2 : self.simsize][::-1],
                                order='C')
        return [ux, uz]


    def _timestep(self, u, tm1, t, tp1, iteration):
        nz, nx = self.shape
        ux, uz = u
        _step_elastic_psv(ux, uz, tp1, t, tm1, 1, nx - 1,  1, nz - 1,
                          self.dt, self.dx, self.dz,
                          self.mu, self.lamb, self.density)
        _apply_damping(ux[t], nx, nz, self.padding, self.taper)
        _apply_damping(uz[t], nx, nz, self.padding, self.taper)
        # Free-surface boundary conditions
        Mx1, Mx2, Mx3 = self.Mx
        Mz1, Mz2, Mz3 = self.Mz
        ux[tp1,0,:] = scipy.sparse.linalg.spsolve(
            Mx1, Mx2*ux[tp1,1,:] + Mx3*uz[tp1,1,:])
        uz[tp1,0,:] = scipy.sparse.linalg.spsolve(
            Mz1, Mz2*uz[tp1,1,:] + Mz3*ux[tp1,1,:])
        _nonreflexive_psv_boundary_conditions(
            ux, uz, tp1, t, tm1, nx, nz, self.dt, self.dx, self.dz,
            self.mu, self.lamb, self.density)
        _apply_damping(ux[tp1], nx, nz, self.padding, self.taper)
        _apply_damping(uz[tp1], nx, nz, self.padding, self.taper)
        for pos, xsrc, zsrc in self.sources:
            i, j = pos
            ux[tp1, i, j] += xsrc(iteration*self.dt)
            uz[tp1, i, j] += zsrc(iteration*self.dt)

    def make_free_surface_matrices(self):
        # Pre-compute the matrices required for the free-surface BC
        nz, nx = self.shape
        dzdx = 1
        identity = scipy.sparse.identity(nx)
        B = scipy.sparse.eye(nx, nx, k=1) - scipy.sparse.eye(nx, nx, k=-1)
        gamma = scipy.sparse.spdiags(
            self.lamb[0]/(self.lamb[0] + 2*self.mu[0]), [0], nx, nx)
        Mx1 = identity - 0.0625*(dzdx**2)*B*gamma*B
        Mx2 = identity + 0.0625*(dzdx**2)*B*gamma*B
        Mx3 = 0.5*dzdx*B
        Mz1 = identity - 0.0625*(dzdx**2)*gamma*B*B
        Mz2 = identity + 0.0625*(dzdx**2)*gamma*B*B
        Mz3 = 0.5*dzdx*gamma*B
        self.Mx = [Mx1, Mx2, Mx3]
        self.Mz = [Mz1, Mz2, Mz3]

    def animate(self, every=1, plottype=['wavefield'], cutoff=None,
                cmap=plt.cm.seismic, scale=1, every_particle=5,
                ax=None,  interval=100, embed=False, blit=False,
                fps=10, dpi=70, writer='avconv', **kwargs):
        nz, nx = self.shape
        mx, mz = nx*self.dx, nz*self.dz
        if ax is None:
            plt.figure(facecolor='white')
            ax = plt.subplot(111)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_xlim(0, mx)
            ax.set_ylim(0, mz)
            ax.invert_yaxis()
        fig = ax.get_figure()
        wavefield = None
        particles = None
        vectors = None
        if 'wavefield' in plottype:
            extent = [0, mx, mz, 0]
            p = numpy.empty(self.shape, dtype=numpy.float)
            s = numpy.empty(self.shape, dtype=numpy.float)
            imshow_args = dict(cmap=cmap, extent=extent)
            if cutoff is not None:
                imshow_args['vmin'] = -cutoff
                imshow_args['vmax'] = cutoff
            wavefield = ax.imshow(numpy.zeros(self.shape), **imshow_args)
            fig.colorbar(wavefield, pad=0, aspect=30).set_label(
                'Divergence + Curl')
        if 'particles' in plottype or 'vectors' in plottype:
            xs = numpy.linspace(0, mx, nx)[::every_particle]
            zs = numpy.linspace(0, mz, nz)[::every_particle]
            x, z = numpy.meshgrid(xs, zs)
        if 'particles' in plottype:
            markersize = kwargs.get('markersize', 1)
            style = kwargs.get('style', '.k')
            particles, = plt.plot(x.ravel(), z.ravel(), style,
                                  markersize=markersize)
        if 'vectors' in plottype:
            linewidth = kwargs.get('linewidth', 0.1)
            vectors = plt.quiver(x, z, numpy.zeros_like(x),
                                 numpy.zeros_like(z),
                                 scale=1/scale, linewidth=linewidth,
                                 pivot='tail', angles='xy',
                                 scale_units='xy')
        # Check the aspect ratio of the plot and adjust figure size to match
        aspect = min(self.shape)/max(self.shape)
        try:
            aspect /= ax.get_aspect()
        except TypeError:
            pass
        if nx > nz:
            width = 10
            height = width*aspect*0.8
        else:
            height = 8
            width = height*aspect*1.5
        fig.set_size_inches(width, height)
        def plot(i):
            ax.set_title('iteration: {:d}'.format(i*every))
            ux, uz = self[i*every]
            if wavefield is not None:
                _xz2ps(ux, uz, p, s, nx, nz, self.dx, self.dz)
                wavefield.set_array(p + s)
            if particles is not None or vectors is not None:
                ux = ux[::every_particle, ::every_particle]
                uz = uz[::every_particle, ::every_particle]
            if particles is not None:
                particles.set_data(x.ravel() + scale*ux.ravel(),
                                   z.ravel() + scale*uz.ravel())
            if vectors is not None:
                vectors.set_UVC(ux, uz)
            return wavefield, particles, vectors
        frames = self.simsize//every
        anim = animation.FuncAnimation(fig, plot, frames=frames, blit=blit,
                                       interval=interval)
        if embed:
            return anim_to_html(anim, fps=fps, dpi=dpi, writer=writer)
        else:
            plt.show()
            return anim


class Scalar(WaveFD2D):
    r"""
    Simulate scalar waves using an explicit finite differences scheme 4th order
    space. Space increment must be equal in x and z.

    Parameters:

    * velocity: 2D-array (defines shape simulation)
        The wave velocity at all the grid nodes
    * spacing: (dx, dz)
        space increment for x and z direction
    * cachefile: str
        The hdf5 cachefile file path to store the simulation
    * dt: float
        time increment for simulation (recommended not to set)
    * padding : int
        Number of grid nodes to use for the absorbing boundary region
    * taper : float
        The intensity of the Gaussian taper function used for the absorbing
        boundary conditions
    * verbose: bool
        True to show simulation progress bar

    """
    def __init__(self, velocity, spacing, cachefile=None, dt=None,
                 padding=50, taper=0.007, verbose=True):
        super(Scalar, self).__init__(cachefile, spacing, velocity.shape, dt, padding,
                                        taper, verbose)
        self.velocity = velocity
        if self.dt is None:
            self.dt = self.maxdt()

    def maxdt(self):
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

        Returns:

        * maxdt : float
            The maximum time step

        """
        min_spacing = min(self.dx, self.dz)
        maxvel = numpy.max(self.velocity)
        factor = numpy.sqrt(3. / 8.)
        factor -= factor / 100.  # 1% smaller to guarantee criteria
        # the closer to stability criteria the better the convergence
        return factor * min_spacing / maxvel

    def __getitem__(self, index):
        """
        Get an iteration or array slicing of simulation.
        The simulation is stored in hdf5 file as a 3D array with
        shape (`simsize`, nz, nx) where `simsize` is the last
        iteration that has been run.

        Parameters:

        * index: int or array slicing
            simulation iteration index or
            slicing over simulation

        Returns:

        * 3D array
            two 2D simulation panels shape (`simsize`, nz, nx)
        """
        with self._get_cache() as f:
            data = f['panels'][index]
        return data

    def add_point_source(self, position, wavelet):
        """"
        Adds a point source to this simulation.

        Parameters:

        * position : tuple
            The (x, z) coordinates of the source
        * source : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an example source)

        """
        self.sources.append([position, wavelet])

    def _init_cache(self, npanels, chunks=None, compression='lzf', shuffle=True):
        """
        Initiate the hdf5 cache file with this simulation parameters
        (only called by run)

        Parameters:

        * npanels: int
            number of 2D panels needed for storing this simulation run
        * chunks : HDF5 data set option (Tuple)
            Chunk shape, or True to enable auto-chunking.
        * compression: HDF5 data set option (String or int)
            Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        * shuffle: HDF5 data set option (bool)
            Enable shuffle filter.
        """
        nz, nx = self.shape
        if chunks is None:
            chunks = (1, nz//10, nx//10)
        with self._get_cache(mode='w') as f:  # create HDF5 data sets
            nz, nx = self.shape
            pad = self.padding
            dset = f.create_dataset('panels', (npanels, nz, nx),
                                     maxshape=(None, nz, nx),
                                     chunks=chunks,
                                     compression=compression,
                                     shuffle=shuffle,
                                     dtype=numpy.float)
            dset.attrs['shape'] = self.shape
            # simsize stores the total size of this simulation
            # after some or many runs
            dset.attrs['simsize'] = self.simsize
            # it is the `run` iteration time step indexer
            dset.attrs['iteration'] = self.it
            dset.attrs['dx'] = self.dx
            dset.attrs['dz'] = self.dz
            dset.attrs['dt'] = self.dt
            dset.attrs['padding'] = self.padding
            dset.attrs['taper'] = self.taper
            f.create_dataset('velocity', data=self.velocity[:-pad, pad:-pad])
            f.create_dataset(
                'sources', data=numpy.void(pickle.dumps(self.sources)))

    @staticmethod
    def from_cache(fname, verbose=True):
        """
        Creates a simulation object from a pre-existing HDF5 file

        Parameters

        * fname: str
            HDF5 file path containing a previous simulation stored

        * verbose: bool
            Progress status shown or not

        Returns:

        * from_cache: object
            a new :class:`~fatiando.seismic.wavefd.Scalar` with
            the simulation previously stored

        """
        with h5py.File(fname, 'r') as f:
            vel = f['velocity']
            panels = f['panels']
            dx = panels.attrs['dx']
            dz = panels.attrs['dz']
            dt = panels.attrs['dt']
            padding = panels.attrs['padding']
            taper = panels.attrs['taper']
            # created from velocity so it has the shape without padding
            sim = Scalar(vel[:], (dx, dz), dt=dt, padding=padding,
                         taper=taper, cachefile=fname)
            sim.simsize = panels.attrs['simsize']
            sim.it = panels.attrs['iteration']
            sim.sources = pickle.loads(f['sources'].value.tostring())
        sim.set_verbose(verbose)
        return sim

    def _expand_cache(self, npanels):
        """
        Expand the hdf5 cache file of this simulation parameters
        for more iterations.
        (only called by run)

        Parameters:

        *  npanels: int
            number of additional 2D panels needed for a simulation run
        """
        with self._get_cache(mode='a') as f:
            cache = f['panels']
            cache.resize(self.simsize + npanels, axis=0)

    def _cache_panels(self, u, tp1, iteration, simul_size):
        """
        Save the last calculated time step panels and information about it
        in the hdf5 cache file
        (only called by run)

        Parameters:

        * u : 3D array
            two 2D simulation panels shape (2, nz+pad, nx+2pad)
        * tp1 : int
            panel time index
        * iteration:
            iteration time step number
        * simul_size:
            number of iterations that has been run
        """
        # Save the panel to disk
        pad = self.padding
        with self._get_cache(mode='a') as f:
            cache = f['panels']
            cache[simul_size - 1] = u[tp1, :-pad, pad:-pad]
            cache.attrs['simsize'] = simul_size  # total iterations ran
            cache.attrs['iteration'] = iteration

    def _init_panels(self):
        """
        Initiate the simulation panels used for finite differences solution.
        Keep consistency of simulation if loaded from file.
        (only called by run)

        Returns:

        * _init_panels: 3D array
            two 2D simulation panels shape (2, nz+pad, nx+2pad)

        """
        # If this is the first run, start with zeros, else, get the last two
        # panels from the cache so that the simulation can be resumed
        nz, nx = self.shape  # self.shape is not changed
        pad = self.padding
        # Add some padding to x and z. The padding region is where
        # the wave is absorbed
        nx += 2 * pad
        nz += pad  # free up
        if self.simsize == 0:
            # Pack the particle position u at 2 different times in one 3d array
            # u[0] = u(t-1)
            # u[1] = u(t)
            # The next time step overwrites the t-1 panel
            u = numpy.zeros((2, nz, nx), dtype=numpy.float)
        else:  # came from cache or another run
            with self._get_cache() as f:
                cache = f['panels']
                u = cache[self.simsize - 2: self.simsize][::-1]  # 0 and 1 invert
                # u was saved with just the unpad region
                u_ = numpy.zeros((2, nz, nx), dtype=numpy.float)
                u_[:, :-pad, pad:-pad] += u
                u = u_
        # Pad the velocity if needed
        if self.velocity.shape != (nz, nx):
            self.velocity = _add_pad(self.velocity, pad, (nz, nx))
        return u

    def _timestep(self, u, tm1, t, tp1, iteration):
        """
        Performs a single step on time (finite differences solution)
        (only called by run)

        Parameters:

        * u: 3D array
            two 2D simulation panels shape (2, nz+pad, nx+2pad)
        * tm1: int
            panel index (t minus 1)
            (to avoid copying between 2D arrays)
        * t: int
            panel index (t)
            (to avoid copying between 2D arrays)
        * tp1: int
            panel index (t plus 1)
            (to avoid copying between 2D arrays)
        * iteration: int
            iteration time step of this simulation
        """

        nz, nx = self.shape
        # due dump regions
        nz += self.padding
        nx += self.padding*2
        ds = self.dx  # increment equal
        _step_scalar(u[tp1], u[t], u[tm1], 2, nx - 2, 2, nz - 2,
                     self.dt, ds, self.velocity)
        # forth order +2-2 indexes needed
        # Damp the regions in the padding to make waves go to infinity
        _apply_damping(u[t], nx, nz, self.padding, self.taper)
        # not PML yet or anything similar
        _reflexive_scalar_boundary_conditions(u[tp1], nx, nz)
        # Damp the regions in the padding to make waves go to infinity
        _apply_damping(u[tp1], nx, nz, self.padding, self.taper)
        for pos, src in self.sources:
            i, j = pos
            u[tp1, i, j + self.padding] += src(iteration*self.dt)

    def _plot_snapshot(self, frame, **kwargs):
        """
        Plots the 2D wavefield at an iteration time index using matplotlib.

        Parameters:

        frame: int
            time step iteration index smaller than `simsize`
        """
        data = self.__getitem__(frame)
        scale = numpy.abs(data).max()
        nz, nx = self.shape
        dx, dz = nx*self.dx, nz*self.dz
        extent = [0, dx, dz, 0]
        if 'cmap' not in kwargs:
            kwargs['cmap'] = plt.cm.seismic
        plt.imshow(data, extent=extent, vmin=-scale, vmax=scale, **kwargs)
        plt.colorbar(pad=0, aspect=30).set_label('Displacement')


# class MexHatSource(object):
#
#     r"""
#     A wave source that vibrates as a Mexican hat (Ricker) wavelet.
#
#     .. math::
#
#         \psi(t) = A(1 - 2 \pi^2 f^2 t^2)exp(-\pi^2 f^2 t^2)
#
#     Parameters:
#
#     * x, z : float
#         The x, z coordinates of the source
#     * area : [xmin, xmax, zmin, zmax]
#         The area bounding the finite difference simulation
#     * shape : (nz, nx)
#         The number of nodes in the finite difference grid
#     * amp : float
#         The amplitude of the source (:math:`A`)
#     * frequency : float
#         The peak frequency of the wavelet
#     * delay : float
#         The delay before the source starts
#
#         .. note:: If you want the source to start with amplitude close to 0,
#             use ``delay = 3.5/frequency``.
#
#     """
#
#     def __init__(self, x, z, area, shape, amp, frequency, delay=0):
#         nz, nx = shape
#         dz, dx = sum(area[2:]) / (nz - 1), sum(area[:2]) / (nx - 1)
#         self.i = int(round((z - area[2]) / dz))
#         self.j = int(round((x - area[0]) / dx))
#         self.x, self.z = x, z
#         self.amp = amp
#         self.frequency = frequency
#         self.f2 = frequency ** 2
#         self.delay = delay
#
#     def __call__(self, time):
#         t2 = (time - self.delay) ** 2
#         pi2 = numpy.pi ** 2
#         psi = self.amp * (1 - 2 * pi2 * self.f2 * t2) * \
#             numpy.exp(-pi2 * self.f2 * t2)
#         return psi
#
#     def coords(self):
#         """
#         Get the x, z coordinates of the source.
#
#         Returns:
#
#         * (x, z) : tuple
#             The x, z coordinates
#
#         """
#         return (self.x, self.z)
#
#     def indexes(self):
#         """
#         Get the i,j coordinates of the source in the finite difference grid.
#
#         Returns:
#
#         * (i,j) : tuple
#             The i,j coordinates
#
#         """
#         return (self.i, self.j)


# def blast_source(x, z, area, shape, amp, frequency, delay=0,
#                  sourcetype=MexHatSource):
#     """
#     Uses several MexHatSources to create a blast source that pushes in all
#     directions.
#
#     Parameters:
#
#     * x, z : float
#         The x, z coordinates of the source
#     * area : [xmin, xmax, zmin, zmax]
#         The area bounding the finite difference simulation
#     * shape : (nz, nx)
#         The number of nodes in the finite difference grid
#     * amp : float
#         The amplitude of the source
#     * frequency : float
#         The frequency of the source
#     * delay : float
#         The delay before the source starts
#     * sourcetype : source class
#         The type of source to use, like
#         :class:`~fatiando.seismic.wavefd.MexHatSource`.
#
#     Returns:
#
#     * [xsources, zsources]
#         Lists of sources for x- and z-displacements
#
#     """
#     nz, nx = shape
#     xsources, zsources = [], []
#     center = sourcetype(x, z, area, shape, amp, frequency, delay)
#     i, j = center.indexes()
#     tmp = numpy.sqrt(2)
#     locations = [[i - 1, j - 1, -amp, -amp],
#                  [i - 1, j, 0, -tmp * amp],
#                  [i - 1, j + 1, amp, -amp],
#                  [i, j - 1, -tmp * amp, 0],
#                  [i, j + 1, tmp * amp, 0],
#                  [i + 1, j - 1, -amp, amp],
#                  [i + 1, j, 0, tmp * amp],
#                  [i + 1, j + 1, amp, amp]]
#     locations = [[i, j, xamp, zamp] for i, j, xamp, zamp in locations
#                  if i >= 0 and i < nz and j >= 0 and j < nx]
#     for i, j, xamp, zamp in locations:
#         xsrc = sourcetype(x, z, area, shape, xamp, frequency, delay)
#         xsrc.i, xsrc.j = i, j
#         zsrc = sourcetype(x, z, area, shape, zamp, frequency, delay)
#         zsrc.i, zsrc.j = i, j
#         xsources.append(xsrc)
#         zsources.append(zsrc)
#     return xsources, zsources


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


def _add_pad(array, pad, shape):
    """
    Pad the array with the values of the borders
    """
    array_pad = numpy.zeros(shape, dtype=numpy.float)
    array_pad[:-pad, pad:-pad] = array
    for k in xrange(pad):
        array_pad[:-pad, k] = array[:, 0]
        array_pad[:-pad, -(k + 1)] = array[:, -1]
    for k in xrange(pad):
        array_pad[-(pad - k), :] = array_pad[-(pad + 1), :]
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
    dz, dx = (z2 - z1) / (nz - 1), (x2 - x1) / (nx - 1)
    # Get the index of the closest point to the stations and start the
    # seismograms
    if stations is not None:
        stations = [[int(round((z - z1) / dz)), int(round((x - x1) / dx))]
                    for x, z in stations]
        seismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
    else:
        stations, seismograms = [], []
    # Add some padding to x and z. The padding region is where the wave is
    # absorbed
    pad = int(padding)
    nx += 2 * pad
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
        u[1, i, j + pad] += (dt ** 2 / density[i, j]) * src(0)
    # Update seismograms
    for station, seismogram in zip(stations, seismograms):
        i, j = station
        seismogram[0] = u[1, i, j + pad]
    if snapshot is not None:
        yield 0, u[1, :-pad, pad:-pad], seismograms
    for iteration in xrange(1, iterations):
        t, tm1 = iteration % 2, (iteration + 1) % 2
        tp1 = tm1
        _step_elastic_sh(u[tp1], u[t], u[tm1], 3, nx - 3, 3, nz - 3, dt, dx,
                         dz, mu_pad, dens_pad)
        _apply_damping(u[t], nx, nz, pad, taper)
        _nonreflexive_sh_boundary_conditions(u[tp1], u[t], nx, nz, dt, dx, dz,
                                             mu_pad, dens_pad)
        _apply_damping(u[tp1], nx, nz, pad, taper)
        for src in sources:
            i, j = src.indexes()
            u[tp1, i, j +
                pad] += (dt ** 2 / density[i, j]) * src(iteration * dt)
        # Update seismograms
        for station, seismogram in zip(stations, seismograms):
            i, j = station
            seismogram[iteration] = u[tp1, i, j + pad]
        if snapshot is not None and iteration % snapshot == 0:
            yield iteration, u[tp1, :-pad, pad:-pad], seismograms
    yield iteration, u[tp1, :-pad, pad:-pad], seismograms


def elastic_psv(mu, lamb, density, area, dt, iterations, sources,
                stations=None, snapshot=None, padding=50, taper=0.002,
                xz2ps=False):
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
    dz, dx = (z2 - z1) / (nz - 1), (x2 - x1) / (nx - 1)
    xsources, zsources = sources
    # Get the index of the closest point to the stations and start the
    # seismograms
    if stations is not None:
        stations = [[int(round((z - z1) / dz)), int(round((x - x1) / dx))]
                    for x, z in stations]
        xseismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
        zseismograms = [numpy.zeros(iterations) for i in xrange(len(stations))]
    else:
        stations, xseismograms, zseismograms = [], [], []
    # Add padding to have an absorbing region to simulate an infinite medium
    pad = int(padding)
    nx += 2 * pad
    nz += pad
    mu_pad = _add_pad(mu, pad, (nz, nx))
    lamb_pad = _add_pad(lamb, pad, (nz, nx))
    dens_pad = _add_pad(density, pad, (nz, nx))
    # Pre-compute the matrices required for the free-surface boundary
    dzdx = dz / dx
    identity = scipy.sparse.identity(nx)
    B = scipy.sparse.eye(nx, nx, k=1) - scipy.sparse.eye(nx, nx, k=-1)
    gamma = scipy.sparse.spdiags(lamb_pad[0] / (lamb_pad[0] + 2 * mu_pad[0]),
                                 [0], nx, nx)
    Mx1 = identity - 0.0625 * (dzdx ** 2) * B * gamma * B
    Mx2 = identity + 0.0625 * (dzdx ** 2) * B * gamma * B
    Mx3 = 0.5 * dzdx * B
    Mz1 = identity - 0.0625 * (dzdx ** 2) * gamma * B * B
    Mz2 = identity + 0.0625 * (dzdx ** 2) * gamma * B * B
    Mz3 = 0.5 * dzdx * gamma * B
    # Compute and yield the initial solutions
    ux = numpy.zeros((2, nz, nx), dtype=numpy.float)
    uz = numpy.zeros((2, nz, nx), dtype=numpy.float)
    if xz2ps:
        p, s = numpy.empty_like(mu), numpy.empty_like(mu)
    for src in xsources:
        i, j = src.indexes()
        ux[1, i, j + pad] += (dt ** 2 / density[i, j]) * src(0)
    for src in zsources:
        i, j = src.indexes()
        uz[1, i, j + pad] += (dt ** 2 / density[i, j]) * src(0)
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
        t, tm1 = iteration % 2, (iteration + 1) % 2
        tp1 = tm1
        _step_elastic_psv(ux, uz, tp1, t, tm1, 1, nx - 1,  1, nz - 1, dt, dx,
                          dz, mu_pad, lamb_pad, dens_pad)
        _apply_damping(ux[t], nx, nz, pad, taper)
        _apply_damping(uz[t], nx, nz, pad, taper)
        # Free-surface boundary conditions
        ux[tp1, 0, :] = scipy.sparse.linalg.spsolve(
            Mx1, Mx2*ux[tp1, 1, :] + Mx3*uz[tp1, 1, :])
        uz[tp1, 0, :] = scipy.sparse.linalg.spsolve(
            Mz1, Mz2*uz[tp1, 1, :] + Mz3*ux[tp1, 1, :])
        _nonreflexive_psv_boundary_conditions(ux, uz, tp1, t, tm1, nx, nz, dt,
                                              dx, dz, mu_pad, lamb_pad,
                                              dens_pad)
        _apply_damping(ux[tp1], nx, nz, pad, taper)
        _apply_damping(uz[tp1], nx, nz, pad, taper)
        for src in xsources:
            i, j = src.indexes()
            ux[tp1, i, j + pad] += (dt**2 / density[i, j])*src(iteration*dt)
        for src in zsources:
            i, j = src.indexes()
            uz[tp1, i, j +
                pad] += (dt ** 2 / density[i, j]) * src(iteration * dt)
        for station, xseis, zseis in zip(stations, xseismograms, zseismograms):
            i, j = station
            xseis[iteration] = ux[tp1, i, j + pad]
            zseis[iteration] = uz[tp1, i, j + pad]
        if snapshot is not None and iteration % snapshot == 0:
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
    dz, dx = (z2 - z1) / (nz - 1), (x2 - x1) / (nx - 1)
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
    spacing = min([(x2 - x1) / (nx - 1), (z2 - z1) / (nz - 1)])
    return 0.606 * spacing / maxvel





