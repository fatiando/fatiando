from __future__ import division, print_function
from future.builtins import super, object, range
import cPickle as pickle

import numpy as np
from numpy import sqrt
import scipy.sparse
import scipy.sparse.linalg
import numba
import h5py
from matplotlib import animation
from matplotlib import pyplot as plt

from ...vis import anim_to_html
from .utils import apply_damping, lame_lamb, lame_mu, xz2ps
from .base import WaveFD2D


class FDElasticPSV(WaveFD2D):

    def __init__(self, pvel, svel, density, spacing, cachefile=None, dt=None,
                 padding=50, taper=0.005, verbose=True):
        super().__init__(cachefile, spacing, pvel.shape, dt, padding, taper,
                         verbose)
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
        x1, x2, z1, z2 = [0, nx*self.dx, 0, nz*self.dz]
        spacing = min([(x2 - x1) / (nx - 1), (z2 - z1) / (nz - 1)])
        # Be conservative and use 0.6x the recommended value
        return 0.6*0.606*spacing/self.pvel.max()

    def add_blast_source(self, position, wavelet):
        """
        Adds a point source to this simulation

        Parameters:

        * position : tuple
            The (x, z) coordinates of the source

        * wavelet : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an example
            wavelet)

        """
        nz, nx = self.shape
        i, j = position
        amp = 1/(2**0.5)
        locations = [
            [i - 1,     j,    0,   -1],
            [i + 1,     j,    0,    1],
            [i    , j - 1,   -1,    0],
            [i    , j + 1,    1,    0],
            [i - 1, j - 1, -amp, -amp],
            [i + 1, j - 1, -amp,  amp],
            [i - 1, j + 1,  amp, -amp],
            [i + 1, j + 1,  amp,  amp],
            ]
        for k, l, xamp, zamp in locations:
            if k >= 0 and k < nz and l >= 0 and l < nx:
                xwav = wavelet.copy()
                xwav.amp *= xamp
                zwav = wavelet.copy()
                zwav.amp *= zamp
                self.sources.append([[k, l], xwav, zwav])

    def add_point_source(self, position, dip, wavelet):
        """
        Adds a point source to this simulation

        Parameters:

        * position : tuple
            The (x, z) coordinates of the source

        * dip : float
            dip of the source (with respect to the horizontal)
            angle in degrees

        * wavelet : source function
            (see :class:`~fatiando.seismic.wavefd.Ricker` for an example
            wavelet)

        """
        d2r = np.pi/180
        xwav = wavelet.copy()
        xwav.amp *= np.cos(d2r*dip)
        zwav = wavelet.copy()
        zwav.amp *= np.sin(d2r*dip)
        self.sources.append([position, xwav, zwav])

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
            p = np.empty(self.shape, dtype=np.float32)
            s = np.empty(self.shape, dtype=np.float32)
            xz2ps(ux, uz, p, s, nx, nz, self.dx, self.dz)
            data = p + s
            scale = kwargs.pop('cutoff', np.abs(data).max())
            vmin = kwargs.get('vmin', -scale)
            vmax = kwargs.get('vmax', scale)
            plt.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            plt.colorbar(pad=0, aspect=30).set_label('Divergence + Curl')
        if 'particles' in plottype:
            every_particle = kwargs.get('every_particle', 5)
            markersize = kwargs.get('markersize', 1)
            scale = kwargs.get('scale', 1)
            xs = np.linspace(0, mx, nx)[::every_particle]
            zs = np.linspace(0, mz, nz)[::every_particle]
            x, z = np.meshgrid(xs, zs)
            x += scale*ux[::every_particle, ::every_particle]
            z += scale*uz[::every_particle, ::every_particle]
            plt.plot(x, z, '.k', markersize=markersize)
        if 'vectors' in plottype:
            every_particle = kwargs.get('every_particle', 5)
            scale = kwargs.get('scale', 1)
            linewidth = kwargs.get('linewidth', 0.1)
            xs = np.linspace(0, mx, nx)[::every_particle]
            zs = np.linspace(0, mz, nz)[::every_particle]
            x, z = np.meshgrid(xs, zs)
            plt.quiver(x, z,
                       ux[::every_particle, ::every_particle],
                       uz[::every_particle, ::every_particle],
                       scale=1/scale, linewidth=linewidth,
                       pivot='tail', angles='xy', scale_units='xy')

    def _init_cache(self, panels, chunks=None, compression='lzf',
                    shuffle=True):
        """
        Init the hdf5 cache file with this simulation parameters

        Parameters:

        * panels:

        *  chunks : HDF5 data set option
            (Tuple) Chunk shape, or True to enable auto-chunking.
        * compression: HDF5 data set option
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        * shuffle: (bool) HDF5 data set option
            (T/F) Enable shuffle filter.

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
                                    dtype=np.float32)
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
                             dtype=np.float32)
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
                'sources', data=np.void(pickle.dumps(self.sources)))
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
            # the first data set contain all the simulation parameters
            panels = f['xpanels']
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
            tuple or variable containing all 2D panels needed for
            this simulation
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
            ux = np.zeros((2, nz, nx), dtype=np.float32)
            uz = np.zeros((2, nz, nx), dtype=np.float32)
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
                ux = np.copy(f['xpanels'][self.simsize - 2: self.simsize]
                                [::-1], order='C')
                uz = np.copy(f['zpanels'][self.simsize - 2: self.simsize]
                                [::-1], order='C')
        return [ux, uz]

    def _timestep(self, u, tm1, t, tp1, iteration):
        nz, nx = self.shape
        ux, uz = u
        timestep_psg(ux, uz, tp1, t, tm1, 1, nx - 1,  1, nz - 1, self.dt,
                     self.dx, self.dz, self.mu, self.lamb, self.density)
        apply_damping(ux[t], nx, nz, self.padding, self.taper)
        apply_damping(uz[t], nx, nz, self.padding, self.taper)
        # Free-surface boundary conditions
        Mx1, Mx2, Mx3 = self.Mx
        Mz1, Mz2, Mz3 = self.Mz
        ux[tp1, 0, :] = scipy.sparse.linalg.spsolve(
            Mx1, Mx2*ux[tp1, 1, :] + Mx3*uz[tp1, 1, :])
        uz[tp1, 0, :] = scipy.sparse.linalg.spsolve(
            Mz1, Mz2*uz[tp1, 1, :] + Mz3*ux[tp1, 1, :])
        nonreflexive_bc(ux, uz, tp1, t, tm1, nx, nz, self.dt, self.dx,
                        self.dz, self.mu, self.lamb, self.density)
        apply_damping(ux[tp1], nx, nz, self.padding, self.taper)
        apply_damping(uz[tp1], nx, nz, self.padding, self.taper)
        for pos, xsrc, zsrc in self.sources:
            i, j = pos
            scale = self.dt**2/self.density[i, j]
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
                fps=10, dpi=70, writer='ffmpeg', **kwargs):
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
            p = np.empty(self.shape, dtype=np.float32)
            s = np.empty(self.shape, dtype=np.float32)
            imshow_args = dict(cmap=cmap, extent=extent)
            if cutoff is not None:
                imshow_args['vmin'] = -cutoff
                imshow_args['vmax'] = cutoff
            wavefield = ax.imshow(np.zeros(self.shape), **imshow_args)
            fig.colorbar(wavefield, pad=0, aspect=30).set_label(
                'Divergence + Curl')
        if 'particles' in plottype or 'vectors' in plottype:
            xs = np.linspace(0, mx, nx)[::every_particle]
            zs = np.linspace(0, mz, nz)[::every_particle]
            x, z = np.meshgrid(xs, zs)
        if 'particles' in plottype:
            markersize = kwargs.get('markersize', 1)
            style = kwargs.get('style', '.k')
            particles, = plt.plot(x.ravel(), z.ravel(), style,
                                  markersize=markersize)
        if 'vectors' in plottype:
            linewidth = kwargs.get('linewidth', 0.1)
            vectors = plt.quiver(x, z, np.zeros_like(x),
                                 np.zeros_like(z),
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
                xz2ps(ux, uz, p, s, nx, nz, self.dx, self.dz)
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


@numba.jit(nopython=True, nogil=True)
def nonreflexive_bc(ux, uz, tp1, t, tm1, nx, nz, dt, dx, dz, mu, lamb, dens):
    """
    Apply nonreflexive boundary contitions to elastic P-SV waves.
    """
    for i in range(nz):
        # Left
        j = 0
        ux[tp1,i,j] = ux[t,i,j] + dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j+1] - ux[t,i,j])/dx
        uz[tp1,i,j] = uz[t,i,j] + dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j+1] - uz[t,i,j])/dx
        # Right
        j = nx - 1
        ux[tp1,i,j] = ux[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j] - ux[t,i,j-1])/dx
        uz[tp1,i,j] = uz[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j] - uz[t,i,j-1])/dx
    # Bottom
    i = nz - 1
    for j in range(nx):
        ux[tp1,i,j] = ux[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            ux[t,i,j] - ux[t,i-1,j])/dz
        uz[tp1,i,j] = uz[t,i,j] - dt*sqrt((lamb[i,j] + 2*mu[i,j])/dens[i,j])*(
            uz[t,i,j] - uz[t,i-1,j])/dz


@numba.jit(nopython=True, nogil=True)
def timestep_psg(ux, uz, tp1, t, tm1, x1, x2, z1, z2, dt, dx, dz, mu, lamb,
                 dens):
    """
    Perform a single time step in the Finite Difference solution for P-SV
    elastic waves.
    """
    dt2 = dt**2
    for i in range(z1, z2):
        for j in range(x1, x2):
            # Step the ux component
            l = 0.5*(lamb[i,j+1] + lamb[i,j])
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxx_p = (l + 2*m)*(ux[t,i,j+1] - ux[t,i,j])/dx + l*0.25*(
                uz[t,i+1,j+1] + uz[t,i+1,j] - uz[t,i-1,j+1] - uz[t,i-1,j])/dz
            l = 0.5*(lamb[i,j-1] + lamb[i,j])
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxx_m = (l + 2*m)*(ux[t,i,j] - ux[t,i,j-1])/dx + l*0.25*(
                uz[t,i+1,j] + uz[t,i+1,j-1] - uz[t,i-1,j] - uz[t,i-1,j-1])/dz
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauxz_p = m*((ux[t,i+1,j] - ux[t,i,j])/dz + 0.25*(
                uz[t,i+1,j+1] + uz[t,i,j+1]- uz[t,i+1,j-1] - uz[t,i,j-1])/dx)
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauxz_m = m*((ux[t,i,j] - ux[t,i-1,j])/dz + 0.25*(
                uz[t,i,j+1] + uz[t,i-1,j+1]- uz[t,i,j-1]  - uz[t,i-1,j-1])/dx)
            ux[tp1,i,j] = 2*ux[t,i,j] - ux[tm1,i,j] + (dt2/dens[i,j])*(
                (tauxx_p - tauxx_m)/dx + (tauxz_p - tauxz_m)/dz)
            # Step the uz component
            l = 0.5*(lamb[i+1,j] + lamb[i,j])
            m = 0.5*(mu[i+1,j] + mu[i,j])
            tauzz_p = (l + 2*m)*(uz[t,i+1,j] - uz[t,i,j])/dz + l*0.25*(
                ux[t,i+1,j+1] + ux[t,i,j+1] - ux[t,i+1,j-1] - ux[t,i,j-1])/dx
            l = 0.5*(lamb[i-1,j] + lamb[i,j])
            m = 0.5*(mu[i-1,j] + mu[i,j])
            tauzz_m = (l + 2*m)*(uz[t,i,j] - uz[t,i-1,j])/dz + l*0.25*(
                ux[t,i,j+1] + ux[t,i-1,j+1] - ux[t,i,j-1] - ux[t,i-1,j-1])/dx
            m = 0.5*(mu[i,j+1] + mu[i,j])
            tauxz_p = m*((uz[t,i,j+1] - uz[t,i,j])/dx + 0.25*(
                ux[t,i+1,j+1] + ux[t,i+1,j] - ux[t,i-1,j+1] - ux[t,i-1,j])/dz)
            m = 0.5*(mu[i,j-1] + mu[i,j])
            tauxz_m = m*((uz[t,i,j] - uz[t,i,j-1])/dx + 0.25*(
                ux[t,i+1,j] + ux[t,i+1,j-1]- ux[t,i-1,j]  - ux[t,i-1,j-1])/dz)
            uz[tp1,i,j] = 2*uz[t,i,j] - uz[tm1,i,j] + (dt2/dens[i,j])*(
                (tauzz_p - tauzz_m)/dz + (tauxz_p - tauxz_m)/dx)
