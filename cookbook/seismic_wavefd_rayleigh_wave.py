"""
Seismic: 2D finite difference simulation of elastic P and SV wave propagation
in a medium with a discontinuity (i.e., Moho), generating Rayleigh waves
"""
import numpy as np
from matplotlib import animation
from fatiando import gridder
from fatiando.seismic import wavefd
from fatiando.vis import mpl

# Set the parameters of the finite difference grid
shape = (150, 900)
area = [0, 540000, 0, 90000]

# Make a density and wave velocity model
density = 2400 * np.ones(shape)
svel = 3700 * np.ones(shape)
pvel = 6600 * np.ones(shape)
moho = 50
density[moho:] = 2800
svel[moho:] = 4300
pvel[moho:] = 7500
mu = wavefd.lame_mu(svel, density)
lamb = wavefd.lame_lamb(pvel, svel, density)

# Make a wave source from a mexican hat wavelet for the x and z directions
sources = [
    [wavefd.MexHatSource(10000, 10000, area, shape, 100000, 0.5, delay=2)],
    [wavefd.MexHatSource(10000, 10000, area, shape, 100000, 0.5, delay=2)]]

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop below
dt = wavefd.maxdt(area, shape, pvel.max())
duration = 130
maxit = int(duration / dt)
stations = [[400000, 0]]
snapshots = int(1. / dt)
simulation = wavefd.elastic_psv(lamb, mu, density, area, dt, maxit, sources,
                                stations, snapshots, padding=70, taper=0.005,
                                xz2ps=True)

# This part makes an animation using matplotlibs animation API
background = 10 ** -5 * ((density - density.min()) / density.max())
fig = mpl.figure(figsize=(10, 8))
mpl.subplots_adjust(right=0.98, left=0.11, hspace=0.3, top=0.93)
mpl.subplot(3, 1, 1)
mpl.title('x seismogram')
xseismogram, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-0.05, 0.05)
mpl.ylabel('Amplitude')
mpl.subplot(3, 1, 2)
mpl.title('z seismogram')
zseismogram, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-0.05, 0.05)
mpl.ylabel('Amplitude')
ax = mpl.subplot(3, 1, 3)
mpl.title('time: 0.0 s')
wavefield = mpl.imshow(background, extent=area, cmap=mpl.cm.gray_r,
                       vmin=-0.00001, vmax=0.00001)
mpl.points(stations, '^b', size=8)
mpl.text(500000, 20000, 'Crust')
mpl.text(500000, 60000, 'Mantle')
fig.text(0.7, 0.31, 'Seismometer')
mpl.xlim(area[:2])
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
times = np.linspace(0, dt * maxit, maxit)
# This function updates the plot every few timesteps


def animate(i):
    t, p, s, xcomp, zcomp = simulation.next()
    mpl.title('time: %0.1f s' % (times[t]))
    wavefield.set_array((background + p + s)[::-1])
    xseismogram.set_data(times[:t + 1], xcomp[0][:t + 1])
    zseismogram.set_data(times[:t + 1], zcomp[0][:t + 1])
    return wavefield, xseismogram, zseismogram


anim = animation.FuncAnimation(
    fig, animate, frames=maxit / snapshots, interval=1)
mpl.show()
