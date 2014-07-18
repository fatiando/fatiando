"""
Seismic: 2D finite difference simulation of elastic P and SV wave propagation
"""
import numpy as np
from matplotlib import animation
from fatiando import gridder
from fatiando.seismic import wavefd
from fatiando.vis import mpl

# Set the parameters of the finite difference grid
shape = (200, 200)
area = [0, 60000, 0, 60000]
# Make a density and S wave velocity model
density = 2400 * np.ones(shape)
pvel = 6600
svel = 3700
mu = wavefd.lame_mu(svel, density)
lamb = wavefd.lame_lamb(pvel, svel, density)

# Make a wave source from a mexican hat wavelet that vibrates in the x and z
# directions equaly
sources = [[wavefd.MexHatSource(30000, 40000, area, shape, 10000, 1, delay=1)],
           [wavefd.MexHatSource(30000, 40000, area, shape, 10000, 1, delay=1)]]

# Get the iterator for the simulation
dt = wavefd.maxdt(area, shape, pvel)
duration = 20
maxit = int(duration / dt)
stations = [[55000, 0]]  # x, z coordinate of the seismometer
snapshot = int(0.5 / dt)  # Plot a snapshot of the simulation every 0.5 seconds
simulation = wavefd.elastic_psv(lamb, mu, density, area, dt, maxit, sources,
                                stations, snapshot, padding=50, taper=0.01,
                                xz2ps=True)

# This part makes an animation using matplotlibs animation API
fig = mpl.figure(figsize=(12, 5))
mpl.subplot(2, 2, 2)
mpl.title('x component')
xseismogram, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-10 ** (-3), 10 ** (-3))
mpl.subplot(2, 2, 4)
mpl.title('z component')
zseismogram, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-10 ** (-3), 10 ** (-3))
mpl.subplot(1, 2, 1)
# Start with everything zero and grab the plot so that it can be updated later
wavefield = mpl.imshow(np.zeros(shape), extent=area, vmin=-10 ** -6,
                       vmax=10 ** -6, cmap=mpl.cm.gray_r)
mpl.points(stations, '^k')
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
times = np.linspace(0, maxit * dt, maxit)
# This function updates the plot every few timesteps


def animate(i):
    """
    Simulation will yield panels corresponding to P and S waves because
    xz2ps=True
    """
    t, p, s, xcomp, zcomp = simulation.next()
    mpl.title('time: %0.1f s' % (times[t]))
    wavefield.set_array((p + s)[::-1])
    xseismogram.set_data(times[:t + 1], xcomp[0][:t + 1])
    zseismogram.set_data(times[:t + 1], zcomp[0][:t + 1])
    return wavefield, xseismogram, zseismogram


anim = animation.FuncAnimation(
    fig, animate, frames=maxit / snapshot, interval=1)
# anim.save('psv_wave.mp4', fps=20, dpi=200, bitrate=4000)
mpl.show()
