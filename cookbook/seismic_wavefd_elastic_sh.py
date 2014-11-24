"""
Seismic: 2D finite difference simulation of elastic SH wave propagation
"""
import numpy as np
from matplotlib import animation
from fatiando import gridder
from fatiando.seismic import wavefd
from fatiando.vis import mpl

# Set the parameters of the finite difference grid
shape = (150, 150)
area = [0, 60000, 0, 60000]
# Make a density and S wave velocity model
density = 2400 * np.ones(shape)
velocity = 3700
mu = wavefd.lame_mu(velocity, density)

# Make a wave source from a mexican hat wavelet
sources = [wavefd.MexHatSource(30000, 15000, area, shape, 100, 1, delay=2)]

# Get the iterator for the simulation
dt = wavefd.maxdt(area, shape, velocity)
duration = 20
maxit = int(duration / dt)
stations = [[50000, 0]]  # x, z coordinate of the seismometer
snapshot = int(0.5 / dt)  # Plot a snapshot of the simulation every 0.5 seconds
simulation = wavefd.elastic_sh(mu, density, area, dt, maxit, sources, stations,
                               snapshot, padding=50, taper=0.01)

# This part makes an animation using matplotlibs animation API
fig = mpl.figure(figsize=(14, 5))
ax = mpl.subplot(1, 2, 2)
mpl.title('Wavefield')
# Start with everything zero and grab the plot so that it can be updated later
wavefield_plt = mpl.imshow(np.zeros(shape), extent=area, vmin=-10 ** (-5),
                           vmax=10 ** (-5), cmap=mpl.cm.gray_r)
mpl.points(stations, '^b')
mpl.xlim(area[:2])
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.subplot(1, 2, 1)
seismogram_plt, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-10 ** (-4), 10 ** (-4))
mpl.xlabel('time (s)')
mpl.ylabel('Amplitude')
times = np.linspace(0, duration, maxit)
# Update the plot everytime the simulation yields


def animate(i):
    """
    Grab the iteration number, displacment panel and seismograms
    """
    t, u, seismograms = simulation.next()
    mpl.title('time: %0.1f s' % (times[t]))
    wavefield_plt.set_array(u[::-1])  # Revert the z axis so that 0 is top
    seismogram_plt.set_data(times[:t + 1], seismograms[0][:t + 1])
    return wavefield_plt, seismogram_plt


anim = animation.FuncAnimation(
    fig, animate, frames=maxit / snapshot, interval=1)
# anim.save('sh_wave.mp4', fps=10, dpi=200, bitrate=4000)
mpl.show()
