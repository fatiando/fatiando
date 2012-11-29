"""
Seismic: 2D finite difference simulation of elastic SH wave propagation in a
medium with a discontinuity (i.e., Moho).

The simulation shows that the SH waves get trapped in the top most layer and
generate Love waves.

WARNING: Can be very slow!
"""
import numpy as np
from matplotlib import animation
from fatiando import seismic, logger, gridder, vis

log = logger.get()

# Make a wave source from a mexican hat wavelet
sources = [seismic.wavefd.MexHatSource(4, 20, 100, 0.5, delay=1.5),
           seismic.wavefd.MexHatSource(6, 22, 100, 0.5, delay=1.75),
           seismic.wavefd.MexHatSource(8, 24, 100, 0.5, delay=2)]
# Set the parameters of the finite difference grid
shape = (80, 400)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
moho_index = 30
moho = moho_index*spacing[0]
dens = np.ones(shape)
dens[:moho_index,:] *= 2700
dens[moho_index:,:] *= 3100
svel = np.ones(shape)
svel[:moho_index,:] *= 3000
svel[moho_index:,:] *= 6000

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 4200
timesteps = seismic.wavefd.elastic_sh(spacing, shape, svel, dens, dt, maxit,
    sources, padding=0.8)

# This part makes an animation using matplotlibs animation API
rec = 300 # The grid node used to record the seismogram
vmin, vmax = -3*10**(-4), 3*10**(-4)
fig = vis.mpl.figure(figsize=(10,6))
vis.mpl.subplots_adjust(left=0.1, right=0.98)
vis.mpl.subplot(2, 1, 2)
vis.mpl.axis('scaled')
x, z = gridder.regular(area, shape)
wavefield = vis.mpl.pcolor(x, z, np.zeros(shape).ravel(), shape,
    vmin=vmin, vmax=vmax)
vis.mpl.plot([rec*spacing[1]], [2000], '^b')
vis.mpl.hlines([moho], 0, area[1], 'k', '-')
vis.mpl.text(area[1] - 35000, moho + 10000, 'Moho')
vis.mpl.text(area[1] - 90000, 15000,
    r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (2.7, 3))
vis.mpl.text(area[1] - 90000, area[-1] - 10000,
    r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (3.1, 6))
vis.mpl.ylim(area[-1], area[-2])
vis.mpl.m2km()
vis.mpl.xlabel("x (km)")
vis.mpl.ylabel("z (km)")
vis.mpl.subplot(2, 1, 1)
seismogram, = vis.mpl.plot([], [], '-k')
vis.mpl.xlim(0, dt*maxit)
vis.mpl.ylim(vmin*10.**(6), vmax*10.**(6))
vis.mpl.xlabel("Time (s)")
vis.mpl.ylabel("Amplitude ($\mu$m)")
times = []
addtime = times.append
amps = []
addamp = amps.append
# This function updates the plot every few timesteps
steps_per_frame = 100
def animate(i):
    # i is the number of the animation frame
    for t, u in enumerate(timesteps):
        addamp(10.**(6)*u[0, rec])
        addtime(dt*(t + i*steps_per_frame))
        if t == steps_per_frame - 1:
            break
    vis.mpl.title('time: %0.1f s' % (i*steps_per_frame*dt))
    seismogram.set_data(times, amps)
    wavefield.set_array(u[0:-1,0:-1].ravel())
    return seismogram, wavefield
anim = animation.FuncAnimation(fig, animate, frames=maxit/steps_per_frame,
    interval=1, blit=False)
#anim.save('love_wave.mp4', fps=10)
vis.mpl.show()
