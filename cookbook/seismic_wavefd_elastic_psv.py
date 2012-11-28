"""
Seismic: 2D finite difference simulation of elastic P and SV wave propagation in a
medium with a discontinuity (i.e., Moho).

WARNING: Can be very slow!
"""
from matplotlib import animation
import numpy as np
import fatiando as ft

log = ft.logger.get()

# Make some seismic sources using the mexican hat wavelet
sources = [ft.seismic.wavefd.MexHatSource(4+i, 20+i, 50, 0.5, delay=1.5 + 0.25*i)
           for i in xrange(10)]
# Make the velocity and density models
shape = (80, 400)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
moho_index = 30
moho = moho_index*spacing[0]
dens = np.ones(shape)
dens[:moho_index,:] *= 2700.
dens[moho_index:,:] *= 3100.
pvel = np.ones(shape)
pvel[:moho_index,:] *= 4000.
pvel[moho_index:,:] *= 8000.
svel = np.ones(shape)
svel[:moho_index,:] *= 3000.
svel[moho_index:,:] *= 6000.

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 4200
timesteps = ft.seismic.wavefd.elastic_psv(spacing, shape, pvel, svel, dens, dt,
    maxit, sources, sources, padding=0.8)

# This part makes an animation using matplotlibs animation API
rec = 350 # The grid node used to record the seismogram
vmin, vmax = -10*10**(-4), 10*10**(-4)
fig = ft.vis.figure(figsize=(16,7))
ft.vis.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
# A plot for the ux field
plotx = ft.vis.subplot(3, 2, 1)
xseismogram, = ft.vis.plot([0], [0], '-k')
ft.vis.xlim(0, dt*maxit)
ft.vis.ylim(vmin*10.**(6), vmax*10.**(6))
ft.vis.xlabel("Time (s)")
ft.vis.ylabel("Amplitude ($\mu$m)")
ft.vis.subplot(3, 2, 3)
ft.vis.axis('scaled')
x, z = ft.gridder.regular(area, shape)
xwavefield = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    vmin=vmin, vmax=vmax)
ft.vis.plot([rec*spacing[1]], [2000], '^b')
ft.vis.hlines([moho], 0, area[1], 'k', '-')
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
# A plot for the uz field
plotz = ft.vis.subplot(3, 2, 2)
zseismogram, = ft.vis.plot([0], [0], '-k')
ft.vis.xlim(0, dt*maxit)
ft.vis.ylim(vmin*10.**(6), vmax*10.**(6))
ft.vis.xlabel("Time (s)")
ft.vis.ylabel("Amplitude ($\mu$m)")
ft.vis.subplot(3, 2, 4)
ft.vis.axis('scaled')
x, z = ft.gridder.regular(area, shape)
zwavefield = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    vmin=vmin, vmax=vmax)
ft.vis.plot([rec*spacing[1]], [2000], '^b')
ft.vis.hlines([moho], 0, area[1], 'k', '-')
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
# And a plot for the particle movement in the seismic station
ax = ft.vis.subplot(3, 1, 3)
ft.vis.title("Particle movement")
ft.vis.axis('scaled')
particle_movement, = ft.vis.plot([0], [0], '-k')
ft.vis.xlim(vmin*10.**(6), vmax*10.**(6))
ft.vis.ylim(vmax*10.**(6), vmin*10.**(6))
ft.vis.xlabel("ux ($\mu$m)")
ft.vis.ylabel("uz ($\mu$m)")
ax.set_xticks(ax.get_xticks()[1:-1])
ax.set_yticks(ax.get_yticks()[1:-1])
# Record the amplitudes at the seismic station
times = []
addtime = times.append
xamps = []
addxamp = xamps.append
zamps = []
addzamp = zamps.append
# This function updates the plot every few timesteps
steps_per_frame = 100
def animate(i):
    for t, update in enumerate(timesteps):
        ux, uz = update
        addxamp(10.**(6)*ux[0, rec])
        addzamp(10.**(6)*uz[0, rec])
        addtime(dt*(t + i*steps_per_frame))
        if t == steps_per_frame - 1:
            break
    plotx.set_title('x component | time: %0.1f s' % (i*steps_per_frame*dt))
    xseismogram.set_data(times, xamps)
    xwavefield.set_array(ux[0:-1,0:-1].ravel())
    plotz.set_title('z component | time: %0.1f s' % (i*steps_per_frame*dt))
    zseismogram.set_data(times, zamps)
    zwavefield.set_array(uz[0:-1,0:-1].ravel())
    particle_movement.set_data(xamps, zamps)
    return xwavefield, xseismogram, zwavefield, zseismogram, particle_movement
anim = animation.FuncAnimation(fig, animate, interval=1, blit=False,
    frames=maxit/steps_per_frame)
#anim.save('rayleigh_wave.mp4', fps=10)
ft.vis.show()
