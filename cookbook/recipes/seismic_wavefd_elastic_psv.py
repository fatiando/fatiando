"""
Perform a 2D finite difference simulation of P and SV wave propagation in a
medium with a discontinuity (i.e., Moho).

.. warning:: Can be very slow on old computers!


"""
import time
import numpy as np
import fatiando as ft

log = ft.log.get()

sources = [ft.seis.wavefd.MexHatSource(4+i, 20+i, 50, 0.5, delay=1.5 + 0.25*i)
           for i in xrange(10)]
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

dt = 0.05
maxit = 4200
timesteps = ft.seis.wavefd.elastic_psv(spacing, shape, pvel, svel, dens, dt,
    maxit, sources, sources, padding=0.8)

rec = 350
vmin, vmax = -10*10**(-4), 10*10**(-4)
# This part makes an animation by updating the plot every few iterations
ft.vis.ion()
ft.vis.figure(figsize=(16,7))
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
x, z = ft.grd.regular(area, shape)
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
x, z = ft.grd.regular(area, shape)
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
ft.vis.ylim(vmin*10.**(6), vmax*10.**(6))
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
start = time.clock()
# This part animates the plot
for i, update in enumerate(timesteps):
    ux, uz = update
    addxamp(10.**(6)*ux[0, rec])
    addzamp(10.**(6)*uz[0, rec])
    addtime(dt*i)
    if i%100 == 0:
        plotx.set_title('x component | time: %0.1f s' % (i*dt))
        xseismogram.set_ydata(xamps)
        xseismogram.set_xdata(times)
        xwavefield.set_array(ux[0:-1,0:-1].ravel())
        plotz.set_title('z component | time: %0.1f s' % (i*dt))
        zseismogram.set_ydata(zamps)
        zseismogram.set_xdata(times)
        zwavefield.set_array(uz[0:-1,0:-1].ravel())
        particle_movement.set_xdata(xamps)
        particle_movement.set_ydata(zamps)
        ft.vis.draw()
# Comment the above and uncomment bellow to save snapshots of the simulation
# If you don't want the animation to show on the screen, comment ft.vis.ion()
# above
#for i, update in enumerate(timesteps):
    #ux, uz = update
    #addxamp(10.**(6)*ux[0, rec])
    #addzamp(10.**(6)*uz[0, rec])
    #addtime(dt*i)
    #if i%10 == 0:
        #plotx.set_title('x component | time: %0.1f s' % (i*dt))
        #xseismogram.set_ydata(xamps)
        #xseismogram.set_xdata(times)
        #xwavefield.set_array(ux[0:-1,0:-1].ravel())
        #plotz.set_title('z component | time: %0.1f s' % (i*dt))
        #zseismogram.set_ydata(zamps)
        #zseismogram.set_xdata(times)
        #zwavefield.set_array(uz[0:-1,0:-1].ravel())
        #particle_movement.set_xdata(xamps)
        #particle_movement.set_ydata(zamps)
        #ft.vis.draw()
        #ft.vis.savefig('frames-psv/f%06d.png' % ((i)/10 + 1), dpi=60)
ft.vis.ioff()
print 'Frames per second (FPS):', float(i)/(time.clock() - start)
ft.vis.show()
