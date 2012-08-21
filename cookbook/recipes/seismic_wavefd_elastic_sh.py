"""
Perform a 2D finite difference simulation of SH wave propagation in a medium
with a discontinuity (i.e., Moho).

The simulation shows that the SH waves get trapped in the top most layer and
that longer periods travel faster.

.. warning:: Can be very slow on old computers!


"""
import time
import numpy as np
import fatiando as ft

log = ft.log.get()

sources = [ft.seis.wavefd.MexHatSource(0, 20, 1, 2, delay=6)]
shape = (80, 400)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
interface = 30
dens = np.ones(shape)
dens[:interface,:] *= 2700
dens[interface:,:] *= 3100
svel = np.ones(shape)
svel[:interface,:] *= 1000
svel[interface:,:] *= 2000

dt = 0.1
timesteps = ft.seis.wavefd.elastic_sh(spacing, shape, svel, dens, dt, 6500,
    sources, padding=0.8)

ft.vis.ion()
ft.vis.figure(figsize=(10,3))
ft.vis.axis('scaled')
x, z = ft.grd.regular(area, shape)
plot = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    vmin=-10**(-7), vmax=10**(-7))
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
start = time.clock()
for i, u in enumerate(timesteps):
    if i%100 == 0:
        ft.vis.title('time: %g s' % (i*dt))
        plot.set_array(u[0:-1,0:-1].ravel())
        ft.vis.draw()
ft.vis.ioff()
print 'Frames per second (FPS):', float(i)/(time.clock() - start)
ft.vis.show()

#ft.vis.figure(figsize=(10,3))
#ft.vis.axis('scaled')
#x, z = ft.grd.regular(area, shape)
#plot = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    #vmin=-10**(-7), vmax=10**(-7))
#ft.vis.ylim(area[-1], area[-2])
#ft.vis.m2km()
#ft.vis.xlabel("x (km)")
#ft.vis.ylabel("z (km)")
#for i, u in enumerate(timesteps):
    #ft.vis.title('time: %g s' % (i*dt))
    #plot.set_array(u[0:-1,0:-1].ravel())
    #ft.vis.draw()
    #ft.vis.savefig('frames/f%06d.png' % (i + 1), dpi=60)
