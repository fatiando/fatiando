"""
Perform a 2D finite difference simulation of SH wave propagation.
"""
import time
import numpy as np
import fatiando as ft

log = ft.log.get()

sources = [ft.seis.wavefd.MexHatSource(0, 10, 1, 5, delay=0.4)]
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

timesteps = ft.seis.wavefd.elastic_sh(spacing, shape, svel, dens, 0.1, 10000,
    sources)

ft.vis.ion()
ft.vis.figure()
ft.vis.axis('scaled')
x, z = ft.grd.regular(area, shape)
plot = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=-0.5, vmax=0.5)
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
start = time.clock()
for i, u in enumerate(timesteps):
    if i%100 == 0:
        ft.vis.title('it: %d' % (i))
        plot.set_array(u.ravel())
        plot.autoscale()
        ft.vis.draw()
ft.vis.ioff()
print 'Frames per second (FPS):', float(i + 1)/(time.clock() - start)
ft.vis.show()
