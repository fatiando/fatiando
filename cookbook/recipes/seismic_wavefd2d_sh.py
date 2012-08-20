"""
Perform a 2D finite difference simulation of SH wave propagation.
"""
import time
import numpy as np
from matplotlib import pyplot as pl
import fatiando as ft

log = ft.log.get()

sources = [ft.seis.wavefd2d.MexHatSource(50, 50, 1, 0.1, delay=0.4)]
shape = (100, 100)
spacing = (0.5, 0.5)
dens = 2670*np.ones(shape)
lamb, mu = ft.seis.wavefd2d.lame(0, 10., dens)

timesteps = ft.seis.wavefd2d.elastic_sh(spacing, shape, mu, dens, 0.01, 10000,
    sources)

pl.ion()
pl.figure()
pl.axis('scaled')
#plot = pl.pcolor(np.zeros(shape), vmin=-0.1, vmax=0.1)
plot = pl.pcolor(np.zeros(shape))
pl.colorbar()
pl.xlim(0, shape[1])
pl.ylim(shape[0], 0)
start = time.clock()
for i, u in enumerate(timesteps):
    if i%100 == 0:
        pl.title('it: %d' % (i))
        plot.set_array(u.ravel())
        plot.autoscale()
        pl.draw()
pl.ioff()
print 'Frames per second (FPS):', float(i + 1)/(time.clock() - start)
pl.show()
