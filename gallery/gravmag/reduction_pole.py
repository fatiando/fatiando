"""
Reduction to the pole
---------------------

If the direction of magnetization is known, you can reduce a measured total
field magnetic anomaly to the pole. Function
:func:`fatiando.gravmag.transform.reduce_to_pole` implements the reduction
using the FFT and allows using a magnetization direction that is different from
the geomagnetic field direction. This example shows how to use it in this case.
Use ``sinc=inc`` and ``sdec=dec`` if there is only induced magnetization.

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
from fatiando.gravmag import prism, transform
from fatiando.mesher import Prism
from fatiando import gridder, utils

# Create some synthetic magnetic data with a total magnetization that is
# different from the geomagnetic field (so there is remanent magnetization or
# some demagnetizing effect)
inc, dec = -60, 23  # Geomagnetic field direction
sinc, sdec = -30, -20  # Source magnetization direction
mag = utils.ang2vec(1, sinc, sdec)
model = [Prism(-1500, 1500, -500, 500, 0, 2000, {'magnetization': mag})]
area = (-7e3, 7e3, -7e3, 7e3)
shape = (100, 100)
x, y, z = gridder.regular(area, shape, z=-300)
data = prism.tf(x, y, z, model, inc, dec)

# Reduce to the pole
data_at_pole = transform.reduce_to_pole(x, y, data, shape, inc, dec, sinc,
                                        sdec)

# Make some plots
plt.figure(figsize=(8, 6))

ax = plt.subplot(1, 2, 1)
ax.set_title('Original data')
ax.set_aspect('equal')
tmp = ax.tricontourf(y/1000, x/1000, data, 30, cmap='RdBu_r')
plt.colorbar(tmp, pad=0.1, aspect=30, orientation='horizontal').set_label('nT')
ax.set_xlabel('y (km)')
ax.set_ylabel('x (km)')
ax.set_xlim(area[2]/1000, area[3]/1000)
ax.set_ylim(area[0]/1000, area[1]/1000)

ax = plt.subplot(1, 2, 2)
ax.set_title('Reduced to the pole')
ax.set_aspect('equal')
tmp = ax.tricontourf(y/1000, x/1000, data_at_pole, 30, cmap='RdBu_r')
plt.colorbar(tmp, pad=0.1, aspect=30, orientation='horizontal').set_label('nT')
ax.set_xlabel('y (km)')
ax.set_xlim(area[2]/1000, area[3]/1000)
ax.set_ylim(area[0]/1000, area[1]/1000)

plt.tight_layout()
plt.show()
