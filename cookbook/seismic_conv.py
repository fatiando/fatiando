"""
Seismic: Synthetic convolutional seismogram for a simple two layer velocity
model
"""
import numpy as np
import matplotlib.pyplot as plt
from fatiando.seismic import conv
from fatiando.vis import mpl
# model parameters
n_samples, n_traces = [600, 20]
rock_grid = 1500.*np.ones((n_samples, n_traces))
rock_grid[300:, :] = 2500.
# synthetic calculation
vel_l = conv.depth_2_time(rock_grid, rock_grid, dt=2.e-3, dz=1.)
rho_l = np.ones(np.shape(vel_l))
rc = conv.reflectivity(vel_l, rho_l)
synt = conv.convolutional_model(rc, 30., conv.rickerwave, dt=2.e-3)
# plot input model
plt.figure()
plt.subplot(3, 1, 1)
plt.ylabel('Depth (m)')
plt.title("Depth Vp model", fontsize=13, family='sans-serif', weight='bold')
plt.imshow(rock_grid, extent=[0, n_traces, n_samples, 0],
           cmap=mpl.pyplot.cm.bwr, aspect='auto', origin='upper')
# plot resulted seismogram using wiggle
plt.subplot(3, 1, 2)
mpl.seismic_wiggle(synt, dt=2.e-3)
mpl.seismic_image(synt, dt=2.e-3, cmap=mpl.pyplot.cm.jet, aspect='auto')
plt.ylabel('time (seconds)')
plt.title("Convolutional seismogram", fontsize=13, family='sans-serif',
          weight='bold')

# plot resulted seismogram using wiggle over Vp model
plt.subplot(3, 1, 3)
mpl.seismic_image(vel_l, dt=2.e-3, cmap=mpl.pyplot.cm.jet, aspect='auto')
mpl.seismic_wiggle(synt, dt=2.e-3)
plt.ylabel('time (seconds)')
plt.title("Convolutional seismogram over Vp model", fontsize=13,
          family='sans-serif', weight='bold')
plt.show()
