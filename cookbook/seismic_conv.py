"""
Synthetic convolutional seismogram for a simple two layer velocity model
"""
import numpy as np
import gen_models
from fatiando.seismic import conv
from fatiando.vis import mpl
#model parameters
n_samples, n_traces = [600, 500]
rock_grid = 1500.*np.ones((n_samples, n_traces))
rock_grid[300:,:]= 2500.
#synthetic calculation
[vel_l,TWT_ts,rho_l]=conv.depth_2_time(n_samples, n_traces, rock_grid)
synt=conv.seismic_convolutional_model(n_traces, vel_l, 30., conv.rickerwave)
# plot input model
mpl.figure()
mpl.subplot(3,1,1)
mpl.ylabel('Depth (m)')
mpl.title("Depth Vp model", fontsize=13, family='sans-serif', weight='bold')
mpl.imshow(rock_grid, extent=[0,n_samples, n_traces, 0],cmap=mpl.pyplot.cm.bwr,
           aspect='auto', origin='upper')
# plot resulted seismogram using wiggle
mpl.subplot(3, 1, 2)
mpl.seismic_wiggle(synt, dt = 2.e-3, scale = 10**1)
mpl.seismic_image(synt, dt = 2.e-3, cmap=mpl.pyplot.cm.jet, aspect='auto')
mpl.ylabel('time (seconds)')
mpl.title("Convolutional seismogram", fontsize=13, family='sans-serif',
          weight='bold')
# plot resulted seismogram using wiggle over Vp model
mpl.subplot(3, 1, 3)
mpl.imshow(vel_l, extent=[0,n_samples, n_traces, 0],cmap=mpl.pyplot.cm.bwr,
           aspect='auto', origin='upper')
mpl.seismic_wiggle(synt, dt = 2.e-3, scale = 10)
mpl.ylabel('time (seconds)')
mpl.title("Convolutional seismogram over Vp model", fontsize=13, family='sans-serif',
          weight='bold')
mpl.show()
