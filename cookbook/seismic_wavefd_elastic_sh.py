"""
Seismic: 2D finite difference simulation of elastic SH wave propagation
"""
import numpy as np
from matplotlib import animation
from fatiando import seismic, gridder
from fatiando.vis import mpl

# Make a wave source from a mexican hat wavelet
sources = [seismic.wavefd.MexHatSource(25, 25, 100, 0.5, delay=1.5)]
# Set the parameters of the finite difference grid
shape = (100, 100)
spacing = (500, 500)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
dens = 2700*np.ones(shape)
mu = (2700*3000**2)*np.ones(shape)

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 400
timesteps = seismic.wavefd.elastic_sh(spacing, shape, mu, dens, dt, maxit,
    sources, padding=0.5)

#x, z = gridder.regular(area, shape)
#for t, u in enumerate(timesteps):
    #if t%100 == 0:
        #mpl.figure()
        #mpl.axis('scaled')
        #mpl.pcolor(x, z, u.ravel(), shape)
        #mpl.ylim(area[-1], area[-2])


# This part makes an animation using matplotlibs animation API
vmin, vmax = -1*10**(-4), 1*10**(-4)
fig = mpl.figure()
mpl.axis('scaled')
x, z = gridder.regular(area, shape)
# Start with everything zero and grab the plot so that it can be updated later
#wavefield = mpl.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    #vmax=vmax)
wavefield = mpl.pcolor(x, z, np.zeros(shape).ravel(), shape)
# Make z positive down
mpl.ylim(area[-1], area[-2])
mpl.m2km()
mpl.xlabel("x (km)")
mpl.ylabel("z (km)")
# This function updates the plot every few timesteps
steps_per_frame = 100
def animate(i):
    for t, u in enumerate(timesteps):
        if t == steps_per_frame - 1:
            break
        mpl.title('time: %0.1f s' % (i*steps_per_frame*dt))
        wavefield.set_array(u[0:-1,0:-1].ravel())
    return wavefield,
anim = animation.FuncAnimation(fig, animate,
    frames=maxit/steps_per_frame, interval=1, blit=False)
#anim.save('sh_wave.mp4', fps=10)
mpl.show()

