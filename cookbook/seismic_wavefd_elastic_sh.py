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
mu = 2700*(3000**2)*np.ones(shape)

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 1500
timesteps = seismic.wavefd.elastic_sh(spacing, shape, mu, dens, dt, maxit,
    sources, padding=50, taper=0.005)

# This part makes an animation using matplotlibs animation API
vmin, vmax = -1*10**(-5), 1*10**(-5)
fig = mpl.figure(figsize=(14, 5))
mpl.subplot(1, 2, 1)
x, z = gridder.regular(area, shape)
# Start with everything zero and grab the plot so that it can be updated later
initial = timesteps.next()
wavefield = mpl.imshow(initial, vmin=vmin, vmax=vmax)
mpl.xlabel("x (km)")
mpl.ylabel("z (km)")
mpl.set_area([0, initial.shape[1], initial.shape[0], 0])
mpl.subplot(1, 2, 2)
energy, = mpl.plot([],[],'-k')
mpl.xlim(0, maxit)
mpl.ylim(-10**(-4), 10**(-4))
values = []
# This function updates the plot every few timesteps
steps_per_frame = 10
def animate(i):
    for t, u in enumerate(timesteps):
        values.append(u[0, u.shape[1]/2])
        if t == steps_per_frame - 1:
            mpl.title('time: %0.1f s' % (i*steps_per_frame*dt))
            wavefield.set_array(u)
            energy.set_data(range(len(values)),values)
            break
    return wavefield, energy
anim = animation.FuncAnimation(fig, animate,
    frames=maxit/steps_per_frame, interval=10, blit=False)
#anim.save('sh_wave.mp4', fps=10)
mpl.show()

