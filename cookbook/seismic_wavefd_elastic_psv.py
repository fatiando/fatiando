"""
Seismic: 2D finite difference simulation of elastic P and SV wave propagation
"""
from matplotlib import animation
import numpy as np
from fatiando import seismic, gridder, vis

# Make a wave source from a mexican hat wavelet
sources = [seismic.wavefd.MexHatSource(25, 25, 100, 0.5, delay=1.5)]
# Set the parameters of the finite difference grid
shape = (100, 100)
spacing = (500, 500)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
dens = 2700*np.ones(shape)
svel = 3000*np.ones(shape)
pvel = 4000*np.ones(shape)

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop below
dt = 0.05
maxit = 300
timesteps = seismic.wavefd.elastic_psv(spacing, shape, pvel, svel, dens, dt,
    maxit, sources, sources, padding=0.5)

# This part makes an animation using matplotlibs animation API
vmin, vmax = -1*10**(-4), 1*10**(-4)
x, z = gridder.regular(area, shape)
fig = vis.mpl.figure()
ax_x = vis.mpl.subplot(1, 2, 1)
vis.mpl.axis('scaled')
# Start with everything zero and grab the plot so that it can be updated later
wavefieldx = vis.mpl.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    vmax=vmax)
# Make z positive down
vis.mpl.ylim(area[-1], area[-2])
vis.mpl.m2km()
vis.mpl.xlabel("x (km)")
vis.mpl.ylabel("z (km)")
# Do the same for z component of the displacement
ax_z = vis.mpl.subplot(1, 2, 2)
vis.mpl.axis('scaled')
wavefieldz = vis.mpl.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    vmax=vmax)
# Make z positive down
vis.mpl.ylim(area[-1], area[-2])
vis.mpl.m2km()
vis.mpl.xlabel("x (km)")
vis.mpl.ylabel("z (km)")
# This function updates the plot every few timesteps
steps_per_frame = 10
def animate(i):
    for t, update in enumerate(timesteps):
        if t == steps_per_frame - 1:
            ux, uz = update
            ax_x.set_title('ux time: %0.1f s' % (i*steps_per_frame*dt))
            wavefieldx.set_array(ux[0:-1,0:-1].ravel())
            ax_z.set_title('uz time: %0.1f s' % (i*steps_per_frame*dt))
            wavefieldz.set_array(ux[0:-1,0:-1].ravel())
            break
    return wavefieldx, wavefieldz
anim = animation.FuncAnimation(fig, animate,
    frames=maxit/steps_per_frame, interval=1, blit=False)
#anim.save('p_and_sv_waves.mp4', fps=10)
vis.mpl.show()
