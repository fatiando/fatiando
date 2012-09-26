"""
Seis: 2D finite difference simulation of elastic P and SV wave propagation
"""
from matplotlib import animation
import numpy as np
import fatiando as ft

log = ft.log.get()

# Make a wave source from a mexican hat wavelet
sources = [ft.seis.wavefd.MexHatSource(25, 25, 100, 0.5, delay=1.5)]
# Set the parameters of the finite difference grid
shape = (50, 50)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
dens = 2700*np.ones(shape)
svel = 3000*np.ones(shape)
pvel = 4000*np.ones(shape)

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 300
timesteps = ft.seis.wavefd.elastic_psv(spacing, shape, pvel, svel, dens, dt,
    maxit, sources, sources, padding=0.5)

# This part makes an animation using matplotlibs animation API
vmin, vmax = -1*10**(-4), 1*10**(-4)
x, z = ft.grd.regular(area, shape)
fig = ft.vis.figure()
ax_x = ft.vis.subplot(1, 2, 1)
ft.vis.axis('scaled')
# Start with everything zero and grab the plot so that it can be updated later
wavefieldx = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    vmax=vmax)
# Make z positive down
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
# Do the same for z component of the displacement
ax_z = ft.vis.subplot(1, 2, 2)
ft.vis.axis('scaled')
wavefieldz = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape, vmin=vmin,
    vmax=vmax)
# Make z positive down
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
# This function updates the plot every few timesteps
steps_per_frame = 10
def animate(i):
    for t, update in enumerate(timesteps):
        if t == steps_per_frame - 1:
            break
    ux, uz = update
    ax_x.set_title('ux time: %0.1f s' % (i*steps_per_frame*dt))
    wavefieldx.set_array(ux[0:-1,0:-1].ravel())
    ax_z.set_title('uz time: %0.1f s' % (i*steps_per_frame*dt))
    wavefieldz.set_array(ux[0:-1,0:-1].ravel())
    return wavefieldx, wavefieldz
anim = animation.FuncAnimation(fig, animate,
    frames=maxit/steps_per_frame, interval=1, blit=False)
#anim.save('p_and_sv_waves.mp4', fps=10)
ft.vis.show()
