"""
Seismic: 2D finite difference simulation of scalar wave propagation.

Difraction example in cylindrical wedge model. Based on:
R. M. Alford, K. R. Kelly and D. M. Boore -
Accuracy of finite-difference modeling of the acoustic wave equation.
Geophysics  1974
"""
import numpy as np
from matplotlib import animation
from fatiando.seismic import wavefd
from fatiando.vis import mpl

# Set the parameters of the finite difference grid
shape = (200, 200)
ds = 100.  # spacing
area = [0, shape[0] * ds, 0, shape[1] * ds]
# Set the parameters of the finite difference grid
velocity = np.zeros(shape) + 6000.
velocity[100:, 100:] = 0.
fc = 15.
simulation = wavefd.Scalar(velocity, (ds, ds))
simulation.add_point_source((125, 75), -1*wavefd.Gauss(1., fc))
duration = 2.1
maxit = int(duration / simulation.dt)
maxt = duration

# This part makes an animation using matplotlibs animation API
background = (velocity - 6000) * 10 ** -3
fig = mpl.figure(figsize=(8, 6))
mpl.subplots_adjust(right=0.98, left=0.11, hspace=0.5, top=0.93)
mpl.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
wavefield = mpl.imshow(np.zeros_like(velocity), extent=area,
                       cmap=mpl.cm.gray_r, vmin=-0.05, vmax=0.05)
mpl.points([75*ds, 125*ds], '^b', size=8)  # seismometer position
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
mpl.subplot2grid((4, 3), (3, 0), colspan=3)
seismogram1, = mpl.plot([], [], '-k')
mpl.xlim(0, duration)
mpl.ylim(-0.05, 0.05)
mpl.ylabel('Amplitude')
mpl.xlabel('Time (s)')
times = np.linspace(0, maxt, maxit)
# This function updates the plot every few timesteps

simulation.run(maxit)
seismogram = simulation[:, 125, 75]  # (time, z and x) shape


def animate(i):
    u = simulation[i]
    seismogram1.set_data(times[:i], seismogram[:i])
    wavefield.set_array(background[::-1] + u[::-1])
    return wavefield, seismogram1

anim = animation.FuncAnimation(
    fig, animate, frames=maxit, interval=1)
mpl.show()
