"""
Perform a 2D finite difference simulation of SH wave propagation in a medium
with a discontinuity (i.e., Moho).

The simulation shows that the SH waves get trapped in the top most layer and
that longer periods travel faster.

WARNING: Can be very slow on old computers!

"""
import time
import numpy as np
import fatiando as ft

log = ft.log.get()

# Make a wave source from a mexican hat wavelet
sources = [ft.seis.wavefd.MexHatSource(0, 20, 2, 0.7, delay=1.5)]
# Set the parameters of the finite difference grid
shape = (80, 400)
spacing = (1000, 1000)
area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
# Make a density and S wave velocity model
moho_index = 30
moho = moho_index*spacing[0]
dens = np.ones(shape)
dens[:moho_index,:] *= 2700
dens[moho_index:,:] *= 3100
svel = np.ones(shape)
svel[:moho_index,:] *= 3000
svel[moho_index:,:] *= 6000

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop bellow
dt = 0.05
maxit = 4200
timesteps = ft.seis.wavefd.elastic_sh(spacing, shape, svel, dens, dt, maxit,
    sources, padding=0.8)

# This part makes an animation by updating the plot every few iterations
#ft.vis.ion()
#ft.vis.figure(figsize=(10,6))
#ft.vis.subplots_adjust(left=0.1, right=0.98)
# This part plots the wavefield (uy particle movement)
#ft.vis.subplot(2, 1, 2)
#ft.vis.axis('scaled')
#x, z = ft.grd.regular(area, shape)
#wavefield = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    #vmin=-10**(-7), vmax=10**(-7))
#rec = 300
# The location of the seismometer
#ft.vis.plot([rec*spacing[1]], [2000], '^b')
#ft.vis.hlines([moho], 0, area[1], 'k', '-')
#ft.vis.text(area[1] - 35000, moho + 10000, 'Moho')
#ft.vis.text(area[1] - 90000, 15000,
    #r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (2.7, 3))
#ft.vis.text(area[1] - 90000, area[-1] - 10000,
    #r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (3.1, 6))
#ft.vis.ylim(area[-1], area[-2])
#ft.vis.m2km()
#ft.vis.xlabel("x (km)")
#ft.vis.ylabel("z (km)")
# and this part plots a seismogram
#ft.vis.subplot(2, 1, 1)
#seismogram, = ft.vis.plot([0], [0], '-k')
#ft.vis.xlim(0, dt*maxit)
#ft.vis.ylim(-0.15, 0.15)
#ft.vis.xlabel("Time (s)")
#ft.vis.ylabel("Amplitude ($\mu$m)")
# Record the time and amplitude at the seismometer
#times = []
#addtime = times.append
#amps = []
#addamp = amps.append
# Run the iterations and update the plot
#start = time.clock()
#for i, u in enumerate(timesteps):
    # Get the amplitude at the seismometer
    #addamp(10.**(6)*u[0, rec])
    #addtime(dt*i)
    #if i%100 == 0:
        #ft.vis.title('time: %0.1f s' % (i*dt))
        #seismogram.set_ydata(amps)
        #seismogram.set_xdata(times)
        #wavefield.set_array(u[0:-1,0:-1].ravel())
        #ft.vis.draw()
#ft.vis.ioff()
#print 'Frames per second (FPS):', float(i)/(time.clock() - start)
#ft.vis.show()

# Comment the above and uncomment bellow to save snapshots of the simulation
# to png files
ft.vis.figure(figsize=(10,6))
ft.vis.subplots_adjust(left=0.1, right=0.98)
# This part plots the wavefield (uy particle movement)
ft.vis.subplot(2, 1, 2)
ft.vis.axis('scaled')
x, z = ft.grd.regular(area, shape)
wavefield = ft.vis.pcolor(x, z, np.zeros(shape).ravel(), shape,
    vmin=-10**(-7), vmax=10**(-7))
rec = 300
# The location of the seismometer
ft.vis.plot([rec*spacing[1]], [2000], '^b')
ft.vis.hlines([moho], 0, area[1], 'k', '-')
ft.vis.text(area[1] - 35000, moho + 10000, 'Moho')
ft.vis.text(area[1] - 90000, 15000,
    r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (2.7, 3))
ft.vis.text(area[1] - 90000, area[-1] - 10000,
    r'$\rho = %g g/cm^3$ $\beta = %g km/s$' % (3.1, 6))
ft.vis.ylim(area[-1], area[-2])
ft.vis.m2km()
ft.vis.xlabel("x (km)")
ft.vis.ylabel("z (km)")
# and this part plots a seismogram
ft.vis.subplot(2, 1, 1)
seismogram, = ft.vis.plot([0], [0], '-k')
ft.vis.xlim(0, dt*maxit)
ft.vis.ylim(-0.15, 0.15)
ft.vis.xlabel("Time (s)")
ft.vis.ylabel("Amplitude ($\mu$m)")
# Record the time and amplitude at the seismometer
times = []
addtime = times.append
amps = []
addamp = amps.append
# Run the iterations and update the plot
start = time.clock()
for i, u in enumerate(timesteps):
    # Get the amplitude at the seismometer
    addamp(10.**(6)*u[0, rec])
    addtime(dt*i)
    if i%10 == 0:
        ft.vis.title('time: %0.1f s' % (i*dt))
        seismogram.set_ydata(amps)
        seismogram.set_xdata(times)
        wavefield.set_array(u[0:-1,0:-1].ravel())
        ft.vis.draw()
        ft.vis.savefig('frames/f%06d.png' % ((i)/10 + 1), dpi=60)
