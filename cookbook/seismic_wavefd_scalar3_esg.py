"""
Seismic: 3D finite difference simulation of Equivalent Staggered Grid (ESG)
acoustic wave equation scheme of Di Bartolo et al. (2012).
North and Down velocity gradient
"""
import numpy as np
from fatiando.seismic import wavefd

# Set the parameters of the finite difference grid 3D
shape = (100, 100, 100)
ds = 10.  # spacing
area = [0, shape[0]*ds, 0, shape[1]*ds, 0, shape[2]*ds]
# Set the parameters of the finite difference grid
velocity = np.ones(shape)*1500.  # m/s
density = np.ones(shape)*1000.  # kg/m^3
for i in xrange(100):  # density/velocity changing
    velocity[:, :, i] += i*20  # m/s
    # increasing with depth
    velocity[i, :, :] += i*20  # m/s
# avoiding spatial alias, frequency of source should be smaller than this
fc = 0.5*np.min(velocity)/ds  # based on plane waves v=l*f
fc -= 0.5*fc
sources = [wavefd.GaussSource((50*ds, 50*ds, 40*ds), area, shape,  10**(-8), fc)]
dt = wavefd.maxdt(area, shape, np.max(velocity))
duration = 0.35
maxit = int(duration/dt)
# x, y, z coordinate of the seismometer
stations = [[45*ds, 45*ds, 65*ds], [65*ds, 65*ds, 30*ds]]
snapshots = 5  # every 1 iterations plots one
simulation = wavefd.scalar3_esg(velocity, density, area, dt, maxit,
                                sources, stations, snapshots)
# 3d contour iso surfaces
from mayavi import mlab
fig = mlab.figure(size=(600,600))
t, u, seismogram = simulation.next()
min = u.min(); max = u.max()
min = min+0.65*(max-min); max = min+0.9*(max-min)
u = u.transpose()[::-1]
sscalar = mlab.contour3d(u, contours=[min, 0.8*max])
extent = [0, shape[0], 0, shape[1], 0, shape[2]]
mlab.axes(extent=extent)
mlab.outline(extent=extent)
azimuth = 27.087178769208965
elevation = -120.9368000828039
distance = 334.60652149512919
focalpoint = [53.01525703,  57.20435378,  61.16758842]
mlab.view(azimuth, elevation, distance, focalpoint, reset_roll=True)
import sys
for t, u, seismogram in simulation:
    min = u.min(); max = u.max()
    min = min+0.65*(max-min); max = min+0.9*(max-min)
    u = u.transpose()[::-1] # solving z up
    sscalar.mlab_source.set(scalars=u, contours=[min, 0.8*max])
    sys.stdout.write("\rprogressing .. %.1f%% time %.3f"%(100.0*float(t)/maxit, (dt*t)))
    sys.stdout.flush()
mlab.show()  # make possible to manipulate at the end

# mlab.pipeline.surface
# with enable contours = True; would be perfect
# or mlab.pipeline.volume