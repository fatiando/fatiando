"""
Run a Finite Differences simulation of the 1D wave equation
"""
import logging
logging.basicConfig()

import pylab
import numpy
import time

from fatiando.seismo.wavefd import WaveFD1D, SinSQWaveSource

# The simulation parameters
offset = 10
deltag = 10
num_gp = 14
xmax = 150.
num_nodes = 300
deltat = 10**(-4)
tmax = 0.1
period = 5*10**(-3)
source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=num_nodes)

# Make a two velocity model
velocities = 4000*numpy.ones(num_nodes)
velocities[0:num_nodes/2] = 0.40*velocities[0:num_nodes/2]

# Run the simulation by hand to save each time step in a figure

# Extend the model region so that the seismograms won't be affected
# by the reflection in the borders. Also increase the number of nodes
# so that the resolution in the area of interest is not lost
extended = 2*xmax

ext_vels = numpy.append(velocities, velocities[-1]*numpy.ones(num_nodes))

ext_vels = numpy.append(ext_vels[0]*numpy.ones(num_nodes), ext_vels)

extended_nodes = 3*num_nodes
        
solver = WaveFD1D(x1=-xmax, x2=extended, \
                  num_nodes=extended_nodes, \
                  delta_t=deltat, velocities=ext_vels, \
                  source=source, left_bc='fixed', right_bc='fixed')

solver.set_geophones(offset, deltag, num_gp)

start = time.clock()

i = 0

for t in numpy.arange(0, tmax, deltat):
    
    solver.timestep()
    
#    solver.plot(velocity=True, seismogram=True, tmax=tmax, xmin=0, xmax=xmax, \
#                exaggerate=6000)
#
#    pylab.savefig("figures/wave%05d.png" % i, dpi=150)
#    pylab.close()
    
    i += 1
    
solver.plot_seismograms(exaggerate=6000)
pylab.xlim(0, xmax)
pylab.savefig("seismogram.png")

solver.plot_velocity()
pylab.xlim(0, xmax)
pylab.savefig("velocity_structure.png")
    
solver.plot(velocity=True, seismogram=True, tmax=tmax, xmin=0, xmax=xmax, \
            exaggerate=6000)
pylab.savefig("full_plot.png")

end = time.clock()
print "Time: %g s" % (end - start)

pylab.show()