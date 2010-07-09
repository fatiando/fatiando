from fatiando.directmodels.seismo.wavefd import WaveFD1D, SinSQWaveSource
from fatiando.data.seismo import Seismogram

import pylab
import numpy
import time

import logging
logging.basicConfig()

#data = Seismogram()
#
#data.synthetic_1d(offset=10, deltag=10, num_gp=14, xmax=150.,  \
#                  num_nodes=300, velocities=vel, deltat=deltat, tmax=0.08, \
#                  stddev=0.02, percent=True)
#
#data.plot(exaggerate=7000)

num_nodes = 300

xmax = 150.

velocities = 4000*numpy.ones(num_nodes)

velocities[0:150] = 0.50*velocities[0:150]

deltat = 10**(-4)

tmax = 0.08

period = 5*10**(-3)

source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=num_nodes)

# Extend the model region so that the seismograms won't be affected
# by the reflection in the borders. Also increase the number of nodes
# so that the resolution in the area of interest is not lost
extended = 2*xmax

velocities = numpy.append(velocities, \
                          velocities[-1]*numpy.ones(num_nodes))

velocities = numpy.append(velocities[0]*numpy.ones(num_nodes), \
                          velocities)

extended_nodes = 3*num_nodes
        
solver = WaveFD1D(x1=-xmax, x2=extended, \
                  num_nodes=extended_nodes, \
                  delta_t=deltat, velocities=velocities, \
                  source=source, left_bc='fixed', right_bc='fixed')

solver.set_geophones(offset=10, deltag=10, num=14)

start = time.clock()

i = 0

for t in numpy.arange(0, tmax, deltat):
    
    solver.timestep()
    
#    solver.plot(velocity=True, seismogram=True, tmax=tmax, exaggerate=5000)
#
#    pylab.savefig("figures/wave%04d.png" % i, dpi=200)
#    pylab.close()
    
    i += 1
    
solver.plot_seismograms(exaggerate=6000)
#pylab.savefig("seismogram.png")

solver.plot_velocity()
#pylab.savefig("velocity_structure.png")

end = time.clock()
print "Time: %g s" % (end - start)

pylab.show()