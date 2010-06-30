from fatiando.directmodels.seismo.wavefd import WaveFD1D, SinSQWaveSource

import pylab
import numpy
import time

import logging
logging.basicConfig()

period = 5*10**(-3)
source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=0)

size = 200
vel = 4000*numpy.ones(size)

solver = WaveFD1D(x1=1., x2=100., num_nodes=size, delta_t=10**(-4), velocities=vel, \
                  source=source, left_bc='free', right_bc='fixed')

start = time.clock()

for i in xrange(300):
    
    solver.timestep()
    solver.plot()
    pylab.savefig("wave%d.png" % i)

end = time.clock()
print "Time: %g s" % (end - start)