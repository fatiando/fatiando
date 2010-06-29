from fatiando.directmodels.seismo.wavefd import WaveFD1D, SinSQWaveSource

import pylab
import numpy

import logging
logging.basicConfig()

source = SinSQWaveSource(amplitude=1, period=5, duration=5, index=0)

vel = numpy.ones(100)

solver = WaveFD1D(x1=1., x2=100., num_nodes=100, delta_t=0.1, velocities=vel, \
                  source=source)

for i in xrange(10):
    
    solver.timestep()
    solver.plot()
    
pylab.show()