from fatiando.directmodels.seismo.wavefd import WaveFD1D, SinSQWaveSource

import pylab
import numpy
import time

import logging
logging.basicConfig()

period = 5*10**(-3)
source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=0)

size = 300
vel = 4000*numpy.ones(size)
vel[0:100] = 0.50*vel[0:100]
deltat = 10**(-4)
iterations = 500

solver = WaveFD1D(x1=1., x2=150., num_nodes=size, delta_t=deltat, velocities=vel, \
                  source=source, left_bc='fixed', right_bc='fixed')

solver.set_geophones([39, 79, 119, 159])

start = time.clock()

for i in xrange(iterations):
    
    solver.timestep()
    
    solver.plot(seismogram=True, tmax=iterations*deltat)
    pylab.savefig("figures/wave%04d.png" % i, dpi=50)
    pylab.close()
    
#solver.plot_seismograms()
#pylab.savefig("seismogram.png")

#solver.plot_velocity()
#pylab.savefig("velocity_structure.png")

#pylab.show()

end = time.clock()
print "Time: %g s" % (end - start)