"""
Generate the synthetic seismogram and save a figure of each time step.
Model is a 1D line.
"""
import time
import logging
logging.basicConfig()
logger = logging.getLogger("make_seismograms.py")
logger.setLevel(logging.DEBUG)

import pylab
import numpy

from fatiando.directmodels.seismo.wavefd import WaveFD1D, SinSQWaveSource
from fatiando.data.seismo import Seismogram

# The simulation parameters
offset = 30
deltag = 30
num_gp = 4
xmax = 150.
num_nodes = 300
deltat = 10**(-4)
tmax = 0.1
period = 5*10**(-3)
source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=0)
    
# Make a two velocity model
velocities = 4000*numpy.ones(num_nodes)
velocities[0:num_nodes/2] = 1500*numpy.ones_like(velocities[0:num_nodes/2])


def using_seismogram_class():
    
    # Run the simulation using the Seismogram class and save the output
    logger.info("Using the Seismogram class:")
    
    data = Seismogram()
    
    data.synthetic_1d(offset, deltag, num_gp, xmax, num_nodes, source, \
                      velocities, deltat, tmax, stddev=0.2, percent=True)
    
    data.dump('synthetic_2vels.txt')
    
    data.plot(title="Synthetic seismogram", exaggerate=15000)
    pylab.xlim(0, xmax)
    pylab.savefig("seismogram.png")
    

def by_hand():
    
    # Run the simulation by hand to save each time step in a figure
    logger.info("Running by hand and saving figures of the time steps:")
    logger.info("  WARNING: this may take a long time")
    
    # Extend the model region so that the seismograms won't be affected
    # by the reflection in the borders. Also increase the number of nodes
    # so that the resolution in the area of interest is not lost
    extended = 2*xmax
    
    ext_vels = numpy.append(velocities, velocities[-1]*numpy.ones(num_nodes))
    
    ext_vels = numpy.append(ext_vels[0]*numpy.ones(num_nodes), ext_vels)
    
    extended_nodes = 3*num_nodes
        
    new_source = source.copy()
    
    new_source.move(new_source.pos() + num_nodes)
            
    solver = WaveFD1D(x1=-xmax, x2=extended, \
                      num_nodes=extended_nodes, \
                      delta_t=deltat, velocities=ext_vels, \
                      source=new_source, left_bc='fixed', right_bc='fixed')
    
    solver.set_geophones(offset, deltag, num_gp)
    
    start = time.clock()
    
    i = 0
    
    for t in numpy.arange(0, tmax, deltat):
        
        solver.timestep()
        
#        solver.plot(velocity=True, seismogram=True, tmax=tmax, \
#                    xmin=0, xmax=xmax, exaggerate=6000)
#    
#        pylab.savefig("animation/wave%05d.png" % i, dpi=50)
#        pylab.close()
        
        # Save a snapshot of the initial thing
        if i == 0:
            
            solver.plot(velocity=True, seismogram=True, tmax=tmax, \
                        xmin=0, xmax=xmax, exaggerate=6000)
        
            pylab.savefig("initial_layout.png")
            
        i += 1
            
    solver.plot_velocity()
    pylab.xlim(0, xmax)
    pylab.savefig("velocity_structure.png")
        
    solver.plot(velocity=True, seismogram=True, tmax=tmax, xmin=0, xmax=xmax, \
                exaggerate=6000)
    pylab.savefig("final_layout.png")
    
    end = time.clock()
    print "Time: %g s" % (end - start)
    
    
if __name__ == '__main__':
    
    using_seismogram_class()
    
#    by_hand()
    
    pylab.show()
