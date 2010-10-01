"""
Run a 1D finite differences simulation of the heat diffusion of a hotter 
intrusion.
"""

import logging
log = logging.getLogger()
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter())
log.addHandler(shandler)
log.setLevel(logging.DEBUG)

import numpy
import pylab

from fatiando.heat import diffusionfd1d


def snapshots(nodes, times, deltax, deltat, initial, diffusivity):
    """Run the simulation while saving snapshots of the temperature profile"""

    time = 0
    x = numpy.arange(0, deltax*nodes, deltax)
    
    pylab.figure()
    pylab.title("Time: %.1f" % (time))
    pylab.plot(initial, x, '-r')
    pylab.grid()
    pylab.xlim(35, 95)
    pylab.ylim(x.max(), x.min())
    pylab.xlabel("Temperatura")
    pylab.ylabel("Profundidade")
    pylab.savefig("temp%05d.png" % (0), dpi=50)
    pylab.close()
    
    next = initial
    
    log.info("Running simulation while taking snapshots... (this may take long)")
    
    for t in xrange(times):
        
        time += deltat
        
        prev = next
        
        next = diffusionfd1d.timestep(prev, deltax, deltat, diffusivity, 
                                      start_bc, end_bc)
        
        pylab.figure()
        pylab.title("Time: %.1f" % (time))
        pylab.plot(next, x, '-r')
        pylab.grid()
        pylab.xlim(35, 95)
        pylab.ylim(x.max(), x.min())
        pylab.xlabel("Temperatura")
        pylab.ylabel("Profundidade")
        pylab.savefig("temp%05d.png" % (t+1), dpi=50)
        pylab.close()

def run(nodes, times, deltax, deltat, initial, diffusivity):
    """Run the full simulation and plot the end result"""
    
    x = numpy.arange(0, deltax*nodes, deltax)
    
    temps = diffusionfd1d.run(deltax, deltat, diffusivity, initial, start_bc, 
                              end_bc, times)
    
    pylab.figure()
    pylab.title("Time: %g" % (deltat*times))
    pylab.plot(temps, x, '-r')
    pylab.grid()
    pylab.xlim(35, 95)
    pylab.ylim(x.max(), x.min())
    pylab.xlabel("Temperatura")
    pylab.ylabel("Profundidade")
    
    pylab.show()
    
    
if __name__ == '__main__':
        
    # Define the simulation parameters
    nodes = 40
    deltax = 1
    deltat = 0.4
    initial = 40*numpy.ones(nodes)
    initial[10:30] += 50
    
    # Assume a homogeneous thermal diffusivity
    diffusivity = numpy.ones(nodes)
    
    # Define the boundary conditions and free surfaces
    start_bc, end_bc = diffusionfd1d.free_bc()
    
    run(nodes, 20, deltax, deltat, initial, diffusivity)
    
#    snapshots(nodes, 450, deltax, deltat, initial, diffusivity)

    

