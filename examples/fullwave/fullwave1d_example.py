"""
Invert a synthetic seismogram for the 1D velocity structure along a line.
"""

import logging
logging.basicConfig()
logger = logging.getLogger("fullwave1D_example.py")
logger.setLevel(logging.DEBUG)

# Set logging to a file
baselogger = logging.getLogger()
baselogger.addHandler(logging.FileHandler("fullwave1d_example.log", 'w'))


import pylab
import numpy

from fatiando.data.seismo import Seismogram
from fatiando.inversion.fullwave import FullWave1D

# To get the simulation parameters
from make_seismograms import offset, deltag, xmax, num_nodes, source


# Load the data
data = Seismogram()

data.load('synthetic_2vels.txt', offset, deltag)

# Initiate the solver class
solver = FullWave1D(xmax=xmax,
                    nx=2, # Assume there are 2 parameters (velocities)
                    num_nodes=num_nodes, 
                    seismogram=data,
                    source=source)

initial = numpy.array([1600,4200])

solver.solve(damping=0, \
             smoothness=0, \
             curvature=0, \
             sharpness=10**(-7), \
             initial_estimate=initial, \
             apriori_var=data.cov[0][0], \
             contam_times=5, \
             max_it=500, max_lm_it=20, lm_start=10**(-3), lm_step=10)

mean_vels = solver.mean
std_vels = solver.std
logger.info("Velocities:")
logger.info("  v1 = %g +- %g" % (mean_vels[0], std_vels[0]))
logger.info("  v2 = %g +- %g" % (mean_vels[1], std_vels[1]))

solver.plot_residuals()
pylab.savefig("residuals.png")

solver.plot_adjustment(exaggerate=15000)
pylab.xlim(0, xmax)
pylab.savefig("adjustment.png")

#solver.map_goal(lower=[1000,1000], upper=[5000,5000], delta=[100,100], \
#                damping=0, \
#                smoothness=0, sharpness=10**(-10))

pylab.show()