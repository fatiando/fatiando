import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.directmodels.seismo.wavefd import SinSQWaveSource
from fatiando.data.seismo import Seismogram
from fatiando.geoinv.fullwave import FullWave1D


period = 5*10**(-3)

source = SinSQWaveSource(amplitude=-0.001, period=period, \
                         duration=period, offset=period, index=0)

nodes = 300

velocities = 4000*numpy.ones(nodes)

velocities[0:150] = 0.50*velocities[0:150]

data = Seismogram()

data.synthetic_1d(offset=10, deltag=30, num_gp=5, xmax=150., source=source, \
                  num_nodes=nodes, velocities=velocities, deltat=10**(-4), \
                  tmax=0.08, stddev=0.05, percent=True)

#data.plot(exaggerate=10000)

#pylab.show()

solver = FullWave1D(xmax=150., nx=2, num_nodes=nodes, seismogram=data, \
                    source=source)

#solver.map_goal(lower=[0,0], upper=[5000,5000], delta=[1000,1000], damping=10**(-11), \
#                smoothness=0)

initial = numpy.array([2500,4200])

solver.solve(damping=0, smoothness=10**(-17), curvature=0, sharpness=0, \
             equality=0, initial_estimate=initial, apriori_var=data.cov[0][0], \
             contam_times=0, \
             max_it=100, max_lm_it=20, lm_start=10**(-2), lm_step=10)

print "Velocities: ", solver.mean

solver.plot_residuals()

solver.plot_adjustment(exaggerate=10000)

pylab.show()