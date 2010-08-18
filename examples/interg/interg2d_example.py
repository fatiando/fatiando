"""
Invert a gravity profile for the relief of an interface
"""
import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.interg import InterG2D

data = TensorComponent('z')
data.load('gzprofile.txt')

nx = 40

solver = InterG2D(x1=0, x2=5000, nx=nx, dens=-500, \
        gz=data)

initial = 500*numpy.ones(nx)

#solver.add_equality(x=2500, z=1000)

solver.solve(damping=1*10**(-10), \
             smoothness=0*10**(-6), \
             curvature=0*10**(-5), \
             sharpness=2*10**(-4), beta=10**(-5), \
             equality=1, \
             initial_estimate=initial, apriori_var=data.cov[0][0], \
             contam_times=1, \
             max_it=100, max_lm_it=20, lm_start=10**(-1), lm_step=10)

true_x, true_z = pylab.loadtxt('true_model.txt', unpack=True)

solver.plot_mean(true_x, true_z)
solver.plot_adjustment()

pylab.show()