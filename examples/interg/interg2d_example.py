"""
Invert a gravity profile for the relief of an interface
"""
import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.data.gravity import TensorComponent
from fatiando.inversion.interg import InterG2D

data = TensorComponent('z')
data.load('gzprofile.txt')

nx = 40

solver = InterG2D(x1=0, x2=5000, nx=nx, dens=-500, \
        gz=data)

initial = 500*numpy.ones(nx)

solver.set_equality(x=2500, z=1000)

solver.solve_lm(damping=10**(-9), \
                smoothness=0*10**(-6), \
                curvature=0, \
                sharpness=3*10**(-4), beta=10**(-7), \
                equality=1, \
                initial=initial, \
                data_variance=data.cov[0][0], \
                contam_times=1, \
                lm_start=1, lm_step=10, it_p_step=10, max_it=100)

true_x, true_z = pylab.loadtxt('true_model.txt', unpack=True)

solver.plot_mean(true_x, true_z)
solver.plot_adjustment()

pylab.show()