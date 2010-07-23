import logging
logging.basicConfig()

import pylab

from fatiando.geoinv.simpletom import SimpleTom
from fatiando.data.seismo import Cart2DTravelTime

# Load the synthetic data
ttdata = Cart2DTravelTime()

ttdata.load('travel-time-data.txt')

# Make a solver and set the model space discretization
solver = SimpleTom(ttdata, x1=0, x2=30, y1=0, y2=30, nx=30, ny=30)

# Solve the linear problem with Tikhonov regularization
solver.solve_linear(damping=10**(-0), \
                    smoothness=0, \
                    curvature=0, \
                    prior_mean=None, \
                    prior_weights=None, \
                    data_variance=ttdata.cov[0][0], \
                    contam_times=20)

# Plot the results
solver.plot_mean(title='Tikhonov Result')
solver.plot_stddev(title='Tikhonov Standard Deviation', cmap=pylab.cm.jet)
solver.plot_residuals(title='Tikhonov Residuals')

pylab.show()        