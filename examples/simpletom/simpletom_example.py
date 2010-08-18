import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.inversion.simpletom import SimpleTom
from fatiando.data.seismo import Cart2DTravelTime

# Load the synthetic data
ttdata = Cart2DTravelTime()

ttdata.load('travel-time-data.txt')

# Make a solver and set the model space discretization
solver = SimpleTom(ttdata, x1=0, x2=30, y1=0, y2=30, nx=30, ny=30)

# Solve the linear problem with Tikhonov regularization
solver.solve_linear(damping=10**(-5), \
                    smoothness=0, \
                    curvature=10**(-1), \
                    prior_mean=None, \
                    prior_weights=None, \
                    data_variance=ttdata.cov[0][0], \
                    contam_times=20)

## Solve the linear problem with Total Variation regularization
#solver.solve_lm(damping=10**(-5), \
#                smoothness=0, \
#                curvature=0, \
#                sharpness=10**(0), beta=10**(-7), \
#                equality=0, \
#                initial=1*numpy.ones(900), \
#                prior_mean=None, prior_weights=None, \
#                data_variance=ttdata.cov[0][0], \
#                contam_times=1, \
#                lm_start=100, lm_step=10, \
#                it_p_step=20, max_it=100)

# Plot the results
solver.plot_mean(title='Mean Result', cmap=pylab.cm.jet)
solver.plot_stddev(title='Standard Deviation', cmap=pylab.cm.jet)
solver.plot_residuals(title='Residuals')

pylab.show()        