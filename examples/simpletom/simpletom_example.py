"""
Perform a tomography on synthetic travel time data.
"""

import logging
logging.basicConfig()
log = logging.getLogger('script')
log.setLevel(logging.DEBUG)

import pylab
import numpy

from fatiando.seismo import synthetic, io
from fatiando.inversion import simpletom                                
import fatiando.utils
import fatiando.stats
from fatiando.visualization import plot_src_rec, \
                                   plot_ray_coverage, \
                                   residuals_histogram, \
                                   plot_square_mesh   

# SYNTHETIC DATA GENERATION FROM AN IMAGE FILE
model = synthetic.vel_from_image('square-model.jpg', vmax=5, vmin=1)

data = synthetic.shoot_cartesian_straight(model, src_n=20, rec_n=10, type='xray')

# Contaminate the data with Gaussian noise
data['traveltime'], error = fatiando.utils.contaminate(data['traveltime'], 
                                                       stddev=0.01, 
                                                       percent=True, 
                                                       return_stddev=True)

log.info("Contaminated the data with %g noise" % (error))

data['error'] = error*numpy.ones(len(data['traveltime']))

io.dump_traveltime('traveltimes.txt', data)

# PERFORM THE INVERSION
model_ny, model_nx = model.shape

mesh = simpletom.make_mesh(x1=0, x2=model_nx, y1=0, y2=model_ny, nx=5, 
                           ny=5)

initial = numpy.ones(mesh.size)
damping = 10**(-5)
smoothness=10**(-2)
curvature=0
sharpness=0
beta=10**(-5)

estimate, goals = simpletom.solve(data, mesh, initial, damping, smoothness, 
                                  curvature, sharpness, beta, 
                                  lm_start=1)

simpletom.fill_mesh(estimate, mesh)

residuals = simpletom.residuals(data, estimate)

# Contaminate the data with Gaussian noise and re-run the inversion to estimate
# the error 
estimates = [estimate]

for i in xrange(2):
    
    cont_data = data.copy()
    
    cont_data['traveltime'] = fatiando.utils.contaminate(data['traveltime'], 
                                                         stddev=error, 
                                                         percent=False, 
                                                         return_stddev=False)
    
    new_estimate, new_goal = simpletom.solve(cont_data, mesh, initial, damping, 
                                             smoothness, curvature, sharpness, 
                                             beta, lm_start=1)
    
    estimates.append(new_estimate)
        
stddev_estimate = fatiando.stats.stddev(estimates)
std_mesh = simpletom.copy_mesh(mesh)
simpletom.fill_mesh(stddev_estimate, std_mesh)

pylab.figure()
pylab.axis('scaled')
pylab.title("Stddev")
plot_square_mesh(std_mesh)
pylab.colorbar()

pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

# PLOT THE SYNTHETIC DATA AND INVERSION RESULTS
pylab.figure(figsize=(12,10))

pylab.subplot(2,2,1)
pylab.axis('scaled')
pylab.title("Velocity model")

ax = pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")

pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(2,2,2)
pylab.axis('scaled')
pylab.title("Ray coverage")

pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")

plot_ray_coverage(data['src'], data['rec'], '-k')

plot_src_rec(data['src'], data['rec'])

pylab.xlim(-35, model.shape[1] + 35)
pylab.ylim(-35, model.shape[0] + 35)

pylab.subplot(2,2,3)
pylab.axis('scaled')
pylab.title("Mean inversion result")

plot_square_mesh(mesh)
#plot_square_mesh(mesh, vmin=1, vmax=5)
cb = pylab.colorbar()
cb.set_label("Velocity")

pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(2,2,4)
pylab.title("Histogram of residuals")

residuals_histogram(residuals)

pylab.xlabel("Residual travel time")
pylab.ylabel("Residual count")
pylab.show()