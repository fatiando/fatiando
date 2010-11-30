"""
Perform a tomography on synthetic travel time data using Total Variation
regularization for sharpness.
"""

import pickle

import pylab
import numpy

from fatiando.seismo import io
from fatiando.inv import simpletom    
import fatiando.mesh                            
import fatiando.utils
import fatiando.stats
import fatiando.vis

# Make a logger for the script
log = fatiando.utils.get_logger()

log.info(fatiando.utils.header())

# Load the synthetic model for comparison
modelfile = open("model.pickle")
model = pickle.load(modelfile)
modelfile.close()

# Load the travel time data
data = io.load_traveltime('traveltimedata.txt')

error = data['error'][0]

# Make the model space mesh
model_ny, model_nx = model.shape

mesh = fatiando.mesh.square_mesh(x1=0, x2=model_nx, y1=0, y2=model_ny, 
                                     nx=model_nx, ny=model_ny)

# Inversion parameters
initial = 2.*numpy.ones(mesh.size)
damping = 1*10**(-3)
smoothness = 0
curvature = 0
sharpness = 5*10**(-1)
beta = 10**(-2)
lm_start = 100

simpletom.set_bounds(1., 5.)

# Solve
estimate, residuals, goals = simpletom.solve(data, mesh, initial, damping, 
                                             smoothness, curvature, sharpness, 
                                             beta, lm_start=lm_start)

# Put the result in the mesh (for plotting)
fatiando.mesh.fill(estimate, mesh)

# Contaminate the data with Gaussian noise and re-run the inversion to estimate
# the error 
estimates = [estimate]
contam_times = 5

log.info("Contaminating data with %g error and re-running %d times" 
         % (error, contam_times))

for i in xrange(contam_times):
    
    cont_data = data.copy()
    
    cont_data['traveltime'] = fatiando.utils.contaminate(data['traveltime'], 
                                                         stddev=error, 
                                                         percent=False, 
                                                         return_stddev=False)
    
    new_results = simpletom.solve(cont_data, mesh, initial, damping, smoothness,
                                  curvature, sharpness, beta, lm_start=lm_start)
    
    new_estimate = new_results[0]
    
    estimates.append(new_estimate)
        
# Calculate the standard deviation of the estimates
stddev_estimate = fatiando.stats.stddev(estimates)
std_mesh = fatiando.mesh.copy(mesh)
fatiando.mesh.fill(stddev_estimate, std_mesh)

# Plot the synthetic model and inversion results
pylab.figure(figsize=(12,8))
pylab.suptitle("X-ray simulation: Sharp tomography", fontsize=14)

vmin = min(estimate.min(), model.min())
vmax = max(estimate.max(), model.max())

pylab.subplot(2,2,1)
pylab.axis('scaled')
pylab.title("Synthetic velocity model")
ax = pylab.pcolor(model, cmap=pylab.cm.jet, vmin=vmin, vmax=vmax)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(2,2,2)
pylab.axis('scaled')
pylab.title("Inversion result")
fatiando.vis.plot_square_mesh(mesh, vmin=vmin, vmax=vmax)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(2,2,3)
pylab.axis('scaled')
pylab.title("Result Standard Deviation")
fatiando.vis.plot_square_mesh(std_mesh)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(2,2,4)
pylab.title("Histogram of residuals")
fatiando.vis.residuals_histogram(residuals, nbins=len(residuals)/10)
pylab.xlabel("Residual travel time")
pylab.ylabel("Residual count")

pylab.show()