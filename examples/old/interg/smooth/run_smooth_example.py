"""
Invert a gravity profile for the relief of a basin using Smoothness.
"""

import pickle

import numpy
import pylab

from fatiando.inv import interg2d
from fatiando.grav import io
import fatiando.mesh
import fatiando.utils
import fatiando.vis


# Get a logger for the script
log = fatiando.utils.get_logger()

# Set logging to a file
fatiando.utils.set_logfile("smooth_example.log")

log.info(fatiando.utils.header())

# Load up the gravity data and the synthetic model
data = {}
data['gz'] = io.load('gzprofile.txt')

modelfile = open('model.pickle')
synthetic = pickle.load(modelfile)
modelfile.close()

# Make a model space mesh
mesh = fatiando.mesh.line_mesh(0, 5000, 50)

# Define the inversion parameters
density = -200.
ref_surf = numpy.zeros(mesh.size)
initial = 1000*numpy.ones(mesh.size)

# Regularization parameters
damping = 0
smoothness = 1*10**(-6)
curvature = 0
sharpness = 0
beta = 0
lm_start = 0.00001

# Solve
estimate, residuals, goals = interg2d.solve(data, mesh, density, ref_surf, 
                                            initial, damping, smoothness, 
                                            curvature, sharpness, beta, 
                                            lm_start=lm_start)

# Fill in the mesh with the inversion results
fatiando.mesh.fill(estimate, mesh, key='z2')
fatiando.mesh.fill(ref_surf, mesh, key='z1')

# Pickle the result to use later
resultfile = open("result-smooth.pickle", 'w')
pickle.dump(mesh, resultfile)
resultfile.close()

# Compute the adjusted data
adjusted = interg2d.adjustment(estimate, profile=True)

# Contaminate the data with more noise and re-run to test the stability of the
# solution
estimates = []
contam_times = 3
error = data['gz']['error'][0]

log.info("Contaminating data %d times with %g mGal noise" 
         % (contam_times, error))

for i in xrange(contam_times):
    
    contam = {}
    contam['gz'] = data['gz'].copy()
    
    contam['gz']['value'] = fatiando.utils.contaminate(data['gz']['value'],
                                                       stddev=error, 
                                                       percent=False)

    new_results = interg2d.solve(contam, mesh, density, ref_surf, initial, 
                                 damping, smoothness, curvature, sharpness, 
                                 beta, lm_start=lm_start)
    
    new_estimate = new_results[0]
    
    estimates.append(new_estimate)

# Pickle the estimates to save them for later reference
resultfile = open("estimates-smooth.pickle", 'w')
pickle.dump(estimates, resultfile)
resultfile.close()
    
# Plot the results
pylab.figure(figsize=(14,8))
pylab.subplots_adjust(hspace=0.3)
pylab.suptitle("Smoothness Inversion", fontsize=14)

# Adjustment X Synthetic data
pylab.subplot(2,2,1)
pylab.title("Adjustment")
pylab.plot(data['gz']['x'], data['gz']['value'], '.k', label="Synthetic")
pylab.plot(adjusted['gz']['x'], adjusted['gz']['value'], '-r', 
           label="Adjusted")
pylab.ylabel("mGal")
pylab.legend(loc='upper left', prop={'size':10}, shadow=True)

# Histogram of residuals
pylab.subplot(2,2,2)
pylab.title("Residuals")
fatiando.vis.residuals_histogram(residuals)
pylab.xlabel("mGal")
pylab.ylabel("Number of occurrences")

pylab.subplot(2,2,3)
pylab.title("Inversion result")
# Synthetic model
fatiando.vis.plot_2d_interface(synthetic, key='z2', style='-r', linewidth=1,  
                               label='Synthetic', fill=synthetic, fillkey='z1',
                               fillcolor='r', alpha=0.5)
# Initial estimate
initial_mesh = fatiando.mesh.copy(mesh)
fatiando.mesh.fill(initial, initial_mesh)
fatiando.vis.plot_2d_interface(initial_mesh, style='-.g', label='Initial')

# Also plot the stability estimates
for new_estimate in estimates:    
    new_mesh = fatiando.mesh.copy(mesh)
    fatiando.mesh.fill(new_estimate, new_mesh, key='z2')
    plot = fatiando.vis.plot_2d_interface(new_mesh, key='z2', style='-b', 
                                          linewidth=1)
plot.set_label("Dispersion")   

# Inversion result
fatiando.vis.plot_2d_interface(mesh, key='z2', style='-k', linewidth=1.5, 
                               label="Inverted")
 
pylab.legend(loc='lower right', prop={'size':10}, shadow=True)
pylab.ylim(2500., -200)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")

# Goal Function X Iteration
pylab.subplot(2,2,4)
pylab.title("Goal function")
pylab.plot(goals, '.-k')
pylab.xlabel("Iteration")

pylab.show()