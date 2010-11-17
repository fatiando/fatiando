"""
Example script for doing the least-squares inversion of synthetic FTG data
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.inversion import pgrav3d
from fatiando.grav import io
import fatiando.mesh
import fatiando.utils
import fatiando.vis as vis
import fatiando.stats

# Get a logger for the script
log = fatiando.utils.get_logger()

# Set logging to a file
fatiando.utils.set_logfile('run_example.log')

# Log a header with the current version info
log.info(fatiando.utils.header())

# Load the synthetic data
gzz = io.load('gzz_data.txt')

data = {'gzz':gzz}

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

# Generate a model space mesh
x1, x2 = 0, 3000
y1, y2 = 0, 3000
z1, z2 = 0, 3000
mesh = fatiando.mesh.prism_mesh(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, 
                                nx=12, ny=12, nz=12)

# Inversion parameters
damping = 0
smoothness = 10**(-4)
curvature = 0
sharpness = 0
beta = 10**(-5)
compactness = 10**(-2)
epsilon = 10**(-5)
initial = 500*numpy.ones(mesh.size)
lm_start = 10**(-5)
lm_step = 10
max_it = 500
max_steps = 20

pgrav3d.use_depth_weights(mesh, grid_height=150, normalize=True)

pgrav3d.set_bounds(vmin=0., vmax=1000.)

# Run the inversion
results = pgrav3d.solve(data, mesh, initial, damping, smoothness, curvature, 
                        sharpness, beta, compactness, epsilon, 
                        max_it, lm_start, lm_step, max_steps)

estimate, residuals, goals = results

fatiando.mesh.fill(estimate, mesh)

adjusted = pgrav3d.adjustment(data, residuals)

# Contaminate the data and re-run the inversion to test the stability
estimates = [estimate]
error = 2
contam_times = 0

log.info("Contaminating with %g Eotvos and re-running %d times" 
          % (error, contam_times))

for i in xrange(contam_times):
    
    contam = {}   
    
    for field in data.keys():
        
        contam[field] = data[field].copy()
         
        contam[field]['value'] = fatiando.utils.contaminate(
                                        data[field]['value'], stddev=error, 
                                        percent=False, return_stddev=False)
        
        results = pgrav3d.solve(data, mesh, initial, damping, smoothness, 
                                curvature, sharpness, beta, compactness, 
                                epsilon, max_it, lm_start, lm_step, max_steps)
    
    estimates.append(results[0])
    
stddev = fatiando.stats.stddev(estimates)
stddev = stddev.max()
log.info("Max stddev = %g" % (stddev))

# Make a plot of the results
pylab.figure(figsize=(14,6))
pylab.suptitle("Inversion results", fontsize=16)

# Plot the residuals
pylab.subplot(2,2,1)
pylab.title("Residuals")
fatiando.vis.residuals_histogram(residuals)

# And the goal function per iteration
pylab.subplot(2,2,3)
pylab.title("Goal function")
pylab.plot(goals, '.-k')
pylab.xlabel("Iteration")

# Plot the adjustment
pylab.subplot(1,2,2)
pylab.title("Adjustment: $g_{zz}$")
pylab.axis('scaled')
levels = vis.contour(data['gzz'], levels=5, color='b', label="Data")
vis.contour(adjusted['gzz'], levels=levels, color='r', label="Adjusted")

pylab.savefig('results.png')

pylab.show()

# Plot the 3D result model
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
plot = fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', 
                                    label='Synthetic')
fatiando.vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[x1, x2, y1, y2, -z2, -z1])

mlab.show()
