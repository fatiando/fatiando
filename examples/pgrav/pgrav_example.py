"""
Example script for doing the inversion of synthetic FTG data
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.inversion import pgrav3d
from fatiando.grav import io
import fatiando.mesh
import fatiando.utils
import fatiando.vis
import fatiando.stats

# Get a logger for the script
log = fatiando.utils.get_logger()

# Load the synthetic data
gzz = io.load('gzz_data.txt')

data = {'gzz':gzz}

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

# Generate a model space mesh
mesh = fatiando.mesh.prism_mesh(x1=-800, x2=800, y1=-800, y2=800,
                                    z1=0, z2=1600, nx=8, ny=8, nz=8)

# Inversion parameters
damping = 0
smoothness = 10**(-6)
curvature = 0
sharpness = 0
beta = 10**(-5)
compactness = 10**(-6)
epsilon = 10**(-5)
initial = 500*numpy.ones(mesh.size)
lm_start = 1
lm_step = 2

pgrav3d.use_depth_weights(mesh, grid_height=150, normalize=True)

# Run the inversion
estimate, goals = pgrav3d.solve(data, mesh, initial, damping, smoothness,
                                curvature, sharpness, beta,  
                                compactness, epsilon, lm_start=lm_start, 
                                lm_step=lm_step)

fatiando.mesh.fill(estimate, mesh)

residuals = pgrav3d.residuals(data, estimate)

# Contaminate the data and re-run the inversion to test the stability
estimates = [estimate]
error = 0.1
contam_times = 2

log.info("Contaminating with %g Eotvos and re-running %d times" 
          % (error, contam_times))

for i in xrange(contam_times):
    
    contam = {}   
    
    for field in data.keys():
        
        contam[field] = data[field].copy()
         
        contam[field]['value'] = fatiando.utils.contaminate(
                                        data[field]['value'], stddev=error, 
                                        percent=False, return_stddev=False)
    
    new_estimate, new_goal = pgrav3d.solve(contam, mesh, initial, damping,
                                           smoothness, curvature, sharpness, 
                                           beta, compactness, epsilon,
                                           lm_start=lm_start, lm_step=lm_step)
    
    estimates.append(new_estimate)
    
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

# Get the adjustment and plot it
pylab.subplot(1,2,2)
pylab.title("Adjustment: $g_{zz}$")
pylab.axis('scaled')
adjusted = pgrav3d.calc_adjustment(estimate, grid=True)
X, Y, Z = fatiando.utils.extract_matrices(data['gzz'])
ct_data = pylab.contour(X, Y, Z, 5, colors='b')
ct_data.clabel(fmt='%g')
X, Y, Z = fatiando.utils.extract_matrices(adjusted['gzz'])
ct_adj = pylab.contour(X, Y, Z, ct_data.levels, colors='r')
ct_adj.clabel(fmt='%g')
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

pylab.show()

# Plot the 3D result model
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)

plot = fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', 
                                    label='Synthetic')

fatiando.vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])

mlab.show()
