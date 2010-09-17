"""
Example script for doing the inversion of synthetic FTG data
"""

import pickle
import logging
log = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter())
log.addHandler(handler)
log.setLevel(logging.DEBUG)

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.inversion import pgrav3d
from fatiando.gravity import io
import fatiando.geometry
import fatiando.utils
import fatiando.vis
import fatiando.stats

# Load the synthetic data
gzz = io.load('gzz_data.txt')

data = {'gzz':gzz}

# Generate a model space mesh
mesh = fatiando.geometry.prism_mesh(x1=-800, x2=800, y1=-800, y2=800,
                                    z1=0, z2=1600, nx=8, ny=8, nz=8)

# Inversion parameters
damping = 10**(-10)
smoothness = 10**(-6)
curvature = 0
sharpness = 0
beta = 10**(-5)
compactness = 0*10**(-4)
epsilon = 10**(-5)
initial = numpy.ones(mesh.size)
lm_start = 100000
lm_step = 10

pgrav3d.use_depth_weights(mesh, z0=-55.8202, power=1.49562, grid_height=150, 
                          normalize=True)

pgrav3d.set_targets(1000, 10**(8))

# Run the inversion
estimate, goals = pgrav3d.solve(data, mesh, initial, damping, smoothness,
                                curvature, sharpness, beta,  
                                compactness, epsilon, lm_start=lm_start, 
                                lm_step=lm_step)

pgrav3d.fill_mesh(estimate, mesh)

residuals = pgrav3d.residuals(data, estimate)

# Contaminate the data and re-run the inversion to test the stability
estimates = [estimate]
error = 1
contam_times = 1

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

# Plot the results
pylab.figure()
pylab.suptitle(r"Inversion results: $\sigma_{max} = %g$" % (stddev),
               fontsize=16)

pylab.subplot(2,1,1)
pylab.title("Residuals")
fatiando.vis.residuals_histogram(residuals)

pylab.subplot(2,1,2)
pylab.title("Total goal function")
pylab.plot(goals, '.-k')
pylab.xlabel("Iteration")
pylab.show()

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)

plot = fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', 
                                    label='Synthetic')

fatiando.vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])

mlab.show()
