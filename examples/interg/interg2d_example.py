"""
Invert a gravity profile for the relief of an interface.
"""

import pickle
import logging
log = logging.getLogger()
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter())
log.addHandler(shandler)
fhandler = logging.FileHandler("interg2d_example.log", 'w')
fhandler.setFormatter(logging.Formatter())
log.addHandler(fhandler)
log.setLevel(logging.DEBUG)

import numpy
import pylab

from fatiando.inversion import interg2d
from fatiando.gravity import io
import fatiando.geometry
import fatiando.utils
import fatiando.vis


# Load up the gravity data and the synthetic model
data = {}
data['gz'] = io.load('gzprofile.txt')

modelfile = open('model.pickle')
synthetic = pickle.load(modelfile)
modelfile.close()

# Make a model space mesh and solve
mesh = fatiando.geometry.line_mesh(0, 5000, 100)

density = -500.
ref_surf = numpy.zeros(mesh.size)
initial = 1000*numpy.ones(mesh.size)
damping = 0*10**(-15)
smoothness = 1*10**(-15)
curvature = 0
sharpness = 0
beta = 10**(-5)
lm_start = 100

interg2d.set_bounds(0, 3000)

estimate, goals = interg2d.solve(data, mesh, density, ref_surf, initial, 
                                 damping, smoothness, curvature, sharpness, 
                                 beta, lm_start=lm_start)

interg2d.fill_mesh(estimate, mesh)

resultfile = open("result.pickle", 'w')
pickle.dump(mesh, resultfile)
resultfile.close()

residuals = interg2d.residuals(estimate)
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

    new_estimate, new_goals = interg2d.solve(contam, mesh, density, ref_surf, 
                                             initial, damping, smoothness, 
                                             curvature, sharpness, beta, 
                                             lm_start=lm_start)
    
    estimates.append(new_estimate)

resultfile = open("estimates.pickle", 'w')
pickle.dump(estimates, resultfile)
resultfile.close()
    
# Plot the results
pylab.figure(figsize=(14,8))

pylab.subplot(2,2,1)
pylab.title("Adjustment")
pylab.plot(data['gz']['x'], data['gz']['value'], '.k', label="Data")
pylab.plot(adjusted['gz']['x'], adjusted['gz']['value'], '-r', 
           label="Adjusted")
pylab.ylabel("mGal")

pylab.subplot(2,2,2)
pylab.title("Residuals")
fatiando.vis.residuals_histogram(residuals)

pylab.subplot(2,2,3)
pylab.title("Inversion result")
fatiando.vis.plot_2d_interface(mesh, style='-k', linewidth=1, label="Inverted")
fatiando.vis.plot_2d_interface(synthetic, key='z2', style='-r', linewidth=1,  
                               label='Synthetic')
for new_estimate in estimates:
    new_mesh = fatiando.geometry.copy_mesh(mesh)
    interg2d.fill_mesh(new_estimate, new_mesh)
    fatiando.vis.plot_2d_interface(new_mesh, style='-b', linewidth=1)   
pylab.legend(loc='lower right')
pylab.ylim(2500., -200)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")

pylab.subplot(2,2,4)
pylab.title("Goal function")
pylab.plot(goals, '-k')
pylab.xlabel("Iteration")

pylab.show()