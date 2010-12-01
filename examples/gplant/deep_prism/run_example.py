"""
Example script for doing inverting synthetic FTG data GPlant
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

import fatiando.inv.gplant as gplant
import fatiando.grav.io as io
import fatiando.mesh
import fatiando.utils
import fatiando.vis as vis

# Get a logger
log = fatiando.utils.get_logger()

# Set logging to a file
fatiando.utils.set_logfile('deep_example.log')

# Log a header with the current version info
log.info(fatiando.utils.header())

# Load the synthetic data
gzz = io.load('gzz_data.txt')
#gxx = io.load('gxx_data.txt')
#gxy = io.load('gxy_data.txt')
#gxz = io.load('gxz_data.txt')
#gyy = io.load('gyy_data.txt')
#gyz = io.load('gyz_data.txt')
#gz = io.load('gz_data.txt')

data = {}
data['gzz'] = gzz
#data['gxx'] = gxx
#data['gxy'] = gxy
#data['gxz'] = gxz
#data['gyy'] = gyy
#data['gyz'] = gyz
#data['gz'] = gz

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

# Generate a model space mesh
x1, x2 = 0, 3000
y1, y2 = 0, 3000
z1, z2 = 0, 3000
mesh = fatiando.mesh.prism_mesh(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, 
                                nx=30, ny=30, nz=30)

# Set the seeds and save them for later use
log.info("Getting seeds from mesh:")
seeds = []
seeds.append(gplant.get_seed((1501, 1501, 1501), 1000, mesh))


# Make a mesh for the seeds to plot them
seed_mesh = numpy.array([seed['cell'] for seed in seeds])

# Show the seeds first to confirm that they are right
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
plot = vis.plot_prism_mesh(seed_mesh, style='surface',label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[x1, x2, y1, y2, -z2, -z1])
mlab.show()

# Run the inversion
results = gplant.grow(data, mesh, seeds, compactness=10**(4), power=3, 
                      threshold=10**(-3), norm=2, neighbor_type='reduced',
                      jacobian_file=None, distance_type='cell')

estimate, residuals, misfits, goals = results

adjusted = gplant.adjustment(data, residuals)

fatiando.mesh.fill(estimate, mesh)

log.info("Pickling results")

# Save the resulting model
output = open('result.pickle', 'w')
pickle.dump(mesh, output)
output.close()

# Pickle the seeds for later reference
seed_file = open("seeds.pickle", 'w')
pickle.dump(seeds, seed_file)
seed_file.close()

log.info("Plotting")

# Plot the residuals and goal function per iteration
pylab.figure(figsize=(8,6))
pylab.suptitle("Inversion results:", fontsize=16)
pylab.subplots_adjust(hspace=0.4)

pylab.subplot(2,1,1)
pylab.title("Residuals")
vis.residuals_histogram(residuals)
pylab.xlabel('Eotvos')

ax = pylab.subplot(2,1,2)
pylab.title("Goal function and RMS")
pylab.plot(goals, '.-b', label="Goal Function")
pylab.plot(misfits, '.-r', label="Misfit")
pylab.xlabel("Iteration")
pylab.legend(loc='upper left', prop={'size':9}, shadow=True)
ax.set_yscale('log')
ax.grid()

pylab.savefig('residuals.png')

# Get the adjustment and plot it
pylab.figure(figsize=(16,8))
pylab.suptitle("Adjustment", fontsize=14)

for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):
    
    if field in data:
        
        pylab.subplot(2, 3, i + 1)    
        pylab.title(field)    
        pylab.axis('scaled')    
        levels = vis.contour(data[field], levels=5, color='b', label='Data')
        vis.contour(adjusted[field], levels=levels, color='r', label='Adjusted')
        pylab.legend(loc='lower right', prop={'size':9}, shadow=True)

pylab.savefig("adjustment.png")

#pylab.figure()
#pylab.suptitle("Adjustment", fontsize=14)
#pylab.title("gz")
#pylab.axis('scaled')
#levels = vis.contour(data['gz'], levels=5, color='b', label='Data')
#vis.contour(adjusted['gz'], levels=levels, color='r', label='Adjusted')
#pylab.legend(loc='lower right', prop={'size':9}, shadow=True)
#pylab.savefig('adjustment-gz.png')

pylab.show()

# Plot the adjusted model plus the skeleton of the synthetic model
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
vis.plot_prism_mesh(seed_mesh, style='surface', label='Seed Density')
plot = vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[x1, x2, y1, y2, -z2, -z1])

# Plot the neighbours
#for seed in seeds:
#    
#    neighbor_mesh = []
#    
#    for neighbor in seed['neighbors']:
#        
#        neighbor_mesh.append(mesh.ravel()[neighbor])
#        
#    neighbor_mesh = numpy.array(neighbor_mesh)
#    
#    fatiando.vis.plot_prism_mesh(neighbor_mesh, style='surface', 
#                                 label='neighbors', opacity=0.0)

mlab.show()
