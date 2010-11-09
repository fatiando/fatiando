"""
Example script for doing the inversion of synthetic FTG data using grow
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

import fatiando.inversion.pgrav3d as pgrav3d
import fatiando.grav.io as io
import fatiando.mesh
import fatiando.utils
import fatiando.vis

# Get a logger
log = fatiando.utils.get_logger()

# Set logging to a file
fatiando.utils.set_logfile('pgrav_grow_example.log')

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
mesh = fatiando.mesh.prism_mesh(x1=0, x2=5000, y1=0, y2=5000,
                                z1=0, z2=1000, nx=50, ny=50, nz=10)

# Set the seeds and save them for later use
log.info("Getting seeds from mesh:")
seeds = []
seeds.append(pgrav3d.get_seed((901, 701, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 1201, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 1701, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 2201, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 2701, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 3201, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((901, 3701, 301), 1300, mesh))
seeds.append(pgrav3d.get_seed((3701, 1201, 901), 1000, mesh))
seeds.append(pgrav3d.get_seed((3201, 1201, 901), 1000, mesh))
seeds.append(pgrav3d.get_seed((3701, 1701, 901), 1000, mesh))
seeds.append(pgrav3d.get_seed((3201, 1701, 801), 1000, mesh))
seeds.append(pgrav3d.get_seed((2951, 3951, 301), 1200, mesh))
seeds.append(pgrav3d.get_seed((2951, 3951, 701), 1200, mesh))
seeds.append(pgrav3d.get_seed((2001, 2751, 301), 1500, mesh))
seeds.append(pgrav3d.get_seed((2501, 2751, 301), 1500, mesh))
seeds.append(pgrav3d.get_seed((3001, 2751, 301), 1500, mesh))
seeds.append(pgrav3d.get_seed((3501, 2751, 301), 1500, mesh))
seeds.append(pgrav3d.get_seed((4001, 2751, 301), 1500, mesh))


# Show the seeds before starting
seed_mesh = []
for seed in seeds:
    seed_cell = mesh.ravel()[seed['index']]
    seed_cell['value'] = seed['density']
    seed_mesh.append(seed_cell)
seed_mesh = numpy.array(seed_mesh)
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
plot = fatiando.vis.plot_prism_mesh(seed_mesh, style='surface', 
                                    label='Seed Density')
axes = mlab.axes(plot, nb_labels=9, extent=[0,5000,0,5000,-1000,0])
mlab.show()

# Inversion parameters
compactness = 10**(8)
power = 2

# Run the inversion
estimate, residuals, rmss, goals = pgrav3d.grow(data, mesh, seeds, compactness, 
                                                power=power, threshold=10**(-4), 
                                                jacobian_file='jacobian50.zip', 
                                                distance_type='radial')

adjusted = pgrav3d.adjustment(data, residuals)

fatiando.mesh.fill(estimate, mesh)

# Save the resulting model
output = open('result.pickle', 'w')
pickle.dump(mesh, output)
output.close()

# Pickle the seeds for later reference
seed_file = open("seeds.pickle", 'w')
pickle.dump(seeds, seed_file)
seed_file.close()

# Plot the results
pylab.figure(figsize=(8,6))
pylab.suptitle("Inversion results:", fontsize=16)
pylab.subplots_adjust(hspace=0.4)

# Plot the residuals
pylab.subplot(2,1,1)
pylab.title("Residuals")
fatiando.vis.residuals_histogram(residuals)
pylab.xlabel('Eotvos')

# And the goal function per iteration
ax = pylab.subplot(2,1,2)
pylab.title("Goal function and RMS")
pylab.plot(goals, '.-b', label="Goal Function")
pylab.plot(rmss, '.-r', label="RMS")
pylab.xlabel("Iteration")
pylab.legend(loc='lower left', prop={'size':10}, shadow=True)
ax.set_yscale('log')
ax.grid()
pylab.savefig('residuals.png')

# Get the adjustment and plot it
pylab.figure(figsize=(14,8))
pylab.suptitle("Adjustment")

pylab.subplot(1,2,1)
pylab.title("g_zz")
pylab.axis('scaled')
X, Y, Z = fatiando.utils.extract_matrices(data['gzz'])
ct_data = pylab.contour(X, Y, Z, 5, colors='b')
ct_data.clabel(fmt='%g')
ct_data.collections[0].set_label("Synthetic")
X, Y, Z = fatiando.utils.extract_matrices(adjusted['gzz'])
ct_adj = pylab.contour(X, Y, Z, ct_data.levels, colors='r')
ct_adj.clabel(fmt='%g')
ct_adj.collections[0].set_label("Adjusted")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())
pylab.legend(loc='lower right', prop={'size':10}, shadow=True)

#pylab.subplot(1,2,2)
#pylab.title("g_xy")
#pylab.axis('scaled')
#X, Y, Z = fatiando.utils.extract_matrices(data['gxy'])
#ct_data = pylab.contour(X, Y, Z, 5, colors='b')
#ct_data.clabel(fmt='%g')
#ct_data.collections[0].set_label("Synthetic")
#X, Y, Z = fatiando.utils.extract_matrices(adjusted['gxy'])
#ct_adj = pylab.contour(X, Y, Z, ct_data.levels, colors='r')
#ct_adj.clabel(fmt='%g')
#ct_adj.collections[0].set_label("Adjusted")
#pylab.xlim(X.min(), X.max())
#pylab.ylim(Y.min(), Y.max())
#pylab.legend(loc='lower right', prop={'size':10}, shadow=True)

pylab.savefig("adjustment.png")
pylab.show()

# Plot the adjusted model plus the skeleton of the synthetic model
fig = mlab.figure()

fig.scene.background = (0.1, 0.1, 0.1)
fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
fatiando.vis.plot_prism_mesh(seed_mesh, style='surface', label='Seed Density')
plot = fatiando.vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[0,5000,0,5000,-1000,0])

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
