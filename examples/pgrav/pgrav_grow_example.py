"""
Example script for doing the inversion of synthetic FTG data using grow
"""

import pickle
import logging
log = logging.getLogger()
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter())
log.addHandler(shandler)
fhandler = logging.FileHandler("pgrav_grow_example.log", 'w')
fhandler.setFormatter(logging.Formatter())
log.addHandler(fhandler)
log.setLevel(logging.DEBUG)

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.inversion import pgrav3d
from fatiando.gravity import io
import fatiando.geometry
import fatiando.utils
import fatiando.vis

# Load the synthetic data
gzz = io.load('gzz_data.txt')
#gxy = io.load('gxy_data.txt')

data = {}
data['gzz'] = gzz
#data['gxy'] = gxy

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

# Generate a model space mesh
mesh = fatiando.geometry.prism_mesh(x1=-800, x2=800, y1=-800, y2=800,
                                    z1=0, z2=1600, nx=8, ny=8, nz=8)

# Set the seeds and save them for later use
log.info("Getting seeds from mesh:")
seeds = []
seeds.append(pgrav3d.get_seed((10, 10, 450), 500, mesh))
seeds.append(pgrav3d.get_seed((10, 10, 1050), 1000, mesh))

# Show the seeds
#seed_mesh = []
#for seed in seeds:
#    seed_cell = mesh.ravel()[seed['param']]
#    seed_cell['value'] = seed['density']
#    seed_mesh.append(seed_cell)
#seed_mesh = numpy.array(seed_mesh)
#fig = mlab.figure()
#fig.scene.background = (0.1, 0.1, 0.1)
#fig.scene.camera.pitch(180)
#fig.scene.camera.roll(180)
#fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
#plot = fatiando.vis.plot_prism_mesh(seed_mesh, style='surface', label='Density')
#axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])
#mlab.show()

# Pickle them for later reference
seed_file = open("seeds.pickle", 'w')
pickle.dump(seeds, seed_file)
seed_file.close()

# Inversion parameters
mmi = 1*10**(-1)
power = 5
apriori_variance = 0.1**2

# Run the inversion
estimate, goals = pgrav3d.grow(data, mesh, seeds, mmi, power, apriori_variance)

pgrav3d.fill_mesh(estimate, mesh)

residuals = pgrav3d.residuals(data, estimate)

# The seeds neighbors
#neighbors = []  
#pgrav3d._add_neighbors(seeds[0]['param'], neighbors, mesh, numpy.zeros_like(estimate))
#neighbor_mesh = []
#for neighbor in neighbors:
#    neighbor_mesh.append(mesh.ravel()[neighbor])
#neighbor_mesh = numpy.array(neighbor_mesh)
#fig = mlab.figure()
#fig.scene.background = (0.1, 0.1, 0.1)
#fig.scene.camera.pitch(180)
#fig.scene.camera.roll(180)
#plot = fatiando.vis.plot_prism_mesh(neighbor_mesh, style='surface', 
#                                    label='neighbors')
#axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])
#mlab.show()

# Save the resulting model
output = open('result.pickle', 'w')
pickle.dump(mesh, output)
output.close()

# Plot the results
pylab.figure(figsize=(14,6))
pylab.suptitle("Inversion results:", fontsize=16)

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
pylab.title("Adjustment: gzz")
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

pylab.savefig("adjustment.png")
pylab.show()

# Plot the adjusted model plus the skeleton of the synthetic model
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)
fatiando.vis.plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')
plot = fatiando.vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])

for seed in seeds:
    neighbor_mesh = []
    for neighbor in seed['neighbors']:
        neighbor_mesh.append(mesh.ravel()[neighbor])
    neighbor_mesh = numpy.array(neighbor_mesh)
    fatiando.vis.plot_prism_mesh(neighbor_mesh, style='surface', 
                                 label='neighbors')


# Get the distances and make a mesh with them
distances = pgrav3d._distances
distance_mesh = fatiando.geometry.copy_mesh(mesh)
pgrav3d.fill_mesh(distances, distance_mesh)

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)
plot = fatiando.vis.plot_prism_mesh(distance_mesh, style='surface', label='Distances')
axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])

mlab.show()
