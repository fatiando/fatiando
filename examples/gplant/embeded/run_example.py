"""
Example script for doing inverting synthetic FTG data GPlant
"""
try:
	import cPickle as pickle
except:
	import pickle

import pylab
import numpy
from enthought.mayavi import mlab

import fatiando.inv.gplant as gplant
import fatiando.grav.io as io
import fatiando.grav.synthetic as synthetic
import fatiando.mesh
import fatiando.utils as utils
import fatiando.vis as vis

# Get a logger
log = utils.get_logger()
# Set logging to a file
utils.set_logfile('embeded_example.log')
# Log a header with the current version info
log.info(utils.header())

# GENERATE SYNTHETIC DATA
################################################################################
# Make the prism model
model = []
model.append({'x1':500, 'x2':2500, 'y1':500, 'y2':2500, 'z1':0, 'z2':1000,
              'value':500})
model.append({'x1':1500, 'x2':2000, 'y1':1500, 'y2':2000, 'z1':0, 'z2':500,
              'value':200})
model = numpy.array(model)

extent = [0,3000,0,3000,-2000,0]

# Show the model before calculating to make sure it's right
fig = mlab.figure()
fig.scene.background = (1, 1, 1)
plot = vis.plot_prism_mesh(model[0:1], style='surface', label='Density kg/cm^3', opacity=0.5)
plot = vis.plot_prism_mesh(model[1:], style='surface', label='Density kg/cm^3')
axes = mlab.axes(plot, nb_labels=5, extent=extent, color=(0,0,0))
axes.label_text_property.color = (0,0,0)
axes.title_text_property.color = (0,0,0)
axes.axes.label_format = "%-#.0f"
mlab.outline(color=(0,0,0), extent=extent)
mlab.show()

# Now calculate all the components of the gradient tensor
error = 1
#~ components = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
components = ['gzz']
data = {}
for i, field in enumerate(components):
    data[field] = synthetic.from_prisms(model, x1=0, x2=3000, y1=0, y2=3000,
                                        nx=15, ny=15, height=100, field=field)
    data[field]['value'], error = utils.contaminate(data[field]['value'],
                                                    stddev=error,
                                                    percent=False,
                                                    return_stddev=True)
    data[field]['error'] = error*numpy.ones(len(data[field]['value']))

# Plot the data
pylab.figure(figsize=(16,8))
pylab.suptitle(r'Synthetic FTG data with %g $E\"otv\"os$ noise' % (error))
for i, field in enumerate(components):
    #~ pylab.subplot(2, 3, i + 1)
    pylab.axis('scaled')
    pylab.title(field)
    vis.contourf(data[field], 10)
    cb = pylab.colorbar()
    cb.set_label(r'$E\"otv\"os$')
    pylab.xlim(data[field]['x'].min(), data[field]['x'].max())
    pylab.ylim(data[field]['y'].min(), data[field]['y'].max())
pylab.savefig("data.png")

# RUN THE INVERSION
################################################################################
# Generate a model space mesh
x1, x2, y1, y2, z2, z1 = extent
z2, z1 = -1*z2, -1*z1
mesh = fatiando.mesh.prism_mesh(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, 
                                nx=30, ny=30, nz=20)

# Set the seeds and save them for later use
log.info("Getting seeds from mesh:")
seeds = []
seeds.append(gplant.get_seed((1500, 1500, 500), 500, mesh))

# Make a mesh for the seeds to plot them
seed_mesh = numpy.array([seed['cell'] for seed in seeds])

# Show the seeds first to confirm that they are right
fig = mlab.figure()
fig.scene.background = (1, 1, 1)
vis.plot_prism_mesh(model, style='wireframe', label='Synthetic')
plot = vis.plot_prism_mesh(seed_mesh, style='surface',label='Density')
axes = mlab.axes(plot, nb_labels=5, extent=extent, color=(0,0,0))
axes.label_text_property.color = (0,0,0)
axes.title_text_property.color = (0,0,0)
axes.axes.label_format = "%-#.0f"
mlab.outline(color=(0,0,0), extent=extent)
mlab.show()

# Run the inversion
results = gplant.grow(data, mesh, seeds, compactness=10.**(5), power=5,
                      threshold=1*10**(-4), norm=1, neighbor_type='reduced',
                      jacobian_file=None, distance_type='radial')

# Unpack the results and calculate the adjustment
estimate, residuals, misfits, goals = results
adjusted = gplant.adjustment(data, residuals)
fatiando.mesh.fill(estimate, mesh)

# PLOT THE RESULTS
################################################################################
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

# Plot the ajustment
pylab.figure(figsize=(16,8))
pylab.suptitle("Adjustment", fontsize=14)
for i, field in enumerate(components):
    #~ pylab.subplot(2, 3, i + 1)
    pylab.title(field)
    pylab.axis('scaled')
    levels = vis.contour(data[field], levels=12, color='k', label='Data')
    vis.contour(adjusted[field], levels=levels, color='r', label='Adjusted')
    pylab.legend(loc='lower right', prop={'size':9}, shadow=True)
pylab.savefig("adjustment.png")

pylab.show()

# Plot the adjusted model plus the skeleton of the synthetic model
fig = mlab.figure()
fig.scene.background = (1, 1, 1)
vis.plot_prism_mesh(model, style='wireframe', label='Synthetic')
vis.plot_prism_mesh(seed_mesh, style='surface', label='Seed Density')
plot = vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=5, extent=extent, color=(0,0,0))
axes.label_text_property.color = (0,0,0)
axes.title_text_property.color = (0,0,0)
axes.axes.label_format = "%-#.0f"
mlab.outline(color=(0,0,0), extent=extent)

mlab.show()
