"""
Example script for doing inverting synthetic FTG data GPlant
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

import fatiando.inv.gplant as gplant
from fatiando.grav import synthetic
import fatiando.mesh
from fatiando import utils, vis, geometry

# Get a logger
log = utils.get_logger()
# Set logging to a file
utils.set_logfile('complex_example.log')
# Log a header with the current version info
log.info(utils.header())

# GENERATE SYNTHETIC DATA
################################################################################
# Make the prism model
model = []
model.append(geometry.prism(x1=600, x2=1200, y1=200, y2=4200, z1=100, z2=600,
                            props={'value':1300}))
model.append(geometry.prism(x1=3000, x2=4000, y1=1000, y2=2000, z1=200, z2=800,
                            props={'value':1000}))
model.append(geometry.prism(x1=2700, x2=3200, y1=3700, y2=4200, z1=0, z2=900,
                            props={'value':1200}))
model.append(geometry.prism(x1=1500, x2=4500, y1=2500, y2=3000, z1=100, z2=500,
                            props={'value':1500}))
model = numpy.array(model)

x1, x2 = 0, 5000
y1, y2 = 0, 5000
z1, z2 = 0, 1000
extent = [x1, x2, y1, y2, -z2, -z1]

# Show the model before calculating to make sure it's right
fig = mlab.figure()
fig.scene.background = (1, 1, 1)
plot = vis.plot_prism_mesh(model, style='surface', label='Density kg/cm^3')
axes = mlab.axes(plot, nb_labels=5, extent=extent, color=(0,0,0))
axes.label_text_property.color = (0,0,0)
axes.title_text_property.color = (0,0,0)
axes.axes.label_format = "%-#.0f"
mlab.outline(color=(0,0,0), extent=extent)
mlab.show()

# Now calculate all the components of the gradient tensor
error = 1
data = {}
for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):
    data[field] = synthetic.from_prisms(model, x1=0, x2=5000, y1=0, y2=5000,
                                        nx=30, ny=30, height=150, field=field)
    data[field]['value'], error = utils.contaminate(data[field]['value'],
                                                    stddev=error,
                                                    percent=False,
                                                    return_stddev=True)
    data[field]['error'] = error*numpy.ones(len(data[field]['value']))

# Plot the synthetic data
pylab.figure(figsize=(16,8))
pylab.suptitle(r'Synthetic FTG data with %g $E\"otv\"os$ noise' % (error))
for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):
    pylab.subplot(2, 3, i + 1)
    pylab.axis('scaled')
    pylab.title(field)
    vis.contourf(data[field], 10)
    cb = pylab.colorbar()
    cb.set_label(r'$E\"otv\"os$')
pylab.savefig("ftg_data.png")

# RUN THE INVERSION
################################################################################
# Generate a model space mesh
mesh = fatiando.mesh.prism_mesh(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2, 
                                nx=50, ny=50, nz=10)

# Set the seeds and save them for later use
log.info("Getting seeds:")
seedpoints = []
#seedpoints.append(((901, 401, 301), 1300))
#seedpoints.append(((901, 601, 301), 1300))
#seedpoints.append(((901, 801, 301), 1300))
#seedpoints.append(((901, 1001, 301), 1300))
#seedpoints.append(((901, 1201, 301), 1300))
#seedpoints.append(((901, 1401, 301), 1300))
#seedpoints.append(((901, 1601, 301), 1300))
#seedpoints.append(((901, 1801, 301), 1300))
#seedpoints.append(((901, 2001, 301), 1300))
#seedpoints.append(((901, 2201, 301), 1300))
#seedpoints.append(((901, 2401, 301), 1300))
#seedpoints.append(((901, 2601, 301), 1300))
#seedpoints.append(((901, 2801, 301), 1300))
#seedpoints.append(((901, 3001, 301), 1300))
#seedpoints.append(((901, 3201, 301), 1300))
#seedpoints.append(((901, 3401, 301), 1300))
#seedpoints.append(((901, 3601, 301), 1300))
#seedpoints.append(((901, 3801, 301), 1300))
#seedpoints.append(((901, 4001, 301), 1300))
seedpoints.append(((3701, 1201, 501), 1000))
seedpoints.append(((3201, 1201, 501), 1000))
seedpoints.append(((3701, 1701, 501), 1000))
seedpoints.append(((3201, 1701, 501), 1000))
#seedpoints.append(((2951, 3951, 301), 1200))
#seedpoints.append(((2951, 3951, 701), 1200))
#seedpoints.append(((1701, 2751, 301), 1500))
#seedpoints.append(((1901, 2751, 301), 1500))
#seedpoints.append(((2101, 2751, 301), 1500))
#seedpoints.append(((2301, 2751, 301), 1500))
#seedpoints.append(((2501, 2751, 301), 1500))
#seedpoints.append(((2701, 2751, 301), 1500))
#seedpoints.append(((2901, 2751, 301), 1500))
#seedpoints.append(((3101, 2751, 301), 1500))
#seedpoints.append(((3301, 2751, 301), 1500))
#seedpoints.append(((3501, 2751, 301), 1500))
#seedpoints.append(((3701, 2751, 301), 1500))
#seedpoints.append(((3901, 2751, 301), 1500))
#seedpoints.append(((4101, 2751, 301), 1500))
#seedpoints.append(((4301, 2751, 301), 1500))
seedpoints.append(((901, 701, 301), 1300))
seedpoints.append(((901, 1201, 301), 1300))
seedpoints.append(((901, 1701, 301), 1300))
seedpoints.append(((901, 2201, 301), 1300))
seedpoints.append(((901, 2701, 301), 1300))
seedpoints.append(((901, 3201, 301), 1300))
seedpoints.append(((901, 3701, 301), 1300))
#seedpoints.append(((3601, 1301, 501), 1000))
#seedpoints.append(((3301, 1301, 501), 1000))
#seedpoints.append(((3601, 1601, 501), 1000))
#seedpoints.append(((3301, 1601, 501), 1000))
seedpoints.append(((2951, 3951, 301), 1200))
seedpoints.append(((2951, 3951, 701), 1200))

seedpoints.append(((2001, 2751, 301), 1500))
seedpoints.append(((2501, 2751, 301), 1500))
seedpoints.append(((3001, 2751, 301), 1500))
seedpoints.append(((3501, 2751, 301), 1500))
seedpoints.append(((4001, 2751, 301), 1500))

seeds = [gplant.get_seed(point, dens, mesh) for point, dens in seedpoints]

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
results = gplant.grow(data, mesh, seeds, compactness=10**(15), power=5,
                      threshold=5*10**(-4), norm=2, neighbor_type='reduced',
                      jacobian_file=None, distance_type='radial')

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
pylab.title("Goal function and Data misfit")
pylab.plot(goals, '.-b', label="Goal Function")
pylab.plot(misfits, '.-r', label="Misfit")
pylab.xlabel("Iteration")
pylab.legend(loc='upper left', prop={'size':9}, shadow=True)
ax.set_yscale('log')
ax.grid()
pylab.savefig('residuals.pdf')

# Get the adjustment and plot it
pylab.figure(figsize=(16,8))
pylab.suptitle(r'Adjustment [$E\"otv\"os$]', fontsize=14)
for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):
    if field in data:
        pylab.subplot(2, 3, i + 1)    
        pylab.title(field)    
        pylab.axis('scaled')    
        levels = vis.contour(data[field], levels=5, color='b', label='Data')
        vis.contour(adjusted[field], levels=levels, color='r', label='Adjusted')
        pylab.legend(loc='lower left', prop={'size':9}, shadow=True)
pylab.savefig("adjustment.pdf")

pylab.show()

# Plot the adjusted model plus the skeleton of the synthetic model
fig = mlab.figure()
fig.scene.background = (1, 1, 1)
vis.plot_prism_mesh(model, style='wireframe', label='Synthetic')
vis.plot_prism_mesh(seed_mesh, style='surface', label='Seeds')
plot = vis.plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=5, extent=extent, color=(0,0,0))
axes.label_text_property.color = (0,0,0)
axes.title_text_property.color = (0,0,0)
axes.axes.label_format = "%-#.0f"
mlab.outline(color=(0,0,0), extent=extent)
mlab.show()