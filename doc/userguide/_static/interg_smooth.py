# Generate some synthetic gravity data and invert for the relief of a basin.

import numpy
import pylab

from fatiando.inv import interg2d
from fatiando import utils, stats, vis
import fatiando.mesh
import fatiando.grav.synthetic

# Get a logger for the script (also enables stderr logging)
log = utils.get_logger()

# GENERATE SYNTHETIC DATA
################################################################################
# Make a horizontal line of prisms to simulate the basin
synthetic = fatiando.mesh.line_mesh(0, 5000, 100)

dx = synthetic[0]['x2'] - synthetic[0]['x1']

# Exaggerate the y axis of a 3D prism to make it 2D
y1 = -1000.*dx
y2 = 1000.*dx

# The density contrast of the sediment
density = -500.

# The maximum depth of the basin bottom
amp = 2000.

# Calculate the bottom of the prisms using 2 Gaussian functions
for i, cell in enumerate(synthetic.ravel()):

    x = 0.5*(cell['x1'] + cell['x2'])
    cell['value'] = density
    cell['y1'] = y1
    cell['y2'] = y2
    cell['z1'] = 0
    cell['z2'] = amp*(stats.gaussian(x, 1500., 1000.) +
                      0.4*stats.gaussian(x, 3500., 1000.))

# Calculate the gravitational effect of the synthetic model along a profile
gz = fatiando.grav.synthetic.from_prisms(synthetic, x1=0, x2=5000, y1=0, y2=0,
                                         nx=200, ny=1, height=-1, field='gz')

# Contaminate it with gaussian noise and save
error = 0.5
gz['value'] = fatiando.utils.contaminate(gz['value'], stddev=error,
                                                percent=False)
gz['error'] = error*numpy.ones_like(gz['value'])

# INVERSION
################################################################################

# Inform the inversion that the data is vertical gravity, not gradients
data = {'gz':gz}

# Make a model space mesh
mesh = fatiando.mesh.line_mesh(0, 5000, 50)

# Define the inversion parameters
density = -500.
ref_surf = numpy.zeros(mesh.size) # ie, the topography
initial = 1000*numpy.ones(mesh.size) # The initial estimate

# Solve
results = interg2d.solve(data, mesh, density, ref_surf, initial,
                         smoothness=10**(-5), lm_start=0.00001)

# Unpack the results
estimate, residuals, goals = results

# Fill in the mesh with the inversion results
fatiando.mesh.fill(estimate, mesh, key='z2')
fatiando.mesh.fill(ref_surf, mesh, key='z1')

# Compute the adjusted data
adjusted = interg2d.adjustment(estimate, profile=True)

# PLOTING
################################################################################

# Plot the model and gravity effect
pylab.figure(figsize=(14,12))
pylab.suptitle("Smooth Inversion of Synthetic Gravity Data", fontsize=18)
pylab.subplots_adjust(hspace=0.3)

pylab.subplot(3,2,1)
pylab.title("Synthetic noise-corrupted data")
pylab.plot(gz['x'], gz['value'], '.-k', label=r"Synthetic $g_z$")
pylab.ylabel("mGal")
pylab.legend(loc='lower right', shadow=True)

pylab.subplot(3,2,2)
pylab.title("Synthetic basin model")
vis.plot_2d_interface(synthetic, 'z2', style='-k', linewidth=1, fill=synthetic,
                      fillkey='z1', fillcolor='gray', alpha=0.5, label='Basin')
pylab.ylim(1.2*amp, -300)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")
pylab.text(2500, 500, r"$\Delta\rho = -500\ kg.m^{-3}$", fontsize=16,
           horizontalalignment='center')
pylab.legend(loc='lower right', shadow=True)

# and plot the inversion results

# Adjustment X Synthetic data
pylab.subplot(3,2,3)
pylab.title("Adjustment")
pylab.plot(data['gz']['x'], data['gz']['value'], '.k', label="Synthetic")
pylab.plot(adjusted['gz']['x'], adjusted['gz']['value'], '-r', label="Adjusted")
pylab.ylabel("mGal")
pylab.legend(loc='upper left', prop={'size':14}, shadow=True)

# Inversion result
pylab.subplot(3,2,4)
pylab.title("Inversion result")
vis.plot_2d_interface(synthetic, key='z2', style='-r', linewidth=1,
                      label='Synthetic', fill=synthetic, fillkey='z1',
                      fillcolor='r', alpha=0.5)
initial_mesh = fatiando.mesh.copy(mesh)
fatiando.mesh.fill(initial, initial_mesh)
vis.plot_2d_interface(initial_mesh, style='-.g', linewidth=3, label='Initial')
vis.plot_2d_interface(mesh, key='z2', style='-k', linewidth=2, label="Inverted")

pylab.legend(loc='lower right', prop={'size':14}, shadow=True)
pylab.ylim(2500., -200)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")

# Histogram of residuals
pylab.subplot(3,2,5)
pylab.title("Residuals")
vis.residuals_histogram(residuals)
pylab.xlabel("mGal")
pylab.ylabel("Number of occurrences")

# Goal Function X Iteration
pylab.subplot(3,2,6)
pylab.title("Goal function")
pylab.plot(goals, '.-k')
pylab.xlabel("Iteration")

pylab.savefig('interg_smooth.png', dpi=60)

pylab.show()