"""
Calculate the gravity gradient tensor components from g_z using an equivalent
layer.
"""

import pylab
import sys

from fatiando.grav import synthetic, io, eqlayer
from fatiando import utils, vis, grid, geometry

log = utils.get_logger()
log.info(utils.header())

# Make a synthetic data from a sphere model
log.info("########################################")
log.info("Generating synthetic data")

spheres = []
spheres.append(geometry.sphere(xc=1000, yc=1000, zc=1000, radius=800,
                               props={'density':500}))

data = grid.regular(0, 2000, 0, 2000, 25, 25, 0)

synthetic.from_spheres(spheres, data)

data['value'] = utils.contaminate(data['value'], stddev=0.1, percent=False)

# Calculate the equivalent layer
log.info("########################################")
log.info("Generating equivalent layer:")

layer = grid.regular(-100, 2100, -100, 2100, 30, 30, 1000)

residuals = eqlayer.generate(layer, data, damping=10**(-30), smoothness=0*10**(-21))

# Calculate the adjustment
adjusted = grid.copy(data)

eqlayer.calculate(layer, adjusted)

# Calculate the tensor components
log.info("########################################")
log.info("Calculating components")

components = ['gxx', 'gyy', 'gzz']

for comp in components:

    cp_true = grid.regular(0, 2000, 0, 2000, 25, 25, 0)

    synthetic.from_spheres(spheres, cp_true, field=comp)

    cp_eqlayer = grid.copy(cp_true)

    eqlayer.calculate(layer, cp_eqlayer, field=comp)

    pylab.figure()
    pylab.title(comp)
    pylab.axis("scaled")
    levels = vis.contour(cp_true, 10, color='b')
    vis.contour(cp_eqlayer, levels, color='r')

# Plot the results
log.info("########################################")
log.info("Plotting results")

pylab.figure()
pylab.title("Data at 0 m")
pylab.axis("scaled")
vis.contourf(data, 10)
cb = pylab.colorbar()
cb.set_label("mGal")

pylab.figure()
pylab.title("Residuals")
vis.residuals_histogram(residuals)
pylab.xlabel("mGal")
pylab.ylabel("Number of")

pylab.figure()
pylab.title("Equivalent Layer")
pylab.axis('scaled')
vis.pcolor(layer)
cb = pylab.colorbar()
cb.set_label("Density [kg/m^3]")

pylab.figure()
pylab.title("Adjustment")
pylab.axis("scaled")
levels = vis.contour(data, 10, color='b')
vis.contour(adjusted, levels, color='r')

pylab.show()