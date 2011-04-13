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
spheres.append(geometry.sphere(xc=4000, yc=1000, zc=500, radius=300,
                               props={'density':1000}))
spheres.append(geometry.sphere(xc=4000, yc=4000, zc=1500, radius=1000,
                               props={'density':800}))
spheres.append(geometry.sphere(xc=1000, yc=4000, zc=1000, radius=700,
                               props={'density':-800}))
spheres.append(geometry.sphere(xc=2500, yc=2500, zc=1200, radius=1000,
                               props={'density':1200}))

data = grid.regular(0, 5000, 0, 5000, 25, 25, 0)

synthetic.from_spheres(spheres, data)

data['value'] = utils.contaminate(data['value'], stddev=0.2, percent=False)

# Calculate the equivalent layer
log.info("########################################")
log.info("Generating equivalent layer:")

layer = grid.regular(-500, 5500, -500, 5500, 30, 30, 1000)

residuals = eqlayer.generate(layer, data, damping=10**(-30), smoothness=10**(-22))
#residuals = eqlayer.generate(layer, data, damping=10**(-25), smoothness=0)

# Calculate the adjustment
adjusted = grid.copy(data)

eqlayer.calculate(layer, adjusted)

# Plot the adjustment and data
pylab.figure(figsize=(14,10))

pylab.subplot(2,2,1)
pylab.title("Data: gz")
pylab.axis("scaled")
levels = vis.contourf(data, 15)
vis.contour(data, levels)

pylab.subplot(2,2,2)
pylab.title("Equivalent Layer")
pylab.axis('scaled')
vis.pcolor(layer)
cb = pylab.colorbar()
cb.set_label("Density [kg/m^3]")

pylab.subplot(2,2,3)
pylab.title("Adjustment: gz")
pylab.axis("scaled")
levels = vis.contour(data, 15, color='b')
vis.contour(adjusted, levels, color='r')

pylab.subplot(2,2,4)
pylab.title("Residuals")
vis.residuals_histogram(residuals)
pylab.xlabel("mGal")
pylab.ylabel("Number of")

# Calculate the tensor components
log.info("########################################")
log.info("Calculating components")

components = ['gxx', 'gyy', 'gzz']

pylab.figure(figsize=(16,6))

for i, comp in enumerate(components):

    cp_true = grid.copy(data)

    synthetic.from_spheres(spheres, cp_true, field=comp)

    cp_eqlayer = grid.copy(cp_true)

    eqlayer.calculate(layer, cp_eqlayer, field=comp)

    pylab.subplot(1, 3, i + 1)
    pylab.title(comp)
    pylab.axis("scaled")
    levels = vis.contour(cp_true, 10, color='b')
    vis.contour(cp_eqlayer, levels, color='r')

pylab.show()