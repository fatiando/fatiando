"""
Perform upward continuation using an equivalent layer.
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

residuals = eqlayer.generate(layer, data, damping=10**(-30), smoothness=10**(-20))

# Calculate the adjustment
adjusted = grid.copy(data)

eqlayer.calculate(layer, adjusted)

# Now upward continue the data
newheight = -1000

log.info("########################################")
log.info("Upward continuing to %g m:" % (-newheight))

up_true = grid.regular(0, 2000, 0, 2000, 25, 25, newheight)

synthetic.from_spheres(spheres, up_true)

up_eqlayer = grid.copy(up_true)

eqlayer.calculate(layer, up_eqlayer)

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

pylab.figure()
pylab.title("Upward continued to %g m" % (-newheight))
pylab.axis("scaled")
levels = vis.contour(up_true, 10, color='b')
vis.contour(up_eqlayer, levels, color='r')

pylab.show()