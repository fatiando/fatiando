"""
Create noise-corrupted synthetic data from a right rectangular prism model.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, mesher, gridder, vis, logger, stats

log = logger.get()
log.info(logger.header())
log.info("Example of generating noise-corrupted synthetic data using prisms")

log.info("Contaminating data with 1 mGal noise...")
prisms = [mesher.RightRectangularPrism(-1000,1000,-1000,1000,0,2000,density=1000)]
shape = (100,100)
xp, yp, zp = gridder.regular(-5000, 5000, -5000, 5000, shape, z=-100)
gz = stats.contaminate(potential.prism.gz(prisms, xp, yp, zp), 1)

log.info("Plotting...")
pyplot.title("gz contaminated with 1 mGal noise")
pyplot.axis('scaled')
levels = vis.contourf(xp, yp, gz, (100,100), 12)
vis.contour(xp, yp, gz, shape, levels)
pyplot.show()
