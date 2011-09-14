"""
Create noise-corrupted synthetic data from a right rectangular prism model.
"""
from matplotlib import pyplot as mpl
import numpy
from fatiando import potential, mesher, gridder, vis, logger, stats

log = logger.get()
log.info(logger.header())
log.info("Example of generating noise-corrupted synthetic data using prisms")

log.info("Contaminating data with 1 mGal noise...")
prisms = [mesher.RightRectangularPrism(-1000,1000,-1000,1000,0,2000,density=1000)]
xp, yp, zp = gridder.regular(-5000, 5000, -5000, 5000, 100, 100, z=-100)
gz = stats.contaminate(potential.prism.gz(prisms, xp, yp, zp), 1)

log.info("Plotting...")
mpl.title("gz contaminated with 1 mGal noise")
mpl.axis('scaled')
levels = vis.contourf(xp, yp, gz, (100,100), 12)
vis.contour(xp, yp, gz, (100,100), levels)
mpl.show()
