"""
Generate noise-corrupted synthetic gz data from a prism model.
"""
from matplotlib import pyplot
from fatiando import potential, gridder, vis, logger, utils
from fatiando.mesher.ddd import Prism

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Calculating...")
prisms = [Prism(-1000,1000,-1000,1000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-100)
gz = utils.contaminate(potential.prism.gz(xp, yp, zp, prisms), 0.05,
                       percent=True)

log.info("Plotting...")
pyplot.title("gz contaminated with 1 mGal noise")
pyplot.axis('scaled')
levels = vis.map.contourf(xp, yp, gz, (100,100), 12)
vis.map.contour(xp, yp, gz, shape, levels)
pyplot.show()
