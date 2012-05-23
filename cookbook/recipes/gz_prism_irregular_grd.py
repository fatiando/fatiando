"""
Generate synthetic gz data on an irregular grid.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, gridder, logger, vis
from fatiando.mesher.ddd import Prism

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Calculating...")
prisms = [Prism(-2000,2000,-2000,2000,0,2000,{'density':1000})]
xp, yp, zp = gridder.scatter((-5000, 5000, -5000, 5000), n=500, z=-100)
gz = potential.prism.gz(xp, yp, zp, prisms)

log.info("Plotting...")
shape = (100,100)
pyplot.axis('scaled')
pyplot.title("gz produced by prism model on an irregular grid (mGal)")
pyplot.plot(xp, yp, '.k', label='Grid points')
levels = vis.map.contourf(xp, yp, gz, shape, 12, interp=True)
vis.map.contour(xp, yp, gz, shape, levels, interp=True)
pyplot.legend(loc='lower right', numpoints=1)
pyplot.show()
