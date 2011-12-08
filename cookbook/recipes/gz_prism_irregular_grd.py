"""
Generate synthetic gz data on an irregular grid.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, mesher, gridder, logger, vis

log = logger.get()
log.info(logger.header())
log.info("Example of generating and plotting irregular grids")

log.info("Calculating...")
prisms = [mesher.volume.Prism3D(-2000,2000,-2000,2000,0,2000,{'density':1000})]
xp, yp, zp = gridder.scatter((-5000, 5000, -5000, 5000), n=500, z=-100)
gz = potential.prism.gz(xp, yp, zp, prisms)

log.info("Plotting...")
shape = (100,100)
pyplot.axis('scaled')
pyplot.title("gz produced by prism model on an irregular grid (mGal)")
pyplot.plot(xp, yp, '.k', label='Grid points')
levels = vis.contourf(xp, yp, gz, shape, 12, interpolate=True)
vis.contour(xp, yp, gz, shape, levels, interpolate=True)
pyplot.legend(loc='lower right')
pyplot.show()
