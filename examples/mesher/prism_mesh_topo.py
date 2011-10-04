"""
Example of generating a 3D prism mesh with topography
"""
from enthought.mayavi import mlab
import numpy
from matplotlib import pyplot
from fatiando import stats, gridder, logger, vis
from fatiando.mesher.prism import Mesh3D, flagtopo, mesh2prisms

# Avoid importing mlab twice since it's very slow
vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh with topography")

x1, x2 = -100, 100
y1, y2 = -200, 200

log.info("Generating synthetic topography")
x, y = gridder.regular((x1, x2, y1, y2), (50,50))
height = (100 +
          80*stats.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*stats.gaussian2d(x, y, 50, 100, x0=80, y0=170))

pyplot.figure()
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.pcolor(x, y, height, (50,50))
pyplot.colorbar()
pyplot.show()

log.info("Generating the prism mesh")
mesh = flagtopo(x, y, height, Mesh3D(x1, x2, y1, y2, -200, 0, (20,40,20)))
vis.prisms3D(mesh2prisms(mesh), mesh['cells'])
mlab.show()
