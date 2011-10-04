"""
Example of generating a topography, creating a 3D prism model of it and
calculating gz
"""
from enthought.mayavi import mlab
import numpy
from matplotlib import pyplot
from fatiando import stats, gridder, logger, vis, potential
from fatiando.mesher.prism import Relief3D, relief2prisms, fill_relief

# Avoid importing mlab twice since it's very slow
vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism model of the topography")

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (100, 50)
x, y = gridder.regular(area, shape)
height = (100 +
          80*stats.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*stats.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
scalars = [2670 for i in xrange(len(height))]
nodes = (x, y, -1*height)
relief = fill_relief(scalars, Relief3D(0,gridder.spacing(area,shape),nodes))

log.info("Calculating gz effect")
gridarea = (-80, 80, -220, 220)
xp, yp, zp = gridder.regular(gridarea, shape, z=-200)
gz = potential.prism.gz(xp, yp, zp, relief2prisms(relief, 'density'))

pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.pcolor(x, y, height, shape)
pyplot.colorbar()
vis.square(gridarea, label='Computation grid')
pyplot.legend()

pyplot.subplot(1,2,2)
pyplot.title("Topographic gz effect")
pyplot.axis('scaled')
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()

vis.prisms3D(relief2prisms(relief), relief['cells'])
mlab.show()
