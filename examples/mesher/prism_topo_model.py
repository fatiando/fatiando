"""
Example of generating a 3D prism mesh with topography and calculating its
gravitational effect
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
          50*stats.gaussian2d(x, y, -50, -100, cov=[[5000,5000],[5000,30000]]) +
          80*stats.gaussian2d(x, y, 80, 170, cov=[[5000,0],[0,3000]]))

log.info("Generating the 3D relief")
scalars = [2670 for i in xrange(len(height))]
nodes = (x,y,-1*height)
relief = fill_relief(scalars,Relief3D(0,gridder.spacing(area,shape),nodes))

log.info("Calculating gz effect")
xp, yp, zp = gridder.regular((-80,80,-150,250) , shape, z=-200)
gz = potential.prism.gz(xp, yp, zp, relief2prisms(relief, 'density'))

pyplot.figure()
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.pcolor(x, y, height, shape)
pyplot.colorbar()

pyplot.figure()
pyplot.title("Topographic gz effect")
pyplot.axis('scaled')
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()

vis.prisms3D(relief2prisms(relief), relief['cells'])
mlab.show()
