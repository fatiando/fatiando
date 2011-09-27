"""
Example of generating a 3D prism mesh with topography and calculating its
gravitational effect
"""
from enthought.mayavi import mlab
import numpy
from matplotlib import pyplot
from fatiando import stats, gridder, logger, vis, potential
from fatiando.mesher.prism import Mesh3D, flagtopo, mesh2prisms, fill_mesh

# Avoid importing mlab twice since it's very slow
vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh with topography")

x1, x2 = -100, 100
y1, y2 = -200, 200

log.info("Generating synthetic topography")
gridshape = (40, 20)
x, y = gridder.regular((x1, x2, y1, y2), gridshape)
height = (100 +
          50*stats.gaussian2d(x, y, -50, -50, cov=[[5000,15000],[5000,50000]]))

log.info("Generating the prism mesh")
shape = (10,40,20)
nz, ny, nx = shape
scalars = [2670 for i in xrange(nx*ny*nz)]
mesh = fill_mesh(scalars,flagtopo(x,y,height,Mesh3D(x1,x2,y1,y2,-200,0,shape)))
gz = potential.prism.gz(x, y, -250*numpy.ones_like(x), mesh2prisms(mesh, 'density'))

pyplot.figure()
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.pcolor(x, y, height, gridshape)
pyplot.colorbar()

pyplot.figure()
pyplot.title("Topographic gz effect")
pyplot.axis('scaled')
vis.pcolor(x, y, gz, gridshape)
pyplot.colorbar()
pyplot.show()

vis.prisms3D(mesh2prisms(mesh), mesh['cells'])
mlab.show()
