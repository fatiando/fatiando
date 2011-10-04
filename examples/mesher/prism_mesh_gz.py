"""
Example of generating a 3D prism mesh and calculating its gravitational effect
"""
from enthought.mayavi import mlab
import numpy
from matplotlib import pyplot
from fatiando import gridder, logger, vis, potential
from fatiando.mesher.prism import Mesh3D, mesh2prisms, fill_mesh

# Avoid importing mlab twice since it's very slow
vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh and calculating gz")

log.info("Generating the prism mesh")
meshshape = (10,40,20)
nz, ny, nx = meshshape
x1, x2 = -100, 100
y1, y2 = -200, 200
scalars = [2670 for i in xrange(nx*ny*nz)]
mesh = fill_mesh(scalars, Mesh3D(x1, x2, y1, y2, -200, 0, meshshape))

log.info("The computation grid:")
gridshape = (40, 20)
area = (-50, 50, -100, 100)
xp, yp, zp = gridder.regular(area, gridshape, z=-250)
gz = potential.prism.gz(xp, yp, zp, mesh2prisms(mesh, 'density'))

pyplot.figure()
pyplot.title("gz")
pyplot.axis('scaled')
vis.pcolor(xp, yp, gz, gridshape)
pyplot.colorbar()
pyplot.show()

vis.prisms3D(mesh2prisms(mesh), mesh['cells'])
mlab.show()
