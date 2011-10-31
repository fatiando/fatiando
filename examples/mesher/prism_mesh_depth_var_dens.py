"""
Example of generating a prism mesh and extracting a submesh
"""
from enthought.mayavi import mlab
from fatiando import logger, vis
from fatiando.mesher.prism import Mesh3D, fill_mesh, vfilter, mesh2prisms, extract

vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a prism mesh with depth varying density")

shape = (10, 20, 10)
nz, ny, nx = shape
density = 1
def fill(i):
    return i/(nx*ny)
scalars = [fill(i) for i in xrange(shape[0]*shape[1]*shape[2])]
mesh = Mesh3D(0, 100, 0, 200, 0, 50, shape, scalars)

odd = vfilter(-1, 0, 'density', mesh2prisms(mesh, prop='density'))
even = vfilter(0, 1, 'density', mesh2prisms(mesh, prop='density'))

mlab.figure()
vis.prisms3D(mesh2prisms(mesh), mesh['cells'])

mlab.show()
