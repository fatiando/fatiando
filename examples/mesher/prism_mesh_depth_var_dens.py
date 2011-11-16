"""
Example of generating a prism mesh with depth varying density
"""
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab
from fatiando import logger, vis
from fatiando.mesher.volume import PrismMesh3D

vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a prism mesh with depth varying density")

shape = (10, 20, 10)
nz, ny, nx = shape
mesh = PrismMesh3D((0, 100, 0, 200, 0, 50), shape)
def fill(i):
    k = i/(nx*ny) 
    return k
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

mlab.figure(bgcolor=(1,1,1))
plot = vis.prisms3D(mesh, mesh.props['density'])
axes = vis.add_axes3d(plot)
mlab.show()
