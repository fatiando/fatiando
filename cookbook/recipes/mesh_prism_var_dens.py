"""
Making a 3D prism mesh with depth-varying density.
"""
from fatiando import logger, vis
from fatiando.mesher.volume import PrismMesh3D

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

vis.mayavi_figure()
plot = vis.prisms3D(mesh, mesh.props['density'])
axes = vis.add_axes3d(plot)
vis.mlab.show()
