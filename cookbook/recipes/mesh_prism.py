"""
Making a 3D prism mesh.
"""
from fatiando import mesher, logger, vis

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh")

mesh = mesher.volume.PrismMesh3D((-2, 2, -3, 3, 0, 1), (4,4,4))

vis.mayavi_figure()
plot = vis.prisms3D(mesh, [0 for i in xrange(mesh.size)])
axes = vis.add_axes3d(plot)
vis.mlab.show()
