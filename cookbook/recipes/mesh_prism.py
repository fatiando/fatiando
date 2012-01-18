"""
Making a 3D prism mesh.
"""
from fatiando import mesher, logger, vis
from fatiando.mesher.ddd import PrismMesh

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh")

mesh = PrismMesh(bounds=(-2, 2, -3, 3, 0, 1), shape=(4,4,4))

vis.mayavi_figure()
plot = vis.prisms3D(mesh, [0 for i in xrange(mesh.size)])
axes = vis.add_axes3d(plot)
vis.mlab.show()
