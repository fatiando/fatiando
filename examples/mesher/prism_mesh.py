"""
Example of how to generate a regular mesh of 3D prisms and plot it in Mayavi2
"""
from mayavi import mlab
from fatiando import mesher, logger, vis

# vis does a lazy import of mlab. So we can avoid importing mlab twice since
# it's very slow
vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a 3D prism mesh")

mesh = mesher.volume.PrismMesh3D((-2, 2, -3, 3, 0, 1), (4,4,4))

mlab.figure(bgcolor=(1,1,1))
plot = vis.prisms3D(mesh, [0 for i in xrange(mesh.size)])
axes = vis.add_axes3d(plot)
mlab.show()
