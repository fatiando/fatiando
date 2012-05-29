"""
Make and plot a 3D prism mesh.
"""
from fatiando import mesher, logger, vis
from fatiando.mesher.ddd import PrismMesh

log = logger.get()
log.info(logger.header())
log.info(__doc__)

mesh = PrismMesh(bounds=(-2, 2, -3, 3, 0, 1), shape=(4,4,4))

vis.vtk.figure()
plot = vis.vtk.prisms(mesh)
axes = vis.vtk.add_axes(plot)
vis.vtk.mlab.show()
