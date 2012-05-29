"""
Make and plot a 3D prism mesh with depth-varying density.
"""
from fatiando import logger, vis
from fatiando.mesher.ddd import PrismMesh

log = logger.get()
log.info(logger.header())
log.info(__doc__)

shape = (10, 20, 10)
nz, ny, nx = shape
mesh = PrismMesh((0, 100, 0, 200, 0, 50), shape)
def fill(i):
    k = i/(nx*ny) 
    return k
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

vis.vtk.figure()
vis.vtk.prisms(mesh, prop='density')
vis.vtk.add_axes(vis.vtk.add_outline(), fmt='%.0f')
vis.vtk.mlab.show()
