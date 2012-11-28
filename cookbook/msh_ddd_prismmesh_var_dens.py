"""
Meshing: Make a 3D prism mesh with depth-varying density
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

shape = (10, 20, 10)
nz, ny, nx = shape
mesh = ft.mesher.PrismMesh((0, 100, 0, 200, 0, 50), shape)
def fill(i):
    k = i/(nx*ny)
    return k
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

ft.vis.figure3d()
ft.vis.prisms(mesh, prop='density')
ft.vis.axes3d(ft.vis.outline3d(), fmt='%.0f')
ft.vis.show3d()
