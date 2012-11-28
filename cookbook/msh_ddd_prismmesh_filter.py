"""
Meshing: Filter prisms from a 3D prism mesh based on their physical properties
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

shape = (5, 20, 10)
bounds = (0, 100, 0, 200, 0, 50)
mesh = ft.msh.ddd.PrismMesh(bounds, shape)
# Fill the even prisms with 1 and odd with -1
def fill(i):
    if i%2 == 0:
        return 1
    return -1
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

# Separate even and odd prisms
odd = ft.msh.ddd.vfilter(-1, 0, 'density', mesh)
even = ft.msh.ddd.vfilter(0, 1, 'density', mesh)

log.info("Showing solid ODD prisms and wireframe EVEN")
ft.vis.figure3d()
ft.vis.prisms(odd, prop='density', vmin=-1, vmax=1)
ft.vis.prisms(even, prop='density', style='wireframe', vmin=-1, vmax=1)
ft.vis.axes3d(ft.vis.outline3d(bounds))
ft.vis.show3d()
