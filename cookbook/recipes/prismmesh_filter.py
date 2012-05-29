"""
Filter values from a 3D prism mesh with alternating density contrasts.
"""
from fatiando import logger, vis
from fatiando.mesher.ddd import PrismMesh, vfilter, extract

log = logger.get()
log.info(logger.header())
log.info(__doc__)

shape = (5, 20, 10)
bounds = (0, 100, 0, 200, 0, 50)
mesh = PrismMesh(bounds, shape)
# Fill the even prisms with 1 and odd with -1
def fill(i):
    if i%2 == 0:
        return 1
    return -1
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

# Separate even and odd prisms
odd = vfilter(-1, 0, 'density', mesh)
even = vfilter(0, 1, 'density', mesh)

log.info("Showing solid ODD prisms and wireframe EVEN")
vis.vtk.figure()
vis.vtk.prisms(odd, prop='density', vmin=-1, vmax=1)
vis.vtk.prisms(even, prop='density', style='wireframe', vmin=-1, vmax=1)
vis.vtk.add_axes(vis.vtk.add_outline(bounds))
vis.vtk.mlab.show()
