"""
Example of generating a prism mesh and filtering elements by density value
"""
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab
from fatiando import logger, vis
from fatiando.mesher.volume import PrismMesh3D, vfilter, extract

vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Generating prism mesh with alternating density contrast")

shape = (5, 20, 10)
bounds = (0, 100, 0, 200, 0, 50)
mesh = PrismMesh3D(bounds, shape)
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
mlab.figure(bgcolor=(1,1,1))
vis.prisms3D(odd, extract('density', odd))
plot = vis.prisms3D(even, extract('density', even), style='wireframe')
vis.add_axes3d(plot, extent=bounds)
mlab.show()
