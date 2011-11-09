try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab
from fatiando import logger, vis
from fatiando.mesher.prism import PrismMesh3D, vfilter, extract

vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a prism mesh and filtering it")

shape = (5, 20, 10)
mesh = PrismMesh3D(0, 100, 0, 200, 0, 50, shape)
# Fill the even prisms with 1 and odd with -1
def fill(i):
    if i%2 == 0:
        return 1
    return -1
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

log.info("Showing solid ODD prisms and wireframe EVEN")
mlab.figure()
odd = vfilter(-1, 0, 'density', mesh)
even = vfilter(0, 1, 'density', mesh)
vis.prisms3D(odd, extract('density', odd))
vis.prisms3D(even, extract('density', even), style='wireframe')

mlab.show()
