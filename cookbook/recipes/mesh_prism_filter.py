"""
Filter values from a 3D prism mesh.
"""
from fatiando import logger, vis
from fatiando.mesher.volume import PrismMesh3D, vfilter, extract

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
vis.mayavi_figure()
vis.prisms3D(odd, extract('density', odd), vmin=-1, vmax=1)
plot = vis.prisms3D(even, extract('density', even), style='wireframe', vmin=-1,
                    vmax=1)
vis.add_axes3d(plot, extent=bounds)
vis.mlab.show()
